#!/usr/bin/env python
"""X-MAG-IDS first runnable research pipeline.

This script intentionally keeps the first reviewer-facing implementation in one
file. It supports schema audit, synthetic smoke-test generation, known/unknown
splitting, transparent multi-agent partitioning, lightweight local models,
raw/logit/explanation-token communication baselines, and result export.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LABEL_CANDIDATES = ["Attack Type", "Attack_Type", "attack_type", "Label", "label", "class", "Class"]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def generate_synthetic(rows=1200, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["Benign", "UDP Flood", "HTTP Flood", "SYN Flood", "UDP Scan"], size=rows, p=[0.45, 0.20, 0.15, 0.10, 0.10])
    df = pd.DataFrame({
        "Flow Duration": rng.gamma(2.0, 1200.0, rows),
        "Tot Fwd Pkts": rng.poisson(12, rows).astype(float),
        "Tot Bwd Pkts": rng.poisson(9, rows).astype(float),
        "Fwd Pkt Len Mean": rng.normal(300, 50, rows),
        "Bwd Pkt Len Mean": rng.normal(280, 60, rows),
        "Flow IAT Mean": rng.gamma(2.0, 10.0, rows),
        "SYN Flag Cnt": rng.poisson(0.5, rows).astype(float),
        "ACK Flag Cnt": rng.poisson(4, rows).astype(float),
        "Pkt Len Var": rng.gamma(2.5, 80.0, rows),
        "Dst Port Entropy": rng.normal(0.25, 0.08, rows),
    })
    for attack in np.unique(labels):
        mask = labels == attack
        if attack == "SYN Flood":
            df.loc[mask, "SYN Flag Cnt"] += rng.poisson(15, mask.sum())
            df.loc[mask, "Flow IAT Mean"] *= 0.35
        elif attack == "UDP Flood":
            df.loc[mask, "Tot Fwd Pkts"] += rng.poisson(80, mask.sum())
            df.loc[mask, "Flow Duration"] *= 0.55
        elif attack == "HTTP Flood":
            df.loc[mask, "Tot Fwd Pkts"] += rng.poisson(35, mask.sum())
            df.loc[mask, "ACK Flag Cnt"] += rng.poisson(20, mask.sum())
        elif attack == "UDP Scan":
            df.loc[mask, "Dst Port Entropy"] += rng.normal(0.55, 0.08, mask.sum())
            df.loc[mask, "Tot Bwd Pkts"] *= 0.30
    df["Attack Type"] = labels
    df["Label"] = np.where(labels == "Benign", "Benign", "Malicious")
    df["Attack Tool"] = np.where(labels == "Benign", "None", "synthetic_tool")
    df["sVid"] = rng.integers(1, 4, rows)
    df["dVid"] = rng.integers(1, 4, rows)
    df["54"] = rng.normal(0, 1, rows)
    return df


def infer_label_column(columns, configured=None):
    if configured and configured in columns:
        return configured
    if configured and configured not in columns:
        raise ValueError(f"Configured label column not found: {configured}")
    for c in LABEL_CANDIDATES:
        if c in columns:
            return c
    raise ValueError("Could not infer label column; set dataset.label_column")


def leakage_columns(df, label_col, drop_columns, patterns):
    out = {label_col}
    for c in drop_columns or []:
        if c in df.columns:
            out.add(c)
    pats = [str(p).lower() for p in (patterns or [])]
    for c in df.columns:
        norm = c.lower().replace("_", " ")
        if any(p in norm for p in pats):
            out.add(c)
    return [c for c in df.columns if c in out]


def clean_features(df, drop_cols):
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
    keep = X.nunique(dropna=False)
    X = X[keep[keep > 1].index.tolist()]
    return X.replace([np.inf, -np.inf], np.nan)


def hash_agents(X, y_seed, n_agents):
    if len(X) == 0:
        return np.array([], dtype=int)
    seed = pd.Series(y_seed).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(seed, index=False).to_numpy(dtype="uint64")
    return (hashed % np.uint64(n_agents)).astype(int)


def make_preprocessor(X):
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", enc)]), categorical))
    if not transformers:
        raise ValueError("No usable features remain after leakage control")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def prepare(df, cfg):
    exp = cfg["experiment"]
    ds = cfg["dataset"]
    ag = cfg["agents"]
    label_col = infer_label_column(df.columns, ds.get("label_column"))
    labels = df[label_col].astype(str).str.strip()
    unknown = ds.get("unknown_attack")
    if unknown and unknown not in set(labels):
        raise ValueError(f"Unknown holdout '{unknown}' not present in labels")
    drops = leakage_columns(df, label_col, ds.get("drop_columns", []), ds.get("leakage_patterns", []))
    X_all = clean_features(df, drops)
    unknown_mask = labels.eq(unknown) if unknown else pd.Series(False, index=df.index)
    X_known, y_known = X_all.loc[~unknown_mask].reset_index(drop=True), labels.loc[~unknown_mask].reset_index(drop=True)
    X_unknown, y_unknown = X_all.loc[unknown_mask].reset_index(drop=True), labels.loc[unknown_mask].reset_index(drop=True)
    strat = y_known if y_known.nunique() > 1 and y_known.value_counts().min() >= 2 else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_known, y_known, test_size=exp.get("test_size", 0.3), random_state=exp.get("random_state", 42), stratify=strat)
    strat2 = y_trainval if y_trainval.nunique() > 1 and y_trainval.value_counts().min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=exp.get("validation_size", 0.2), random_state=exp.get("random_state", 42), stratify=strat2)
    n_agents = ag.get("n_agents", 8)
    classes = sorted(y_train.unique().tolist())
    summary = {
        "label_column": label_col,
        "unknown_attack": unknown,
        "dropped_columns": drops,
        "feature_count": int(X_all.shape[1]),
        "known_classes": classes,
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "n_test_known": int(len(X_test)),
        "n_test_unknown": int(len(X_unknown)),
        "train_class_counts": y_train.value_counts().to_dict(),
        "test_class_counts": y_test.value_counts().to_dict(),
        "unknown_class_counts": y_unknown.value_counts().to_dict(),
    }
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True), hash_agents(X_train, y_train, n_agents), X_val.reset_index(drop=True), y_val.reset_index(drop=True), hash_agents(X_val, y_val, n_agents), X_test.reset_index(drop=True), y_test.reset_index(drop=True), hash_agents(X_test, y_test, n_agents), X_unknown.reset_index(drop=True), y_unknown.reset_index(drop=True), hash_agents(X_unknown, y_unknown, n_agents), classes, summary


def rf(cfg, seed):
    m = cfg.get("model", {})
    return RandomForestClassifier(n_estimators=m.get("n_estimators", 100), min_samples_leaf=m.get("min_samples_leaf", 1), class_weight=m.get("class_weight", "balanced_subsample"), random_state=seed, n_jobs=1)


def align_proba(model, X, classes):
    raw = model.predict_proba(X)
    out = np.zeros((X.shape[0], len(classes)))
    local = [str(c) for c in model.classes_]
    for i, lab in enumerate(local):
        if lab in classes:
            out[:, classes.index(lab)] = raw[:, i]
    sums = out.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return out / sums


def train_agents(base, X, y, agent_ids, n_agents, classes):
    agents = []
    y_arr = np.asarray(y).astype(str)
    for aid in range(n_agents):
        mask = agent_ids == aid
        if mask.sum() == 0:
            model = DummyClassifier(strategy="prior").fit(X, y_arr)
        elif np.unique(y_arr[mask]).size < 2:
            model = DummyClassifier(strategy="constant", constant=y_arr[mask][0]).fit(X[mask], y_arr[mask])
        else:
            model = clone(base).fit(X[mask], y_arr[mask])
        agents.append((aid, model))
    return agents


def owner_proba(agents, X, owner_ids, classes):
    out = np.zeros((X.shape[0], len(classes)))
    for aid, model in agents:
        mask = owner_ids == aid
        if np.any(mask):
            out[mask] = align_proba(model, X[mask], classes)
    return out


def average_proba(agents, X, classes):
    return np.mean(np.stack([align_proba(m, X, classes) for _, m in agents], axis=0), axis=0)


def majority_proba(agents, X, classes):
    votes = np.zeros((X.shape[0], len(classes)))
    for _, m in agents:
        p = align_proba(m, X, classes)
        winners = np.argmax(p, axis=1)
        votes[np.arange(X.shape[0]), winners] += 1
    return votes / max(len(agents), 1)


def feature_importance(model, n_features):
    if hasattr(model, "feature_importances_"):
        v = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        v = np.mean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
    else:
        v = np.ones(n_features)
    if v.shape[0] != n_features:
        v = np.resize(v, n_features)
    s = np.sum(np.abs(v))
    return v / s if s > 0 else np.ones(n_features) / n_features


def explanation_tokens(X, model, k, overhead=8, fid_bytes=2, value_bytes=4):
    k = int(max(1, min(k, X.shape[1])))
    contrib = X * feature_importance(model, X.shape[1]).reshape(1, -1)
    dense = np.zeros_like(contrib)
    for r in range(X.shape[0]):
        idx = np.argpartition(np.abs(contrib[r]), -k)[-k:]
        dense[r, idx] = contrib[r, idx]
    return dense, int(overhead + k * (fid_bytes + value_bytes))


def unknown_scores(p):
    return np.array([]) if p.size == 0 else 1 - np.max(p, axis=1)


def eval_method(method, k, pv, pk, pu, yk, classes, msg_bytes, train_s, pred_s):
    pred = np.asarray(classes)[np.argmax(pk, axis=1)]
    kv = unknown_scores(pv)
    kk = unknown_scores(pk)
    ku = unknown_scores(pu)
    if ku.size:
        threshold = float(np.quantile(kv if kv.size else kk, 0.95))
        y_open = np.r_[np.zeros_like(kk, dtype=int), np.ones_like(ku, dtype=int)]
        scores = np.r_[kk, ku]
        try:
            auroc = float(roc_auc_score(y_open, scores))
        except ValueError:
            auroc = float("nan")
        urec = float(recall_score(np.ones_like(ku), (ku >= threshold).astype(int), zero_division=0))
        kfar = float(np.mean(kk >= threshold)) if kk.size else float("nan")
    else:
        threshold, auroc, urec, kfar = float("nan"), float("nan"), float("nan"), float("nan")
    return {
        "method": method,
        "k": k,
        "message_bytes_per_flow": int(msg_bytes),
        "train_seconds": float(train_s),
        "predict_seconds": float(pred_s),
        "known_accuracy": float(accuracy_score(yk, pred)),
        "known_balanced_accuracy": float(balanced_accuracy_score(yk, pred)),
        "known_macro_f1": float(f1_score(yk, pred, average="macro", zero_division=0)),
        "known_weighted_f1": float(f1_score(yk, pred, average="weighted", zero_division=0)),
        "unknown_auroc": auroc,
        "unknown_recall_at_threshold": urec,
        "known_false_alarm_rate_at_threshold": kfar,
        "unknown_threshold": threshold,
    }


def coordinator():
    return OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))


def run_experiment(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    exp = cfg["experiment"]
    outdir = ensure_dir(exp["output_dir"])
    df = pd.read_csv(cfg["dataset"]["path"], nrows=exp.get("max_rows"), low_memory=False)
    parts = prepare(df, cfg)
    Xtr, ytr, atr, Xv, yv, av, Xk, yk, ak, Xu, yu, au, classes, summary = parts
    write_json(outdir / "split_summary.json", summary)
    pre = make_preprocessor(Xtr)
    Xtrt = pre.fit_transform(Xtr)
    Xvt = pre.transform(Xv)
    Xkt = pre.transform(Xk)
    Xut = pre.transform(Xu)
    try:
        names = [str(x) for x in pre.get_feature_names_out()]
    except Exception:
        names = [f"f{i}" for i in range(Xtrt.shape[1])]
    pd.DataFrame({"feature": names}).to_csv(outdir / "transformed_features.csv", index=False)
    results = []
    seed = exp.get("random_state", 42)

    central = rf(cfg, seed)
    t = time.perf_counter(); central.fit(Xtrt, ytr.astype(str)); train_s = time.perf_counter() - t
    t = time.perf_counter(); pv = align_proba(central, Xvt, classes); pk = align_proba(central, Xkt, classes); pu = align_proba(central, Xut, classes) if len(Xut) else np.empty((0, len(classes))); pred_s = time.perf_counter() - t
    results.append(eval_method("centralized_full_features", None, pv, pk, pu, yk, classes, 0, train_s, pred_s))

    t = time.perf_counter(); agents = train_agents(rf(cfg, seed), Xtrt, ytr, atr, cfg["agents"].get("n_agents", 8), classes); agent_train = time.perf_counter() - t
    t = time.perf_counter(); pv = owner_proba(agents, Xvt, av, classes); pk = owner_proba(agents, Xkt, ak, classes); pu = owner_proba(agents, Xut, au, classes) if len(Xut) else np.empty((0, len(classes))); pred_s = time.perf_counter() - t
    results.append(eval_method("local_owner_only", None, pv, pk, pu, yk, classes, 0, agent_train, pred_s))

    for name, func, msg in [("majority_vote_agents", majority_proba, 4), ("logit_average_agents", average_proba, len(classes) * 4)]:
        t = time.perf_counter(); pv = func(agents, Xvt, classes); pk = func(agents, Xkt, classes); pu = func(agents, Xut, classes) if len(Xut) else np.empty((0, len(classes))); pred_s = time.perf_counter() - t
        results.append(eval_method(name, None, pv, pk, pu, yk, classes, msg, agent_train, pred_s))

    owner_models = {aid: model for aid, model in agents}
    for k in cfg["explanation"].get("k_values", [1, 3, 5]):
        imp = feature_importance(central, Xtrt.shape[1])
        idx = np.argpartition(np.abs(imp), -k)[-k:]
        t = time.perf_counter(); rawc = coordinator().fit(Xtrt[:, idx], ytr.astype(str)); train_s = time.perf_counter() - t
        t = time.perf_counter(); pv = rawc.predict_proba(Xvt[:, idx]); pk = rawc.predict_proba(Xkt[:, idx]); pu = rawc.predict_proba(Xut[:, idx]) if len(Xut) else np.empty((0, len(classes))); pred_s = time.perf_counter() - t
        results.append(eval_method("raw_topk_coordinator", int(k), pv, pk, pu, yk, classes, 8 + int(k) * 4, train_s, pred_s))

        def owner_tokens(X, owners):
            dense = np.zeros_like(X)
            msg = 8 + int(k) * 6
            for aid, _ in agents:
                mask = owners == aid
                if np.any(mask):
                    tok, msg = explanation_tokens(X[mask], owner_models[aid], k)
                    dense[mask] = tok
            return dense, msg
        t = time.perf_counter(); Xtt, msg = owner_tokens(Xtrt, atr); tokc = coordinator().fit(Xtt, ytr.astype(str)); train_s = time.perf_counter() - t
        t = time.perf_counter(); Xvv, _ = owner_tokens(Xvt, av); Xkk, _ = owner_tokens(Xkt, ak); Xuu, _ = owner_tokens(Xut, au) if len(Xut) else (np.empty((0, Xtrt.shape[1])), msg); pv = tokc.predict_proba(Xvv); pk = tokc.predict_proba(Xkk); pu = tokc.predict_proba(Xuu) if len(Xuu) else np.empty((0, len(classes))); pred_s = time.perf_counter() - t
        results.append(eval_method("explanation_token_coordinator", int(k), pv, pk, pu, yk, classes, msg, train_s, pred_s))

    out = pd.DataFrame(results)
    out.to_csv(outdir / "results.csv", index=False)
    write_json(outdir / "results.json", out.to_dict(orient="records"))
    write_json(outdir / "run_metadata.json", {"config": str(config_path), "classes": classes, "n_features": len(names)})
    print(out.to_string(index=False))
    print(f"Wrote outputs to {outdir}")
    return out


def audit(csv, outdir, max_rows=None):
    outdir = ensure_dir(outdir)
    df = pd.read_csv(csv, nrows=max_rows, low_memory=False)
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({"column": str(c), "dtype": str(s.dtype), "missing": int(s.isna().sum()), "nunique": int(s.nunique(dropna=False)), "samples": [str(x) for x in s.dropna().head(5).tolist()]})
    candidates = [c for c in LABEL_CANDIDATES if c in df.columns]
    obj = {"csv": str(csv), "n_rows": int(len(df)), "n_columns": int(df.shape[1]), "candidate_label_columns": candidates, "class_counts": {c: df[c].astype(str).value_counts().head(50).to_dict() for c in candidates}, "columns": cols}
    write_json(outdir / "schema_audit.json", obj)
    lines = ["# X-MAG-IDS schema audit", "", f"CSV: `{csv}`", f"Rows sampled: **{len(df)}**", f"Columns: **{df.shape[1]}**", "", "## Candidate label columns", ""]
    lines += [f"- `{c}`" for c in candidates] or ["No standard label column detected."]
    lines += ["", "## Column summary", "", "| Column | Type | Missing | Unique | Samples |", "|---|---:|---:|---:|---|"]
    for item in cols:
        lines.append(f"| `{item['column']}` | `{item['dtype']}` | {item['missing']} | {item['nunique']} | {', '.join(item['samples'][:3])} |")
    (outdir / "schema_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote schema audit to {outdir}")


def main():
    p = argparse.ArgumentParser(description="X-MAG-IDS pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True); s.add_argument("--rows", type=int, default=1200); s.add_argument("--seed", type=int, default=42)
    a = sub.add_parser("audit")
    a.add_argument("--csv", required=True); a.add_argument("--outdir", required=True); a.add_argument("--max-rows", type=int, default=None)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    z = sub.add_parser("summarize")
    z.add_argument("--run-dir", required=True)
    args = p.parse_args()
    if args.cmd == "synth":
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); generate_synthetic(args.rows, args.seed).to_csv(out, index=False); print(f"Wrote {args.rows} rows to {out}")
    elif args.cmd == "audit":
        audit(args.csv, args.outdir, args.max_rows)
    elif args.cmd == "run":
        run_experiment(args.config)
    elif args.cmd == "summarize":
        df = pd.read_csv(Path(args.run_dir) / "results.csv")
        cols = [c for c in ["method", "k", "known_macro_f1", "unknown_auroc", "unknown_recall_at_threshold", "message_bytes_per_flow"] if c in df.columns]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
