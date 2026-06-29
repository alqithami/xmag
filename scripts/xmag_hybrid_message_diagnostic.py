#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

from xmag_pipeline import (
    align_proba,
    average_proba,
    explanation_tokens,
    make_preprocessor,
    owner_proba,
    prepare,
    train_agents,
    rf,
)


def uncertainty_from_proba(p: np.ndarray) -> np.ndarray:
    return np.array([]) if p.size == 0 else 1.0 - np.max(p, axis=1)


def scale_by_ref(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.nanpercentile(ref if ref.size else x, [5, 95])
    den = hi - lo
    if not np.isfinite(den) or den <= 1e-12:
        den = np.nanstd(ref if ref.size else x)
    if not np.isfinite(den) or den <= 1e-12:
        den = 1.0
    return np.clip((x - lo) / den, 0.0, 1.0)


def iforest_raw_score(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -model.decision_function(X)


def train_local_iforests(X_train: np.ndarray, owner_ids: np.ndarray, n_agents: int, seed: int):
    fallback = IsolationForest(n_estimators=100, contamination="auto", random_state=seed, n_jobs=1).fit(X_train)
    models = {}
    for aid in range(n_agents):
        mask = owner_ids == aid
        if int(mask.sum()) >= 50:
            models[aid] = IsolationForest(n_estimators=100, contamination="auto", random_state=seed + aid + 1, n_jobs=1).fit(X_train[mask])
        else:
            models[aid] = fallback
    return models


def score_by_owner(models: dict[int, IsolationForest], X: np.ndarray, owner_ids: np.ndarray) -> np.ndarray:
    s = np.zeros(X.shape[0], dtype=float)
    for aid, model in models.items():
        mask = owner_ids == aid
        if np.any(mask):
            s[mask] = iforest_raw_score(model, X[mask])
    return s


def topm_dense(proba: np.ndarray, m: int) -> np.ndarray:
    if proba.size == 0:
        return proba.copy()
    m = int(max(1, min(m, proba.shape[1])))
    out = np.zeros_like(proba, dtype=float)
    idx = np.argpartition(proba, -m, axis=1)[:, -m:]
    rows = np.arange(proba.shape[0])[:, None]
    out[rows, idx] = proba[rows, idx]
    return out


def build_owner_tokens(X: np.ndarray, owner_ids: np.ndarray, agents, model_by_id, k: int, cfg: dict):
    dense = np.zeros_like(X, dtype=float)
    e = cfg["explanation"]
    msg = int(e.get("message_overhead_bytes", 8)) + int(k) * (int(e.get("feature_id_bytes", 2)) + int(e.get("value_bytes", 4)))
    for aid, _model in agents:
        mask = owner_ids == aid
        if np.any(mask):
            tok, msg = explanation_tokens(
                X[mask],
                model_by_id[aid],
                k,
                int(e.get("message_overhead_bytes", 8)),
                int(e.get("feature_id_bytes", 2)),
                int(e.get("value_bytes", 4)),
            )
            dense[mask] = tok
    return dense, msg


def open_metrics(val_scores, known_scores, unknown_scores, q=0.95):
    threshold = float(np.quantile(val_scores if val_scores.size else known_scores, q))
    y = np.r_[np.zeros_like(known_scores, dtype=int), np.ones_like(unknown_scores, dtype=int)]
    s = np.r_[known_scores, unknown_scores]
    auroc = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
    pred_u = (unknown_scores >= threshold).astype(int)
    pred_k = (known_scores >= threshold).astype(int)
    return auroc, float(recall_score(np.ones_like(pred_u), pred_u, zero_division=0)), float(np.mean(pred_k)), threshold


def fusion_scores(p_val, p_known, p_unknown, a_val, a_known, a_unknown, alpha: float):
    uv = uncertainty_from_proba(p_val)
    uk = uncertainty_from_proba(p_known)
    uu = uncertainty_from_proba(p_unknown)
    val = alpha * scale_by_ref(uv, uv) + (1 - alpha) * scale_by_ref(a_val, a_val)
    known = alpha * scale_by_ref(uv, uk) + (1 - alpha) * scale_by_ref(a_val, a_known)
    unknown = alpha * scale_by_ref(uv, uu) + (1 - alpha) * scale_by_ref(a_val, a_unknown)
    return val, known, unknown


def predict_from_proba(p: np.ndarray, classes: list[str]) -> np.ndarray:
    return np.asarray(classes)[np.argmax(p, axis=1)]


def row(method, k, top_m, y_known, p_known, classes, val_scores, known_scores, unknown_scores, msg, seconds):
    pred = predict_from_proba(p_known, classes)
    auroc, urec, kfar, thr = open_metrics(val_scores, known_scores, unknown_scores)
    return {
        "method": method,
        "k": k,
        "top_m": top_m,
        "known_macro_f1": float(f1_score(y_known, pred, average="macro", zero_division=0)),
        "unknown_auroc": auroc,
        "unknown_recall_at_threshold": urec,
        "known_false_alarm_rate_at_threshold": kfar,
        "message_bytes_per_flow": int(msg),
        "seconds": float(seconds),
        "unknown_threshold": thr,
    }


def aligned_lr_proba(model, X, classes):
    raw = model.predict_proba(X)
    out = np.zeros((X.shape[0], len(classes)))
    local = [str(c) for c in model.classes_]
    for i, lab in enumerate(local):
        if lab in classes:
            out[:, classes.index(lab)] = raw[:, i]
    sums = out.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return out / sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--top-m", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["dataset"]["path"], nrows=cfg["experiment"].get("max_rows"), low_memory=False)
    parts = prepare(df, cfg)
    Xtr, ytr, atr, Xv, yv, av, Xk, yk, ak, Xu, yu, au, classes, _summary = parts

    pre = make_preprocessor(Xtr)
    Xtrt = pre.fit_transform(Xtr)
    Xvt = pre.transform(Xv)
    Xkt = pre.transform(Xk)
    Xut = pre.transform(Xu)
    seed = cfg["experiment"].get("random_state", 42)
    n_agents = cfg["agents"].get("n_agents", 8)

    rows = []
    tic = time.perf_counter()
    agents = train_agents(rf(cfg, seed), Xtrt, ytr, atr, n_agents, classes)
    iforests = train_local_iforests(Xtrt, atr, n_agents, seed)

    owner_p_train = owner_proba(agents, Xtrt, atr, classes)
    owner_p_val = owner_proba(agents, Xvt, av, classes)
    owner_p_known = owner_proba(agents, Xkt, ak, classes)
    owner_p_unknown = owner_proba(agents, Xut, au, classes) if len(Xut) else np.empty((0, len(classes)))

    avg_p_val = average_proba(agents, Xvt, classes)
    avg_p_known = average_proba(agents, Xkt, classes)
    avg_p_unknown = average_proba(agents, Xut, classes) if len(Xut) else np.empty((0, len(classes)))

    a_train_raw = score_by_owner(iforests, Xtrt, atr)
    a_val = scale_by_ref(a_train_raw, score_by_owner(iforests, Xvt, av))
    a_known = scale_by_ref(a_train_raw, score_by_owner(iforests, Xkt, ak))
    a_unknown = scale_by_ref(a_train_raw, score_by_owner(iforests, Xut, au)) if len(Xut) else np.array([])
    setup_seconds = time.perf_counter() - tic

    fv, fk, fu = fusion_scores(avg_p_val, avg_p_known, avg_p_unknown, a_val, a_known, a_unknown, args.alpha)
    rows.append(row("baseline_logit_average_plus_local_anomaly", None, None, yk, avg_p_known, classes, fv, fk, fu, len(classes) * 4 + 4, setup_seconds))

    model_by_id = {aid: model for aid, model in agents}
    for k in cfg["explanation"].get("k_values", [1, 3, 5]):
        tic = time.perf_counter()
        tok_train, token_msg = build_owner_tokens(Xtrt, atr, agents, model_by_id, int(k), cfg)
        tok_val, _ = build_owner_tokens(Xvt, av, agents, model_by_id, int(k), cfg)
        tok_known, _ = build_owner_tokens(Xkt, ak, agents, model_by_id, int(k), cfg)
        tok_unknown, _ = build_owner_tokens(Xut, au, agents, model_by_id, int(k), cfg) if len(Xut) else (np.empty((0, Xtrt.shape[1])), token_msg)

        for m in args.top_m:
            ptr = topm_dense(owner_p_train, m)
            pvv = topm_dense(owner_p_val, m)
            pkk = topm_dense(owner_p_known, m)
            puu = topm_dense(owner_p_unknown, m)

            Xmsg_train = np.c_[ptr, tok_train, scale_by_ref(a_train_raw, a_train_raw)]
            Xmsg_val = np.c_[pvv, tok_val, a_val]
            Xmsg_known = np.c_[pkk, tok_known, a_known]
            Xmsg_unknown = np.c_[puu, tok_unknown, a_unknown] if len(Xut) else np.empty((0, Xmsg_train.shape[1]))

            coord = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
            coord.fit(Xmsg_train, ytr.astype(str))
            pv = aligned_lr_proba(coord, Xmsg_val, classes)
            pk = aligned_lr_proba(coord, Xmsg_known, classes)
            pu = aligned_lr_proba(coord, Xmsg_unknown, classes) if len(Xut) else np.empty((0, len(classes)))
            fv, fk, fu = fusion_scores(pv, pk, pu, a_val, a_known, a_unknown, args.alpha)
            seconds = time.perf_counter() - tic
            msg = token_msg + int(m) * (2 + 4) + 4
            rows.append(row(f"xmag_top{m}_class_token_anomaly_fusion", int(k), int(m), yk, pk, classes, fv, fk, fu, msg, seconds))

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "hybrid_message_diagnostic.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {outdir / 'hybrid_message_diagnostic.csv'}")


if __name__ == "__main__":
    main()
