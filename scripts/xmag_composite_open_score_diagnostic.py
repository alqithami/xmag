#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

from xmag_pipeline import make_preprocessor, owner_proba, prepare, rf, train_agents
from xmag_hybrid_message_diagnostic import (
    aligned_lr_proba,
    build_owner_tokens,
    open_metrics,
    scale_by_ref,
    score_by_owner,
    topm_dense,
    train_local_iforests,
    uncertainty_from_proba,
)


def entropy_from_proba(p: np.ndarray) -> np.ndarray:
    if p.size == 0:
        return np.array([])
    q = np.clip(p, 1e-12, 1.0)
    return -np.sum(q * np.log(q), axis=1) / np.log(max(p.shape[1], 2))


def combine_linear(s1, s2, gamma: float):
    return tuple(gamma * a + (1.0 - gamma) * b for a, b in zip(s1, s2))


def combine_max(s1, s2):
    return tuple(np.maximum(a, b) for a, b in zip(s1, s2))


def prototypes_by_class(Xmsg_train: np.ndarray, y_train: pd.Series, classes: list[str]):
    y = np.asarray(y_train).astype(str)
    proto = {}
    for c in classes:
        mask = y == c
        if np.any(mask):
            proto[c] = np.nanmean(Xmsg_train[mask], axis=0)
    scale = np.nanstd(Xmsg_train, axis=0)
    scale[~np.isfinite(scale)] = 1.0
    scale[scale < 1e-9] = 1.0
    return proto, scale


def residual_to_predicted_proto(Xmsg: np.ndarray, p: np.ndarray, classes: list[str], proto: dict, scale: np.ndarray):
    if Xmsg.size == 0:
        return np.array([])
    pred = np.asarray(classes)[np.argmax(p, axis=1)]
    out = np.zeros(Xmsg.shape[0], dtype=float)
    for i, c in enumerate(pred):
        center = proto.get(str(c))
        if center is not None:
            z = (Xmsg[i] - center) / scale
            out[i] = float(np.sqrt(np.mean(z * z)))
    return out


def add_score_row(rows, method, k, top_m, y_known, p_known, classes, val_s, known_s, unk_s, msg, seconds, score_name):
    pred = np.asarray(classes)[np.argmax(p_known, axis=1)]
    auroc, urec, kfar, thr = open_metrics(val_s, known_s, unk_s)
    rows.append({
        "method": method,
        "score": score_name,
        "k": k,
        "top_m": top_m,
        "known_macro_f1": float(f1_score(y_known, pred, average="macro", zero_division=0)),
        "unknown_auroc": auroc,
        "unknown_recall_at_threshold": urec,
        "known_false_alarm_rate_at_threshold": kfar,
        "message_bytes_per_flow": int(msg),
        "seconds": float(seconds),
        "unknown_threshold": thr,
    })


def main():
    ap = argparse.ArgumentParser(description="Composite open-set scoring for compact X-MAG messages.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-m", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--k-values", type=int, nargs="+", default=[1])
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.25])
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

    agents = train_agents(rf(cfg, seed), Xtrt, ytr, atr, n_agents, classes)
    iforests = train_local_iforests(Xtrt, atr, n_agents, seed)
    model_by_id = {aid: model for aid, model in agents}

    owner_p_train = owner_proba(agents, Xtrt, atr, classes)
    owner_p_val = owner_proba(agents, Xvt, av, classes)
    owner_p_known = owner_proba(agents, Xkt, ak, classes)
    owner_p_unknown = owner_proba(agents, Xut, au, classes) if len(Xut) else np.empty((0, len(classes)))

    a_train_raw = score_by_owner(iforests, Xtrt, atr)
    a_val = scale_by_ref(a_train_raw, score_by_owner(iforests, Xvt, av))
    a_known = scale_by_ref(a_train_raw, score_by_owner(iforests, Xkt, ak))
    a_unknown = scale_by_ref(a_train_raw, score_by_owner(iforests, Xut, au)) if len(Xut) else np.array([])
    local_anomaly = (a_val, a_known, a_unknown)

    rows = []
    for k in args.k_values:
        tok_train, token_msg = build_owner_tokens(Xtrt, atr, agents, model_by_id, int(k), cfg)
        tok_val, _ = build_owner_tokens(Xvt, av, agents, model_by_id, int(k), cfg)
        tok_known, _ = build_owner_tokens(Xkt, ak, agents, model_by_id, int(k), cfg)
        tok_unknown, _ = build_owner_tokens(Xut, au, agents, model_by_id, int(k), cfg) if len(Xut) else (np.empty((0, Xtrt.shape[1])), token_msg)

        for m in args.top_m:
            tic = time.perf_counter()
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
            seconds = time.perf_counter() - tic
            msg = token_msg + int(m) * (2 + 4) + 4
            method = f"xmag_top{m}_class_token_anomaly_fusion"

            proto, proto_scale = prototypes_by_class(Xmsg_train, ytr, classes)
            rv_raw = residual_to_predicted_proto(Xmsg_val, pv, classes, proto, proto_scale)
            rk_raw = residual_to_predicted_proto(Xmsg_known, pk, classes, proto, proto_scale)
            ru_raw = residual_to_predicted_proto(Xmsg_unknown, pu, classes, proto, proto_scale)

            owner_unc = (
                scale_by_ref(uncertainty_from_proba(owner_p_val), uncertainty_from_proba(owner_p_val)),
                scale_by_ref(uncertainty_from_proba(owner_p_val), uncertainty_from_proba(owner_p_known)),
                scale_by_ref(uncertainty_from_proba(owner_p_val), uncertainty_from_proba(owner_p_unknown)),
            )
            residual = (scale_by_ref(rv_raw, rv_raw), scale_by_ref(rv_raw, rk_raw), scale_by_ref(rv_raw, ru_raw))
            uncert_anom = combine_linear(owner_unc, local_anomaly, 0.5)
            resid_owner = combine_linear(residual, owner_unc, 0.25)
            name = "max_resid_owner_b025__uncert_anomaly_b05"
            add_score_row(rows, method, int(k), int(m), yk, pk, classes, *combine_max(resid_owner, uncert_anom), msg, seconds, name)
            for gamma in args.gamma:
                add_score_row(rows, method, int(k), int(m), yk, pk, classes, *combine_linear(resid_owner, uncert_anom, gamma), msg, seconds, f"linear_{name}_gamma_{gamma}")

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "composite_open_score_diagnostic.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {outdir / 'composite_open_score_diagnostic.csv'}")


if __name__ == "__main__":
    main()
