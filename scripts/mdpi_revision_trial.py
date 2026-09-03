#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdpi_revision_common import *


def add_row(rows, *, seed, holdout, method, score_name, bytes_per_flow, p_val, p_known, p_unknown,
            val_score, known_score, unknown_score, y_known, classes, train_s, pred_s,
            model_bytes=np.nan, note="", scenario="standard"):
    met = metrics_for_score(y_known, p_known, classes, val_score, known_score, unknown_score)
    rows.append({
        "seed": seed, "held_out_attack": holdout, "method": method, "score": score_name,
        "scenario": scenario, "message_bytes_per_flow": bytes_per_flow,
        "train_seconds": train_s, "predict_seconds": pred_s,
        "per_flow_predict_us": pred_s / max(1, len(y_known) + len(p_unknown)) * 1e6,
        "model_size_bytes": model_bytes, "note": note, **met,
    })


def fit_message_variant(data, local_cls, local_if, *, m, k, include_class, include_proxy,
                        include_anomaly, quantize16=False):
    p_tr = owner_proba(local_cls, data.X_train, data.a_train, data.classes)
    p_v = owner_proba(local_cls, data.X_val, data.a_val, data.classes)
    p_k = owner_proba(local_cls, data.X_known, data.a_known, data.classes)
    p_u = owner_proba(local_cls, data.X_unknown, data.a_unknown, data.classes)

    a_tr_raw = owner_anomaly(local_if, data.X_train, data.a_train)
    a_v = scale_ref(a_tr_raw, owner_anomaly(local_if, data.X_val, data.a_val))
    a_k = scale_ref(a_tr_raw, owner_anomaly(local_if, data.X_known, data.a_known))
    a_u = scale_ref(a_tr_raw, owner_anomaly(local_if, data.X_unknown, data.a_unknown))

    proxy_tr = owner_proxy(local_cls, data.X_train, data.a_train, k=k, quantize16=quantize16)
    proxy_v = owner_proxy(local_cls, data.X_val, data.a_val, k=k, quantize16=quantize16)
    proxy_k = owner_proxy(local_cls, data.X_known, data.a_known, k=k, quantize16=quantize16)
    proxy_u = owner_proxy(local_cls, data.X_unknown, data.a_unknown, k=k, quantize16=quantize16)

    Mtr = message_matrix(p_tr, proxy_tr, scale_ref(a_tr_raw, a_tr_raw),
                         m, k, include_class, include_proxy, include_anomaly, quantize16)
    Mv = message_matrix(p_v, proxy_v, a_v, m, k, include_class, include_proxy, include_anomaly, quantize16)
    Mk = message_matrix(p_k, proxy_k, a_k, m, k, include_class, include_proxy, include_anomaly, quantize16)
    Mu = message_matrix(p_u, proxy_u, a_u, m, k, include_class, include_proxy, include_anomaly, quantize16)

    t0 = time.perf_counter()
    coord = fit_coordinator(Mtr, data.y_train)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pv = align_proba(coord, Mv, data.classes)
    pk = align_proba(coord, Mk, data.classes)
    pu = align_proba(coord, Mu, data.classes)
    pred_s = time.perf_counter() - t0

    uv, uk, uu = uncertainty_score(pv), uncertainty_score(pk), uncertainty_score(pu)
    rv_raw, proto, rscale = prototype_residual(Mtr, data.y_train, Mv, pv, data.classes)
    rk_raw = residual_with_proto(Mk, pk, data.classes, proto, rscale)
    ru_raw = residual_with_proto(Mu, pu, data.classes, proto, rscale)

    if include_anomaly:
        score_a_v, score_a_k, score_a_u = a_v, a_k, a_u
    else:
        score_a_v = np.zeros_like(a_v)
        score_a_k = np.zeros_like(a_k)
        score_a_u = np.zeros_like(a_u)

    sv = composite_scores(uv, score_a_v, rv_raw, uv, score_a_v, rv_raw)
    sk = composite_scores(uv, score_a_v, rv_raw, uk, score_a_k, rk_raw)
    su = composite_scores(uv, score_a_v, rv_raw, uu, score_a_u, ru_raw)

    return {
        "coord": coord, "Mtr": Mtr, "Mv": Mv, "Mk": Mk, "Mu": Mu,
        "pv": pv, "pk": pk, "pu": pu,
        "u": (uv, uk, uu), "a": (a_v, a_k, a_u), "r": (rv_raw, rk_raw, ru_raw),
        "scores": (sv, sk, su), "train_s": train_s, "pred_s": pred_s,
        "p_owner": (p_tr, p_v, p_k, p_u),
        "proxy": (proxy_tr, proxy_v, proxy_k, proxy_u),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--n-estimators", type=int, default=30)
    ap.add_argument("--skip-heavy-baselines", action="store_true")
    args = ap.parse_args()

    outdir = mkdir(args.out)
    dump_json(outdir / "hardware.json", hardware_metadata())
    data = prepare_data(args.config, args.seed, args.max_rows)
    dump_json(outdir / "protocol.json", {
        "seed": args.seed, "holdout": data.unknown_attack,
        "label_column": data.label_column, "agent_source": data.agent_source,
        "classes": data.classes, "K": len(data.classes),
        "raw_feature_count": data.raw_feature_count,
        "transformed_feature_count_d": data.transformed_feature_count,
        "split_sizes": {
            "train": len(data.y_train), "validation": len(data.y_val),
            "known_test": len(data.y_known), "unknown_test": len(data.y_unknown),
        },
        "split_ratios": {"test": 0.30, "validation_within_trainval": 0.20},
    })

    rows = []
    t0 = time.perf_counter()
    local_cls = train_local_classifiers(data, args.seed, args.n_estimators)
    local_if = train_local_anomaly(data, args.seed)
    local_train_s = time.perf_counter() - t0

    mainv = fit_message_variant(data, local_cls, local_if, m=1, k=1,
                                include_class=True, include_proxy=True, include_anomaly=True)
    sv, sk, su = mainv["scores"]
    add_row(rows, seed=args.seed, holdout=data.unknown_attack, method="X-MAG-COS-24B",
            score_name="composite", bytes_per_flow=24,
            p_val=mainv["pv"], p_known=mainv["pk"], p_unknown=mainv["pu"],
            val_score=sv, known_score=sk, unknown_score=su,
            y_known=data.y_known, classes=data.classes,
            train_s=local_train_s + mainv["train_s"], pred_s=mainv["pred_s"],
            model_bytes=model_size_bytes(*local_cls, *local_if, mainv["coord"]))

    uv, uk, uu = mainv["u"]
    add_row(rows, seed=args.seed, holdout=data.unknown_attack, method="X-MAG-DH-24B",
            score_name="coordinator_uncertainty", bytes_per_flow=24,
            p_val=mainv["pv"], p_known=mainv["pk"], p_unknown=mainv["pu"],
            val_score=uv, known_score=uk, unknown_score=uu,
            y_known=data.y_known, classes=data.classes,
            train_s=local_train_s + mainv["train_s"], pred_s=mainv["pred_s"])

    av, ak, au = mainv["a"]
    rv, rk, ru = mainv["r"]
    score_variants = {
        "owner_or_coordinator_uncertainty": (uv, uk, uu),
        "local_anomaly": (av, ak, au),
        "message_residual": (scale_ref(rv, rv), scale_ref(rv, rk), scale_ref(rv, ru)),
        "uncertainty_anomaly": (
            0.5 * scale_ref(uv, uv) + 0.5 * scale_ref(av, av),
            0.5 * scale_ref(uv, uk) + 0.5 * scale_ref(av, ak),
            0.5 * scale_ref(uv, uu) + 0.5 * scale_ref(av, au),
        ),
    }
    for name, (xv, xk, xu) in score_variants.items():
        add_row(rows, seed=args.seed, holdout=data.unknown_attack, method="X-MAG-COS-24B",
                score_name=name, bytes_per_flow=24,
                p_val=mainv["pv"], p_known=mainv["pk"], p_unknown=mainv["pu"],
                val_score=xv, known_score=xk, unknown_score=xu,
                y_known=data.y_known, classes=data.classes,
                train_s=local_train_s + mainv["train_s"], pred_s=mainv["pred_s"])

    variants = [
        ("Class+anomaly-12B", 12, dict(m=1, k=1, include_class=True, include_proxy=False, include_anomaly=True)),
        ("Class+proxy-18B", 18, dict(m=1, k=1, include_class=True, include_proxy=True, include_anomaly=False)),
        ("X-MAG-COS-16Q", 16, dict(m=1, k=1, include_class=True, include_proxy=True, include_anomaly=True, quantize16=True)),
        ("X-MAG-COS-20B", 20, dict(m=1, k=1, include_class=True, include_proxy=True, include_anomaly=True)),
        ("X-MAG-COS-30B", 30, dict(m=2, k=1, include_class=True, include_proxy=True, include_anomaly=True)),
    ]
    for name, b, kw in variants:
        vv = fit_message_variant(data, local_cls, local_if, **kw)
        xv, xk, xu = vv["scores"]
        add_row(rows, seed=args.seed, holdout=data.unknown_attack, method=name,
                score_name="composite", bytes_per_flow=b,
                p_val=vv["pv"], p_known=vv["pk"], p_unknown=vv["pu"],
                val_score=xv, known_score=xk, unknown_score=xu,
                y_known=data.y_known, classes=data.classes,
                train_s=local_train_s + vv["train_s"], pred_s=vv["pred_s"])

    serializers = {}
    for layout in ("full_16q", "full_20", "full_24"):
        payload = serialize_message(layout, 0, 0.5, 0, 0.1, 0.2, len(data.classes), data.transformed_feature_count)
        serializers[layout] = len(payload)
    dump_json(outdir / "message_serialization.json", serializers)

    t0 = time.perf_counter()
    central_rf = rf_model(args.seed, args.n_estimators).fit(data.X_train, data.y_train)
    ctrain = time.perf_counter() - t0
    t0 = time.perf_counter()
    cpv = align_proba(central_rf, data.X_val, data.classes)
    cpk = align_proba(central_rf, data.X_known, data.classes)
    cpu = align_proba(central_rf, data.X_unknown, data.classes)
    cpred = time.perf_counter() - t0
    ce_v, ce_k, ce_u = entropy_score(cpv), entropy_score(cpk), entropy_score(cpu)
    add_row(rows, seed=args.seed, holdout=data.unknown_attack, method="Central-RF-full",
            score_name="entropy", bytes_per_flow=4 * data.transformed_feature_count,
            p_val=cpv, p_known=cpk, p_unknown=cpu,
            val_score=ce_v, known_score=ce_k, unknown_score=ce_u,
            y_known=data.y_known, classes=data.classes, train_s=ctrain, pred_s=cpred,
            model_bytes=model_size_bytes(central_rf),
            note="Full-feature centralized upper bound.")

    ev_v = evt_score(data.X_train, data.y_train, data.X_val, cpv, data.classes)
    ev_k = evt_score(data.X_train, data.y_train, data.X_known, cpk, data.classes)
    ev_u = evt_score(data.X_train, data.y_train, data.X_unknown, cpu, data.classes)
    add_row(rows, seed=args.seed, holdout=data.unknown_attack, method="Central-RF-full",
            score_name="EVT-tail", bytes_per_flow=4 * data.transformed_feature_count,
            p_val=cpv, p_known=cpk, p_unknown=cpu,
            val_score=ev_v, known_score=ev_k, unknown_score=ev_u,
            y_known=data.y_known, classes=data.classes, train_s=ctrain, pred_s=cpred)

    t0 = time.perf_counter()
    davg_v = average_proba(local_cls, data.X_val, data.classes)
    davg_k = average_proba(local_cls, data.X_known, data.classes)
    davg_u = average_proba(local_cls, data.X_unknown, data.classes)
    dpred = time.perf_counter() - t0
    duv, duk, duu = entropy_score(davg_v), entropy_score(davg_k), entropy_score(davg_u)
    dsv = 0.5 * scale_ref(duv, duv) + 0.5 * scale_ref(av, av)
    dsk = 0.5 * scale_ref(duv, duk) + 0.5 * scale_ref(av, ak)
    dsu = 0.5 * scale_ref(duv, duu) + 0.5 * scale_ref(av, au)
    add_row(rows, seed=args.seed, holdout=data.unknown_attack, method="Distributed-logit-average",
            score_name="entropy+anomaly", bytes_per_flow=4 * len(data.classes) + 4,
            p_val=davg_v, p_known=davg_k, p_unknown=davg_u,
            val_score=dsv, known_score=dsk, unknown_score=dsu,
            y_known=data.y_known, classes=data.classes, train_s=local_train_s, pred_s=dpred)

    if not args.skip_heavy_baselines:
        t0 = time.perf_counter()
        et = ExtraTreesClassifier(
            n_estimators=max(50, args.n_estimators), min_samples_leaf=2,
            class_weight="balanced", random_state=args.seed, n_jobs=-1,
        ).fit(data.X_train, data.y_train)
        etrain = time.perf_counter() - t0
        t0 = time.perf_counter()
        epv, epk, epu = (align_proba(et, X, data.classes) for X in
                          (data.X_val, data.X_known, data.X_unknown))
        epred = time.perf_counter() - t0
        add_row(rows, seed=args.seed, holdout=data.unknown_attack, method="Central-ExtraTrees-full",
                score_name="entropy", bytes_per_flow=4 * data.transformed_feature_count,
                p_val=epv, p_known=epk, p_unknown=epu,
                val_score=entropy_score(epv), known_score=entropy_score(epk), unknown_score=entropy_score(epu),
                y_known=data.y_known, classes=data.classes, train_s=etrain, pred_s=epred,
                model_bytes=model_size_bytes(et))

        t0 = time.perf_counter()
        fed = fit_fedavg_linear(data, rounds=5, seed=args.seed)
        ftrain = time.perf_counter() - t0
        t0 = time.perf_counter()
        fpv, fpk, fpu = (align_proba(fed, X, data.classes) for X in
                          (data.X_val, data.X_known, data.X_unknown))
        fpred = time.perf_counter() - t0
        add_row(rows, seed=args.seed, holdout=data.unknown_attack, method="FedAvg-linear",
                score_name="entropy", bytes_per_flow=0,
                p_val=fpv, p_known=fpk, p_unknown=fpu,
                val_score=entropy_score(fpv), known_score=entropy_score(fpk), unknown_score=entropy_score(fpu),
                y_known=data.y_known, classes=data.classes, train_s=ftrain, pred_s=fpred,
                model_bytes=model_size_bytes(fed),
                note="Model-update communication is reported separately from per-flow evidence bytes.")

    np.savez_compressed(
        outdir / "score_components.npz",
        y_known=data.y_known, y_unknown=data.y_unknown,
        pred_known=predicted_labels(mainv["pk"], data.classes),
        pred_unknown=predicted_labels(mainv["pu"], data.classes),
        val_u=uv, known_u=uk, unknown_u=uu,
        val_a=av, known_a=ak, unknown_a=au,
        val_r=rv, known_r=rk, unknown_r=ru,
        val_composite=sv, known_composite=sk, unknown_composite=su,
        classes=np.asarray(data.classes),
    )

    threshold = float(np.quantile(sv, 0.95, method="higher"))
    accept = su < threshold
    pred_u = predicted_labels(mainv["pu"], data.classes)
    conf = pd.Series(pred_u[accept]).value_counts(dropna=False).rename_axis("assigned_known_class").reset_index(name="count")
    conf["held_out_attack"] = data.unknown_attack
    conf["seed"] = args.seed
    conf["accepted_unknown_total"] = int(accept.sum())
    conf.to_csv(outdir / "accepted_unknown_confusion.csv", index=False)

    profile = {
        "seed": args.seed,
        "holdout": data.unknown_attack,
        "d": data.transformed_feature_count,
        "K": len(data.classes),
        "agent_source": data.agent_source,
        "local_training_seconds": local_train_s,
        "main_coordinator_training_seconds": mainv["train_s"],
        "main_prediction_seconds": mainv["pred_s"],
        "main_per_flow_prediction_us": mainv["pred_s"] / max(1, len(data.y_known) + len(data.y_unknown)) * 1e6,
        "main_model_size_bytes": model_size_bytes(*local_cls, *local_if, mainv["coord"]),
    }
    dump_json(outdir / "profile.json", profile)

    pd.DataFrame(rows).to_csv(outdir / "metrics.csv", index=False)
    print(pd.DataFrame(rows)[[
        "method", "score", "message_bytes_per_flow", "known_macro_f1",
        "unknown_auroc", "unknown_recall_conformal05", "known_fpr_conformal05"
    ]].to_string(index=False))
    print(f"\nSaved trial outputs to {outdir}")


if __name__ == "__main__":
    main()
