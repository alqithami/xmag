#!/usr/bin/env python3
"""Shared implementation for the MDPI Mathematics round-1 revision experiments.

This module is deliberately self-contained. It does not depend on the older
diagnostic scripts, so every reviewer-requested comparison uses one split,
one agent mapping, one thresholding rule, and one metric implementation.
"""
from __future__ import annotations

import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax
from scipy.stats import rankdata, spearmanr, weibull_min
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABEL_CANDIDATES = ["Attack Type", "Attack_Type", "attack_type", "XMagCategory", "Label", "label", "class", "Class"]
AGENT_CANDIDATES = [
    "sVid", "srcVid", "Source Vid", "source_vid", "BTS", "BTS_ID", "bts_id",
    "Src IP", "Source IP", "src_ip", "source_ip", "device_id", "Device ID",
]


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    a_train: np.ndarray
    idx_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    a_val: np.ndarray
    idx_val: np.ndarray
    X_known: np.ndarray
    y_known: np.ndarray
    a_known: np.ndarray
    idx_known: np.ndarray
    X_unknown: np.ndarray
    y_unknown: np.ndarray
    a_unknown: np.ndarray
    idx_unknown: np.ndarray
    classes: list[str]
    feature_names: list[str]
    label_column: str
    agent_source: str
    unknown_attack: str
    benign_class: str
    raw_feature_count: int
    transformed_feature_count: int
    preprocessor: ColumnTransformer


def mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_label(columns: Iterable[str], configured: str | None) -> str:
    cols = list(columns)
    if configured:
        if configured not in cols:
            raise ValueError(f"Configured label column {configured!r} is missing.")
        return configured
    for c in LABEL_CANDIDATES:
        if c in cols:
            return c
    raise ValueError(f"Unable to infer label column. Available columns: {cols[:40]}")


def leakage_columns(df: pd.DataFrame, cfg: dict[str, Any], label: str) -> list[str]:
    ds = cfg["dataset"]
    out = {label}
    for c in ds.get("drop_columns", []):
        if c in df.columns:
            out.add(c)
    patterns = [str(x).lower().replace("_", " ") for x in ds.get("leakage_patterns", [])]
    for c in df.columns:
        norm = str(c).lower().replace("_", " ")
        if any(p in norm for p in patterns):
            out.add(c)
    return [c for c in df.columns if c in out]


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    trs = []
    if numeric:
        trs.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        trs.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", enc),
        ]), categorical))
    if not trs:
        raise ValueError("No usable predictor columns remain after leakage control.")
    return ColumnTransformer(trs, remainder="drop", verbose_feature_names_out=False)


def stable_agents(meta: pd.Series | None, original_index: np.ndarray, n_agents: int) -> tuple[np.ndarray, str]:
    if n_agents < 1:
        raise ValueError("n_agents must be positive.")
    if meta is not None:
        values = meta.astype(str).fillna("<NA>").reset_index(drop=True)
        h = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype="uint64")
        return (h % np.uint64(n_agents)).astype(int), str(meta.name)
    values = pd.Series(original_index.astype(np.int64))
    h = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype="uint64")
    return (h % np.uint64(n_agents)).astype(int), "stable_row_index_hash"


def _stratify_or_none(y: pd.Series) -> pd.Series | None:
    vc = y.value_counts()
    return y if y.nunique() > 1 and len(vc) and int(vc.min()) >= 2 else None


def prepare_data(config_path: str | Path, seed: int, max_rows: int | None = None) -> PreparedData:
    cfg = load_yaml(config_path)
    csv_path = Path(cfg["dataset"]["path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_rows or cfg["experiment"].get("max_rows"), low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()
    label = infer_label(df.columns, cfg["dataset"].get("label_column"))
    y_all = df[label].astype(str).str.strip()
    unknown = str(cfg["dataset"]["unknown_attack"])
    if unknown not in set(y_all):
        raise ValueError(f"Unknown attack {unknown!r} not found. Available: {sorted(y_all.unique())}")
    benign_labels = [str(x) for x in cfg["dataset"].get("benign_labels", ["Benign", "BENIGN", "Normal"])]
    benign = next((b for b in benign_labels if b in set(y_all)), sorted(y_all.unique())[0])

    agent_col = cfg.get("agents", {}).get("agent_column")
    if agent_col and agent_col not in df.columns:
        raise ValueError(f"Configured agent column {agent_col!r} is missing.")
    if not agent_col:
        agent_col = next((c for c in AGENT_CANDIDATES if c in df.columns), None)

    drops = leakage_columns(df, cfg, label)
    Xdf = df.drop(columns=drops, errors="ignore").replace([np.inf, -np.inf], np.nan)
    nunique = Xdf.nunique(dropna=False)
    Xdf = Xdf[nunique[nunique > 1].index.tolist()]
    raw_d = int(Xdf.shape[1])

    idx = np.arange(len(df), dtype=np.int64)
    unknown_mask = y_all.eq(unknown).to_numpy()
    known_idx = idx[~unknown_mask]
    unknown_idx = idx[unknown_mask]
    y_known_all = y_all.iloc[known_idx].reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    trainval_idx, known_test_idx = train_test_split(
        known_idx,
        test_size=float(cfg["experiment"].get("test_size", 0.30)),
        random_state=seed,
        stratify=_stratify_or_none(y_known_all),
    )
    y_trainval = y_all.iloc[trainval_idx].reset_index(drop=True)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=float(cfg["experiment"].get("validation_size", 0.20)),
        random_state=seed,
        stratify=_stratify_or_none(y_trainval),
    )

    pre = make_preprocessor(Xdf.iloc[train_idx])
    Xtr = np.asarray(pre.fit_transform(Xdf.iloc[train_idx]), dtype=np.float32)
    Xv = np.asarray(pre.transform(Xdf.iloc[val_idx]), dtype=np.float32)
    Xk = np.asarray(pre.transform(Xdf.iloc[known_test_idx]), dtype=np.float32)
    Xu = np.asarray(pre.transform(Xdf.iloc[unknown_idx]), dtype=np.float32)
    try:
        feature_names = [str(x) for x in pre.get_feature_names_out()]
    except Exception:
        feature_names = [f"f{i}" for i in range(Xtr.shape[1])]

    n_agents = int(cfg.get("agents", {}).get("n_agents", 8))
    def agent_for(indices: np.ndarray) -> tuple[np.ndarray, str]:
        meta = df.iloc[indices][agent_col] if agent_col else None
        return stable_agents(meta, indices, n_agents)
    atr, source = agent_for(train_idx)
    av, _ = agent_for(val_idx)
    ak, _ = agent_for(known_test_idx)
    au, _ = agent_for(unknown_idx)

    ytr = y_all.iloc[train_idx].astype(str).to_numpy()
    yv = y_all.iloc[val_idx].astype(str).to_numpy()
    yk = y_all.iloc[known_test_idx].astype(str).to_numpy()
    yu = y_all.iloc[unknown_idx].astype(str).to_numpy()
    classes = sorted(np.unique(ytr).tolist())

    return PreparedData(
        Xtr, ytr, atr, train_idx,
        Xv, yv, av, val_idx,
        Xk, yk, ak, known_test_idx,
        Xu, yu, au, unknown_idx,
        classes, feature_names, label, source, unknown, benign,
        raw_d, int(Xtr.shape[1]), pre,
    )


def rf_model(seed: int, n_estimators: int = 30) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def align_proba(model: Any, X: np.ndarray, classes: list[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=np.float64)
    out = np.zeros((len(X), len(classes)), dtype=np.float64)
    local = [str(c) for c in model.classes_]
    for j, c in enumerate(local):
        if c in classes:
            out[:, classes.index(c)] = raw[:, j]
    den = out.sum(axis=1, keepdims=True)
    den[den <= 0] = 1.0
    return out / den


def train_local_classifiers(data: PreparedData, seed: int, n_estimators: int = 30) -> list[Any]:
    n_agents = int(max(data.a_train.max(initial=0), data.a_val.max(initial=0), data.a_known.max(initial=0), data.a_unknown.max(initial=0)) + 1)
    models = []
    for aid in range(n_agents):
        mask = data.a_train == aid
        ys = data.y_train[mask]
        if not np.any(mask):
            model = DummyClassifier(strategy="prior").fit(data.X_train, data.y_train)
        elif len(np.unique(ys)) == 1:
            model = DummyClassifier(strategy="constant", constant=ys[0]).fit(data.X_train[mask], ys)
        else:
            model = rf_model(seed + aid + 1, n_estimators).fit(data.X_train[mask], ys)
        models.append(model)
    return models


def owner_proba(models: list[Any], X: np.ndarray, owners: np.ndarray, classes: list[str]) -> np.ndarray:
    out = np.zeros((len(X), len(classes)), dtype=np.float64)
    for aid, model in enumerate(models):
        mask = owners == aid
        if np.any(mask):
            out[mask] = align_proba(model, X[mask], classes)
    return out


def average_proba(models: list[Any], X: np.ndarray, classes: list[str]) -> np.ndarray:
    return np.mean(np.stack([align_proba(m, X, classes) for m in models], axis=0), axis=0)


def train_local_anomaly(data: PreparedData, seed: int) -> list[IsolationForest]:
    n_agents = int(max(data.a_train.max(initial=0), data.a_val.max(initial=0), data.a_known.max(initial=0), data.a_unknown.max(initial=0)) + 1)
    fallback = IsolationForest(
        n_estimators=100, max_samples=min(4096, len(data.X_train)),
        contamination="auto", random_state=seed, n_jobs=-1,
    ).fit(data.X_train)
    models = []
    for aid in range(n_agents):
        mask = data.a_train == aid
        if int(mask.sum()) >= 64:
            m = IsolationForest(
                n_estimators=100, max_samples=min(4096, int(mask.sum())),
                contamination="auto", random_state=seed + 100 + aid, n_jobs=-1,
            ).fit(data.X_train[mask])
        else:
            m = fallback
        models.append(m)
    return models


def owner_anomaly(models: list[IsolationForest], X: np.ndarray, owners: np.ndarray) -> np.ndarray:
    out = np.zeros(len(X), dtype=np.float64)
    for aid, model in enumerate(models):
        mask = owners == aid
        if np.any(mask):
            out[mask] = -model.decision_function(X[mask])
    return out


def scale_ref(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return x
    lo, hi = np.nanpercentile(ref, [5.0, 95.0])
    den = hi - lo
    if not np.isfinite(den) or den <= 1e-12:
        den = np.nanstd(ref)
    if not np.isfinite(den) or den <= 1e-12:
        den = 1.0
    return np.clip((x - lo) / den, 0.0, 1.0)


def model_importance(model: Any, d: int) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        v = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "coef_"):
        v = np.mean(np.abs(np.asarray(model.coef_, dtype=np.float64)), axis=0)
    else:
        v = np.ones(d, dtype=np.float64)
    if len(v) != d:
        v = np.resize(v, d)
    s = np.abs(v).sum()
    return v / s if s > 0 else np.ones(d) / d


def owner_proxy(models: list[Any], X: np.ndarray, owners: np.ndarray, k: int = 1, quantize16: bool = False) -> np.ndarray:
    dense = np.zeros_like(X, dtype=np.float32)
    k = max(1, min(int(k), X.shape[1]))
    for aid, model in enumerate(models):
        mask = owners == aid
        if not np.any(mask):
            continue
        contrib = X[mask].astype(np.float64) * model_importance(model, X.shape[1])[None, :]
        idx = np.argpartition(np.abs(contrib), -k, axis=1)[:, -k:]
        rows = np.arange(mask.sum())[:, None]
        selected = np.zeros_like(contrib, dtype=np.float32)
        vals = contrib[rows, idx]
        if quantize16:
            vals = vals.astype(np.float16).astype(np.float32)
        selected[rows, idx] = vals
        dense[mask] = selected
    return dense


def topm_dense(p: np.ndarray, m: int, quantize16: bool = False) -> np.ndarray:
    if not len(p):
        return p.copy()
    m = max(1, min(int(m), p.shape[1]))
    out = np.zeros_like(p, dtype=np.float32)
    idx = np.argpartition(p, -m, axis=1)[:, -m:]
    rows = np.arange(len(p))[:, None]
    vals = p[rows, idx]
    if quantize16:
        vals = vals.astype(np.float16).astype(np.float32)
    out[rows, idx] = vals
    return out


def message_matrix(p: np.ndarray, proxy: np.ndarray, anomaly: np.ndarray, m: int = 1, k: int = 1,
                   include_class: bool = True, include_proxy: bool = True, include_anomaly: bool = True,
                   quantize16: bool = False) -> np.ndarray:
    pieces = []
    if include_class:
        pieces.append(topm_dense(p, m, quantize16))
    if include_proxy:
        pieces.append(proxy.astype(np.float16).astype(np.float32) if quantize16 else proxy)
    if include_anomaly:
        a = anomaly.astype(np.float16).astype(np.float32) if quantize16 else anomaly.astype(np.float32)
        pieces.append(a.reshape(-1, 1))
    if not pieces:
        raise ValueError("A message must contain at least one evidence component.")
    return np.concatenate(pieces, axis=1).astype(np.float32)


def fit_coordinator(M: np.ndarray, y: np.ndarray) -> Any:
    """Fit an explicit one-vs-rest logistic coordinator."""
    base = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )
    model = OneVsRestClassifier(base, n_jobs=-1)
    return model.fit(M, y)


def entropy_score(p: np.ndarray) -> np.ndarray:
    if not len(p):
        return np.array([], dtype=np.float64)
    q = np.clip(p, 1e-12, 1.0)
    return -np.sum(q * np.log(q), axis=1) / math.log(max(2, p.shape[1]))


def uncertainty_score(p: np.ndarray) -> np.ndarray:
    return np.array([], dtype=np.float64) if not len(p) else 1.0 - np.max(p, axis=1)


def predicted_labels(p: np.ndarray, classes: list[str]) -> np.ndarray:
    return np.asarray(classes)[np.argmax(p, axis=1)]


def prototype_residual(M_train: np.ndarray, y_train: np.ndarray, M: np.ndarray, p: np.ndarray,
                       classes: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    scale = np.nanstd(M_train, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    proto = {c: np.nanmean(M_train[y_train == c], axis=0) for c in classes if np.any(y_train == c)}
    pred = predicted_labels(p, classes)
    out = np.zeros(len(M), dtype=np.float64)
    for c in classes:
        mask = pred == c
        if np.any(mask) and c in proto:
            z = (M[mask] - proto[c][None, :]) / scale[None, :]
            out[mask] = np.sqrt(np.mean(z * z, axis=1))
    return out, proto, scale


def residual_with_proto(M: np.ndarray, p: np.ndarray, classes: list[str],
                        proto: dict[str, np.ndarray], scale: np.ndarray) -> np.ndarray:
    pred = predicted_labels(p, classes)
    out = np.zeros(len(M), dtype=np.float64)
    for c in classes:
        mask = pred == c
        if np.any(mask) and c in proto:
            z = (M[mask] - proto[c][None, :]) / scale[None, :]
            out[mask] = np.sqrt(np.mean(z * z, axis=1))
    return out


def composite_scores(u_ref: np.ndarray, a_ref: np.ndarray, r_ref: np.ndarray,
                     u: np.ndarray, a: np.ndarray, r: np.ndarray,
                     beta: float = 0.25, lam: float = 0.50, gamma: float = 0.25) -> np.ndarray:
    us = scale_ref(u_ref, u)
    aa = scale_ref(a_ref, a)
    rr = scale_ref(r_ref, r)
    s1 = beta * rr + (1.0 - beta) * us
    s2 = lam * us + (1.0 - lam) * aa
    return gamma * np.maximum(s1, s2) + (1.0 - gamma) * s2


def conformal_pvalues(cal_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Upper-tail split-conformal p-values; smaller means more anomalous."""
    cal = np.sort(np.asarray(cal_scores, dtype=np.float64))
    s = np.asarray(scores, dtype=np.float64)
    left = np.searchsorted(cal, s, side="left")
    ge = len(cal) - left
    return (1.0 + ge) / (len(cal) + 1.0)


def metrics_for_score(y_known: np.ndarray, p_known: np.ndarray, classes: list[str],
                      val_score: np.ndarray, known_score: np.ndarray, unknown_score: np.ndarray,
                      alpha: float = 0.05) -> dict[str, float]:
    pred = predicted_labels(p_known, classes)
    ybin = np.r_[np.zeros(len(known_score), dtype=int), np.ones(len(unknown_score), dtype=int)]
    sboth = np.r_[known_score, unknown_score]
    auroc = float(roc_auc_score(ybin, sboth)) if len(np.unique(ybin)) == 2 else float("nan")
    threshold = float(np.quantile(val_score, 1.0 - alpha, method="higher"))
    recall_q = float(np.mean(unknown_score >= threshold)) if len(unknown_score) else float("nan")
    fpr_q = float(np.mean(known_score >= threshold)) if len(known_score) else float("nan")
    pk = conformal_pvalues(val_score, known_score)
    pu = conformal_pvalues(val_score, unknown_score)
    recall_c = float(np.mean(pu <= alpha)) if len(pu) else float("nan")
    fpr_c = float(np.mean(pk <= alpha)) if len(pk) else float("nan")
    fpr, tpr, _ = roc_curve(ybin, sboth)
    valid = np.where(fpr <= alpha + 1e-12)[0]
    tpr_at = float(np.max(tpr[valid])) if len(valid) else 0.0
    return {
        "known_accuracy": float(accuracy_score(y_known, pred)),
        "known_balanced_accuracy": float(balanced_accuracy_score(y_known, pred)),
        "known_macro_f1": float(f1_score(y_known, pred, average="macro", zero_division=0)),
        "unknown_auroc": auroc,
        "unknown_recall_q95": recall_q,
        "known_fpr_q95": fpr_q,
        "unknown_recall_conformal05": recall_c,
        "known_fpr_conformal05": fpr_c,
        "tpr_at_test_fpr05": tpr_at,
        "q95_threshold": threshold,
    }


def packet_size(layout: str, K: int, d: int) -> int:
    sizes = {
        "class_anomaly_12": 12,
        "class_proxy_18": 18,
        "full_16q": 16,
        "full_20": 20,
        "full_24": 24,
        "top2_30": 30,
        "logits_anomaly": 4 * K + 4,
        "full_features": 4 * d,
    }
    if layout not in sizes:
        raise KeyError(layout)
    return int(sizes[layout])


def serialize_message(layout: str, class_id: int, prob: float, feature_id: int,
                      contribution: float, anomaly: float, K: int, d: int) -> bytes:
    """Actual serialization used to validate message-size accounting."""
    if layout == "full_16q":
        if K > 255 or d > 65535:
            raise ValueError("16-byte encoding requires K<=255 and d<=65535.")
        return struct.pack("<IBeeeH3s", 0, class_id, np.float16(prob),
                           np.float16(contribution), np.float16(anomaly),
                           feature_id, b"\x00\x00\x00")
    if layout == "full_20":
        return struct.pack("<I H f H f f", 0, class_id, prob, feature_id, contribution, anomaly)
    if layout == "full_24":
        return struct.pack("<Q H f H f f", 0, class_id, prob, feature_id, contribution, anomaly)
    raise ValueError(f"No serializer defined for {layout}")


def peer_context(M: np.ndarray, owners: np.ndarray, original_idx: np.ndarray,
                 loss: float = 0.0, jitter_steps: int = 0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build asynchronous pooled context from the latest delivered message of every agent."""
    rng = np.random.default_rng(seed)
    n_agents = int(owners.max(initial=0) + 1)
    state = np.zeros((n_agents, M.shape[1]), dtype=np.float32)
    seen = np.zeros(n_agents, dtype=bool)
    context = np.zeros_like(M, dtype=np.float32)
    delivered = np.ones(len(M), dtype=bool)
    order = np.argsort(original_idx)
    if jitter_steps > 0:
        key = np.arange(len(order)) + rng.integers(-jitter_steps, jitter_steps + 1, len(order))
        order = order[np.argsort(key, kind="stable")]
    for ii in order:
        aid = int(owners[ii])
        peer_mask = seen.copy()
        peer_mask[aid] = False
        if np.any(peer_mask):
            context[ii] = np.mean(state[peer_mask], axis=0)
        if rng.random() < loss:
            delivered[ii] = False
            continue
        state[aid] = M[ii]
        seen[aid] = True
    return context, delivered


def targeted_attack(M: np.ndarray, owners: np.ndarray, classes: list[str], benign: str,
                    fraction: float, attack: str, seed: int,
                    class_width: int, proxy_width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = M.copy()
    n_agents = int(owners.max(initial=0) + 1)
    n_bad = max(1, int(math.ceil(fraction * n_agents))) if fraction > 0 else 0
    bad_agents = rng.choice(np.arange(n_agents), size=n_bad, replace=False) if n_bad else np.array([], dtype=int)
    mask = np.isin(owners, bad_agents)
    if not np.any(mask):
        return out, bad_agents
    benign_idx = classes.index(benign) if benign in classes else 0
    if attack in {"class_suppression", "combined"}:
        out[mask, :class_width] = 0.0
        out[mask, benign_idx] = 1.0
    if attack in {"proxy_replacement", "combined"}:
        start = class_width
        out[mask, start:start + proxy_width] = 0.0
    if attack in {"anomaly_suppression", "combined"}:
        out[mask, -1] = 0.0
    if attack == "replay":
        good = np.where(~mask)[0]
        if len(good):
            repl = rng.choice(good, size=int(mask.sum()), replace=True)
            out[mask] = out[repl]
    return out, bad_agents


def hardware_metadata() -> dict[str, Any]:
    meta = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn
        meta["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import scipy
        meta["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import psutil
        meta["logical_cpus"] = psutil.cpu_count(logical=True)
        meta["physical_cpus"] = psutil.cpu_count(logical=False)
        meta["memory_bytes"] = psutil.virtual_memory().total
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            meta["cpu_brand"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    return meta


def model_size_bytes(*models: Any) -> int:
    total = 0
    with tempfile.TemporaryDirectory() as td:
        for i, model in enumerate(models):
            p = Path(td) / f"m{i}.joblib"
            joblib.dump(model, p, compress=0)
            total += p.stat().st_size
    return int(total)


def timed_predict(fn, repeats: int = 5) -> tuple[Any, float]:
    result = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, float(np.median(times))


def fit_fedavg_linear(
    data: PreparedData,
    rounds: int = 5,
    seed: int = 0,
) -> SGDClassifier:
    """Fit a protocol-matched FedAvg linear baseline without ``partial_fit``."""
    classes = np.asarray(data.classes, dtype=str)
    X_train = np.ascontiguousarray(data.X_train, dtype=np.float64)
    y_train = np.asarray(data.y_train, dtype=str)
    owners = np.asarray(data.a_train, dtype=np.int64)

    if X_train.ndim != 2 or len(X_train) != len(y_train):
        raise ValueError("FedAvg received inconsistent training arrays.")
    if len(classes) < 2:
        raise ValueError("FedAvg requires at least two known classes.")

    n_classes = len(classes)
    n_features = X_train.shape[1]
    coef_rows = 1 if n_classes == 2 else n_classes
    global_coef = np.zeros((coef_rows, n_features), dtype=np.float64)
    global_intercept = np.zeros(coef_rows, dtype=np.float64)

    representative_index = {}
    for class_label in classes:
        locations = np.flatnonzero(y_train == class_label)
        if not len(locations):
            raise ValueError(f"Known class {class_label!r} has no global training exemplar.")
        representative_index[class_label] = int(locations[0])

    final_template = None
    agent_ids = np.unique(owners)
    for round_index in range(int(rounds)):
        local_coef = []
        local_intercept = []
        local_weights = []
        for agent_id in agent_ids:
            original_ids = np.flatnonzero(owners == agent_id)
            if not len(original_ids):
                continue
            present = set(y_train[original_ids].tolist())
            missing = [c for c in classes if c not in present]
            if missing:
                exemplar_ids = np.asarray([representative_index[c] for c in missing], dtype=np.int64)
                fit_ids = np.concatenate([original_ids, exemplar_ids])
                sample_weight = np.ones(len(fit_ids), dtype=np.float64)
                sample_weight[-len(exemplar_ids):] = 1e-6
            else:
                fit_ids = original_ids
                sample_weight = np.ones(len(fit_ids), dtype=np.float64)

            local_model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                learning_rate="constant",
                eta0=1e-3,
                max_iter=1,
                tol=None,
                shuffle=True,
                random_state=seed + 1000 * round_index + int(agent_id),
                n_jobs=1,
            )
            local_model.fit(
                np.ascontiguousarray(X_train[fit_ids], dtype=np.float64),
                y_train[fit_ids],
                sample_weight=sample_weight,
                coef_init=np.ascontiguousarray(global_coef, dtype=np.float64),
                intercept_init=np.ascontiguousarray(global_intercept, dtype=np.float64),
            )
            if list(local_model.classes_.astype(str)) != list(classes):
                raise RuntimeError("FedAvg local class order differs from the global class order.")
            local_coef.append(np.ascontiguousarray(local_model.coef_, dtype=np.float64))
            local_intercept.append(np.ascontiguousarray(local_model.intercept_, dtype=np.float64))
            local_weights.append(float(len(original_ids)))
            final_template = local_model

        if not local_coef or final_template is None:
            raise RuntimeError("FedAvg received no non-empty agent partitions.")
        normalized_weights = np.asarray(local_weights, dtype=np.float64)
        normalized_weights /= normalized_weights.sum()
        global_coef = np.ascontiguousarray(
            np.tensordot(normalized_weights, np.stack(local_coef, axis=0), axes=(0, 0)),
            dtype=np.float64,
        )
        global_intercept = np.ascontiguousarray(
            np.tensordot(normalized_weights, np.stack(local_intercept, axis=0), axes=(0, 0)),
            dtype=np.float64,
        )

    final_model = deepcopy(final_template)
    final_model.coef_ = global_coef
    final_model.intercept_ = global_intercept
    return final_model


def evt_score(X_train: np.ndarray, y_train: np.ndarray, X: np.ndarray, p: np.ndarray,
              classes: list[str], tail_fraction: float = 0.10) -> np.ndarray:
    pred = predicted_labels(p, classes)
    means = {c: np.mean(X_train[y_train == c], axis=0) for c in classes if np.any(y_train == c)}
    scale = np.std(X_train, axis=0)
    scale[scale < 1e-8] = 1.0
    params: dict[str, tuple[float, float]] = {}
    for c, mu in means.items():
        d = np.sqrt(np.mean(((X_train[y_train == c] - mu) / scale) ** 2, axis=1))
        n_tail = max(20, int(math.ceil(tail_fraction * len(d))))
        tail = np.sort(d)[-n_tail:]
        try:
            shape, _, scl = weibull_min.fit(np.maximum(tail, 1e-8), floc=0)
            params[c] = (float(shape), float(scl))
        except Exception:
            params[c] = (1.0, float(np.quantile(tail, 0.95)) or 1.0)
    out = np.zeros(len(X), dtype=np.float64)
    for c in classes:
        mask = pred == c
        if not np.any(mask) or c not in means:
            continue
        d = np.sqrt(np.mean(((X[mask] - means[c]) / scale) ** 2, axis=1))
        shape, scl = params[c]
        out[mask] = weibull_min.cdf(np.maximum(d, 0), shape, loc=0, scale=max(scl, 1e-8))
    return out
