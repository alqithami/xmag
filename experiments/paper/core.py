#!/usr/bin/env python3
"""Matched component replacements and exact empirical-FPR evaluation.

This is an isolated follow-up protocol, not a relabeling of historical results.
No test observation is used to fit a detector or its deployed calibration rule.
Fixed-FPR results are retrospective, boundary-randomized ROC operating points;
neighboring deterministic points are always exported as well.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

import reference

VERSION = 'xmag_reviewer_extension_v1.0'
SEEDS = [7, 21, 42, 84, 123]
HOLDOUTS = ['HTTPFlood', 'ICMPFlood', 'SYNFlood', 'SYNScan', 'SlowrateDoS',
            'TCPConnectScan', 'UDPFlood', 'UDPScan']
LEVELS = [0.001, 0.01, 0.05]
DATASET_SHA256 = '7c238e2d5dabbc1afcd01c50db91372d6f6808f7572bb92baeb2df03a18af90c'
CLASS_COUNTS = {'Benign': 477737, 'HTTPFlood': 140812, 'ICMPFlood': 1155,
                'SYNFlood': 9721, 'SYNScan': 20043, 'SlowrateDoS': 73124,
                'TCPConnectScan': 20052, 'UDPFlood': 457340, 'UDPScan': 15906}
REPLACEMENTS = ['local_extratrees_16q', 'anomaly_gmm_16q', 'coordinator_extratrees_16q',
                'residual_l1_16q', 'fusion_mean_16q', 'fusion_max_16q']
CONTROLS = ['control_12b', 'control_24b', 'uncertainty_16q', 'central_rf']
METHODS = ['baseline_16q'] + REPLACEMENTS + CONTROLS
CONFIG = {
    'dataset': {'label_column': 'Attack Type', 'drop_columns': ['Attack Tool', 'Label', 'sVid', 'dVid', '54'],
                'leakage_patterns': ['label', 'attack tool', 'attack_tool', 'scenario', 'timestamp',
                                     'time', 'source ip', 'destination ip', 'src ip', 'dst ip', 'srcport', 'dstport']},
    'known_test_fraction': 0.30, 'validation_within_trainval': 0.20,
    'normalization_fraction_of_validation': 0.50,
    'nominal_slots': 8, 'rf_trees': 30, 'if_trees': 100, 'if_max_samples': 4096,
    'gmm_components': 8, 'gmm_train_cap': 16384, 'gmm_max_iter': 200,
    'coordinator_et_trees': 50, 'coordinator_et_depth': 12, 'coordinator_et_leaf': 5,
    'beta': 0.25, 'lambda': 0.50, 'gamma': 0.25,
    'fixed_fpr_denominators': ['all_known', 'benign_only'],
    'fixed_fpr_convention': 'expected FPR from randomized acceptance at one tied score boundary; also strict and inclusive endpoints',
    'prediction_parallelism': 1,
}
SPLITS = ['train', 'norm', 'cal', 'known', 'unknown']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def atomic_joblib(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    joblib.dump(value, tmp, compress=3)
    os.replace(tmp, path)


def clean_array(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError('Scores must be nonempty finite one-dimensional arrays.')
    return x


def fixed_fpr(negative, positive, target):
    """Evaluate the same exact *expected* empirical FPR despite tied scores.

    Reject s>t, and reject s==t with probability rho. Selection of t and rho
    uses only the designated negative test population. This is a retrospective
    ROC comparison, NOT an independently calibrated deployment threshold.
    """
    neg, pos = clean_array(negative), clean_array(positive)
    if not 0 < target < 1:
        raise ValueError('FPR target must lie strictly between zero and one.')
    budget = target * len(neg)
    nearest = round(budget)
    if abs(budget - nearest) < 1e-10:
        budget = float(nearest)
    desc = np.sort(neg)[::-1]
    threshold = float(desc[min(int(math.floor(budget)), len(desc) - 1)])
    n_above = int(np.count_nonzero(neg > threshold))
    n_equal = int(np.count_nonzero(neg == threshold))
    p_above = int(np.count_nonzero(pos > threshold))
    p_equal = int(np.count_nonzero(pos == threshold))
    rho = float((budget - n_above) / n_equal)
    if not -1e-12 <= rho <= 1 + 1e-12:
        raise AssertionError('Invalid boundary randomization probability.')
    rho = min(1.0, max(0.0, rho))
    achieved = (n_above + rho * n_equal) / len(neg)
    if abs(achieved - target) > 1e-12:
        raise AssertionError('Fixed expected FPR did not equal its target.')
    return {'target_fpr': float(target), 'expected_fpr': float(achieved),
            'expected_recall': float((p_above + rho * p_equal) / len(pos)),
            'strict_fpr': n_above / len(neg), 'strict_recall': p_above / len(pos),
            'inclusive_fpr': (n_above + n_equal) / len(neg),
            'inclusive_recall': (p_above + p_equal) / len(pos),
            'threshold': threshold, 'boundary_probability': rho,
            'n_negative': len(neg), 'n_unknown': len(pos),
            'negative_above': n_above, 'negative_tied': n_equal,
            'unknown_above': p_above, 'unknown_tied': p_equal}


def conformal_p(cal, score):
    cal, score = np.sort(clean_array(cal)), clean_array(score)
    return (1.0 + len(cal) - np.searchsorted(cal, score, side='left')) / (len(cal) + 1.0)


def conformal_cutoff(cal, alpha):
    cal = np.sort(clean_array(cal))
    k = int(math.floor(alpha * (len(cal) + 1)))
    if k < 1:
        return math.inf
    return float(cal[len(cal) - k])  # reject strictly above this order statistic


def evaluate_scores(cal, known, unknown, benign_mask, y_known, pred_known):
    cal, known, unknown = map(clean_array, (cal, known, unknown))
    benign_mask = np.asarray(benign_mask, dtype=bool)
    if len(known) != len(benign_mask) or not benign_mask.any():
        raise ValueError('Invalid known-test Benign mask.')
    if len(pred_known) != len(known) or len(y_known) != len(known):
        raise ValueError('Misaligned known predictions.')
    y = np.r_[np.zeros(len(known), dtype=np.int8), np.ones(len(unknown), dtype=np.int8)]
    score = np.r_[known, unknown]
    ranking = {'known_macro_f1': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
               'unknown_auroc': float(roc_auc_score(y, score)),
               'unknown_average_precision': float(average_precision_score(y, score)),
               'standardized_partial_auroc_fpr01': float(roc_auc_score(y, score, max_fpr=0.01)),
               'n_known': len(known), 'n_benign': int(benign_mask.sum()),
               'n_unknown': len(unknown), 'n_calibration': len(cal)}
    pk, pu = conformal_p(cal, known), conformal_p(cal, unknown)
    nominal, fixed = [], []
    for alpha in LEVELS:
        k_reject, u_reject = pk <= alpha, pu <= alpha
        nominal.append({'alpha': alpha, 'unknown_recall': float(u_reject.mean()),
                        'known_false_rejection_rate': float(k_reject.mean()),
                        'benign_false_rejection_rate': float(k_reject[benign_mask].mean()),
                        'n_benign_rejected': int(k_reject[benign_mask].sum()), **ranking})
        for denominator, neg in [('all_known', known), ('benign_only', known[benign_mask])]:
            fixed.append({'denominator': denominator, **fixed_fpr(neg, unknown, alpha)})
    return nominal, fixed


class RangeNormalizer:
    def __init__(self, values):
        values = clean_array(values)
        self.lo, hi = np.percentile(values, [5.0, 95.0])
        self.width = float(hi - self.lo)
        if not np.isfinite(self.width) or self.width <= 1e-12:
            self.width = float(np.std(values))
        if not np.isfinite(self.width) or self.width <= 1e-12:
            self.width = 1.0
        self.lo = float(self.lo)

    def __call__(self, x):
        return np.clip((np.asarray(x, dtype=np.float64) - self.lo) / self.width, 0.0, 1.0)


def combine(components, rule='composite'):
    u, a, r = components.T
    if rule == 'mean':
        return (u + a + r) / 3.0
    if rule == 'max':
        return np.maximum.reduce([u, a, r])
    if rule != 'composite':
        raise ValueError(rule)
    b, lam, g = CONFIG['beta'], CONFIG['lambda'], CONFIG['gamma']
    s1, s2 = b * r + (1 - b) * u, lam * u + (1 - lam) * a
    return g * np.maximum(s1, s2) + (1 - g) * s2


def prepare(csv_path, holdout, seed, fixture=False):
    """Training-only screen; original 56/14/30 split, validation split 7/7.

    The explicit split constructor is snapshotted. No claim of bit-identical
    historical results is made without the exported reconciliation check.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed:')].copy()
    if 'Attack Type' not in df or 'sVid' not in df:
        raise ValueError('The real encoded CSV must contain Attack Type and sVid.')
    y = df['Attack Type'].astype(str).str.strip().to_numpy()
    counts = {str(k): int(v) for k, v in pd.Series(y).value_counts().items()}
    if not fixture and counts != CLASS_COUNTS:
        raise ValueError(f'Dataset class counts differ from the completed audit: {counts}')
    if holdout not in counts or holdout == 'Benign':
        raise ValueError(f'Invalid held-out family: {holdout}')
    idx = np.arange(len(df), dtype=np.int64)
    known, unknown = idx[y != holdout], idx[y == holdout]
    trainval, test = train_test_split(known, test_size=0.30, random_state=seed, stratify=y[known])
    train, val = train_test_split(trainval, test_size=0.20, random_state=seed, stratify=y[trainval])
    norm, cal = train_test_split(val, test_size=0.50, random_state=seed, stratify=y[val])
    ids = dict(train=train, norm=norm, cal=cal, known=test, unknown=unknown)
    all_ids = np.concatenate(list(ids.values()))
    if len(np.unique(all_ids)) != len(df) or len(all_ids) != len(df):
        raise AssertionError('Split overlap or omission.')
    drops = set(reference.leakage_columns(df, CONFIG, 'Attack Type')) | {'sVid', 'dVid'}
    xdf = df.drop(columns=list(drops), errors='ignore').replace([np.inf, -np.inf], np.nan)
    n_unique = xdf.iloc[train].nunique(dropna=False)
    columns = list(n_unique[n_unique > 1].index)
    xdf = xdf[columns]
    pre = reference.make_preprocessor(xdf.iloc[train])
    x = {'train': np.asarray(pre.fit_transform(xdf.iloc[train]), dtype=np.float32)}
    for name in SPLITS[1:]:
        x[name] = np.asarray(pre.transform(xdf.iloc[ids[name]]), dtype=np.float32)
    if not all(np.isfinite(a).all() for a in x.values()):
        raise ValueError('Preprocessing produced nonfinite predictors.')
    owners, owner_name = reference.stable_agents(df['sVid'], idx, CONFIG['nominal_slots'])
    active = sorted(np.unique(owners[train]).astype(int).tolist())
    if not set(np.unique(owners)).issubset(active):
        raise ValueError('A test/validation source has no training observations.')
    classes = sorted(np.unique(y[train]).tolist())
    if set(classes) != set(y[known]):
        raise ValueError('A known class is absent from model training.')
    try:
        features = pre.get_feature_names_out().tolist()
    except AttributeError:
        features = [f'f{i}' for i in range(x['train'].shape[1])]
    data = {'X': x, 'y': {s: y[ids[s]] for s in SPLITS},
            'owners': {s: owners[ids[s]] for s in SPLITS}, 'ids': ids,
            'classes': classes, 'active': active, 'preprocessor': pre, 'features': features,
            'raw_columns': columns, 'holdout': holdout, 'seed': seed,
            'source': owner_name, 'class_counts': counts}
    return data


def split_manifest(data):
    return {'seed': data['seed'], 'holdout': data['holdout'], 'classes': data['classes'],
            'transformed_d': data['X']['train'].shape[1], 'raw_columns': data['raw_columns'],
            'feature_names': data['features'], 'active_sources': data['active'],
            'split_counts': {s: len(data['ids'][s]) for s in SPLITS},
            'split_id_sha256': {s: hashlib.sha256(data['ids'][s].astype('<i8').tobytes()).hexdigest() for s in SPLITS},
            'source_counts': {s: {str(a): int(np.sum(data['owners'][s] == a)) for a in range(8)} for s in SPLITS},
            'class_counts': {s: {str(c): int(np.sum(data['y'][s] == c)) for c in np.unique(data['y'][s])} for s in SPLITS},
            'protocol': VERSION, 'calibration_is_disjoint': True, 'predictor_screen_uses_training_only': True}


def cached_model(path, fit):
    path = Path(path)
    if path.is_file():
        print(f'      reusing {path.name}', flush=True)
        return joblib.load(path)
    value = fit()
    atomic_joblib(path, value)
    return value


def fit_sources(data, kind, jobs, quick=False):
    models, log = {}, []
    for a in data['active']:
        mask = data['owners']['train'] == a
        X, y = data['X']['train'][mask], data['y']['train'][mask]
        t0 = time.monotonic()
        if kind in ('rf', 'et'):
            if len(np.unique(y)) == 1:
                model = DummyClassifier(strategy='constant', constant=y[0]).fit(X, y)
            else:
                cls = RandomForestClassifier if kind == 'rf' else ExtraTreesClassifier
                model = cls(n_estimators=3 if quick else 30, min_samples_leaf=2,
                            class_weight='balanced_subsample', random_state=data['seed'] + a + 1,
                            n_jobs=jobs).fit(X, y)
                model.set_params(n_jobs=1)
        elif kind == 'if':
            model = IsolationForest(n_estimators=5 if quick else 100,
                                    max_samples=min(4096, len(X)), contamination='auto',
                                    random_state=data['seed'] + 100 + a, n_jobs=jobs).fit(X)
            model.set_params(n_jobs=1)
        elif kind == 'gmm':
            rng = np.random.default_rng(data['seed'] + 100 + a)
            chosen = np.sort(rng.choice(len(X), min(len(X), CONFIG['gmm_train_cap']), replace=False))
            ncomp = min(2 if quick else 8, len(chosen))
            model = GaussianMixture(n_components=ncomp, covariance_type='diag',
                                    reg_covar=1e-6, max_iter=200, tol=1e-3, n_init=1,
                                    init_params='kmeans', random_state=data['seed'] + 100 + a)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(np.asarray(X[chosen], dtype=np.float64))
            if not model.converged_:
                raise RuntimeError(f'Diagonal GMM failed to converge for training source {a}; no success result written.')
        else:
            raise ValueError(kind)
        models[a] = model
        rec = {'source': a, 'algorithm': kind, 'training_rows': len(X),
               'known_classes': np.unique(y).tolist(), 'fit_seconds': time.monotonic() - t0}
        if kind == 'gmm':
            rec.update(fitted_rows=len(chosen), converged=bool(model.converged_), iterations=int(model.n_iter_))
        log.append(rec)
    return {'models': models, 'log': log}


def source_fields(data, models):
    fields = {}
    for split in SPLITS:
        X, owners = data['X'][split], data['owners'][split]
        n, d = X.shape
        cid, fid = np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32)
        prob, val = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64)
        for a, model in models.items():
            indices = np.flatnonzero(owners == a)
            importance = reference.model_importance(model, d)
            for start in range(0, len(indices), 16384):
                ii = indices[start:start + 16384]
                P = reference.align_proba(model, X[ii], data['classes'])
                # Preserve the reference top-one argpartition convention.
                ci = np.argpartition(P, -1, axis=1)[:, -1]
                phi = X[ii].astype(np.float64) * importance[None, :]
                fi = np.argpartition(np.abs(phi), -1, axis=1)[:, -1]
                cid[ii], fid[ii] = ci, fi
                prob[ii], val[ii] = P[np.arange(len(ii)), ci], phi[np.arange(len(ii)), fi]
        fields[split] = {'class_id': cid, 'probability': prob, 'feature_id': fid, 'contribution': val}
    return fields


def anomaly_fields(data, models, kind):
    raw = {}
    for split in SPLITS:
        X, owners = data['X'][split], data['owners'][split]
        values = np.empty(len(X), dtype=np.float64)
        for a, model in models.items():
            indices = np.flatnonzero(owners == a)
            for start in range(0, len(indices), 32768):
                ii = indices[start:start + 32768]
                values[ii] = -model.decision_function(X[ii]) if kind == 'if' else -model.score_samples(X[ii].astype(np.float64))
        raw[split] = clean_array(values)
    # Same source-side scalar normalization for IF and GMM; training rows only.
    scale = RangeNormalizer(raw['train'])
    return {s: scale(raw[s]) for s in SPLITS}, {'lo': scale.lo, 'width': scale.width}


def packet_roundtrip(fields, anomaly, layout, K, d):
    """Packed structured arrays, little-endian, no native padding or side channel."""
    n = len(anomaly)
    if layout == '16q':
        dtype = np.dtype([('header', '<u4'), ('cid', 'u1'), ('p', '<f2'),
                          ('v', '<f2'), ('a', '<f2'), ('fid', '<u2'), ('reserved', 'u1', (3,))])
        expected = 16
    elif layout == '24b':
        dtype = np.dtype([('header', '<u8'), ('cid', '<u2'), ('p', '<f4'),
                          ('fid', '<u2'), ('v', '<f4'), ('a', '<f4')])
        expected = 24
    elif layout == '12b':
        dtype = np.dtype([('header', '<u2'), ('cid', '<u2'), ('p', '<f4'), ('a', '<f4')])
        expected = 12
    else:
        raise ValueError(layout)
    if dtype.itemsize != expected or K > (256 if layout == '16q' else 65536) or d > 65536:
        raise ValueError('Invalid packed layout or dictionary capacity.')
    cid = np.asarray(fields['class_id'])
    fid = np.asarray(fields['feature_id'])
    if np.any(cid < 0) or np.any(cid >= K) or np.any(fid < 0) or np.any(fid >= d):
        raise ValueError('Identifier out of dictionary bounds.')
    wire = np.zeros(n, dtype=dtype)
    wire['cid'], wire['p'], wire['a'] = cid, fields['probability'], anomaly
    if layout != '12b':
        wire['fid'], wire['v'] = fid, fields['contribution']
    wire = np.frombuffer(wire.tobytes(), dtype=dtype)
    for scalar in (['p', 'a'] if layout == '12b' else ['p', 'v', 'a']):
        if not np.all(np.isfinite(wire[scalar])):
            raise ValueError(f'Nonfinite/overflowed transmitted scalar {scalar} in {layout}.')
    M = np.zeros((n, K + (d if layout != '12b' else 0) + 1), dtype=np.float32)
    rows = np.arange(n)
    M[rows, wire['cid'].astype(int)] = wire['p'].astype(np.float32)
    if layout != '12b':
        M[rows, K + wire['fid'].astype(int)] = wire['v'].astype(np.float32)
    M[:, -1] = wire['a'].astype(np.float32)
    return M


def fit_head(M, y, kind, seed, jobs, quick=False):
    if kind == 'logistic':
        # Explicit wrapper: no deprecated multi_class and no incremental SGD.
        head = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=1.0,
                                   class_weight='balanced', max_iter=1000), n_jobs=1)
    elif kind == 'et':
        head = ExtraTreesClassifier(n_estimators=5 if quick else 50, max_depth=12,
                                   min_samples_leaf=5, class_weight='balanced',
                                   random_state=seed + 500, n_jobs=jobs)
    else:
        raise ValueError(kind)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        head.fit(M, y)
    conv = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if conv:
        raise RuntimeError('Coordinator convergence failure: ' + '; '.join(conv))
    if kind == 'et':
        head.set_params(n_jobs=1)
    return head


def fit_prototypes(M, y, classes):
    scale = np.std(M, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return np.stack([np.mean(M[y == c], axis=0) for c in classes]), scale


def residual(M, probabilities, prototypes, scale, rule='l2'):
    pred = np.argmax(probabilities, axis=1)
    out = np.empty(len(M), dtype=np.float64)
    for start in range(0, len(M), 16384):
        sl = slice(start, start + 16384)
        z = (M[sl] - prototypes[pred[sl]]) / scale[None, :]
        out[sl] = np.sqrt(np.mean(z * z, axis=1)) if rule == 'l2' else np.mean(np.abs(z), axis=1)
    return out


def raw_components(messages, head, prototypes, scale, classes, residual_rule='l2'):
    raw, preds = {}, {}
    for split in ['norm', 'cal', 'known', 'unknown']:
        M = messages[split]
        P = reference.align_proba(head, M, classes)
        raw[split] = np.column_stack([1.0 - P.max(axis=1), M[:, -1].astype(np.float64),
                                     residual(M, P, prototypes, scale, residual_rule)])
        preds[split] = np.asarray(classes)[np.argmax(P, axis=1)]
    return raw, preds


def normalized_scores(raw, rule='composite'):
    normalizers = [RangeNormalizer(raw['norm'][:, j]) for j in range(3)]
    normalized = {s: np.column_stack([normalizers[j](a[:, j]) for j in range(3)]) for s, a in raw.items()}
    scores = {s: combine(a, rule) for s, a in normalized.items()}
    return scores, normalizers, normalized


def prediction_bounds(M, Mq, head, prototypes, scale, normalizers, classes):
    """Interval bound through OVR normalization and possibly changing prototype.

    Uses actual input rounding radii |Mq-M| but not the quantized output score.
    Integer fields are unchanged. All fitted functions are held fixed.
    """
    if len(classes) < 3 or not isinstance(head, OneVsRestClassifier):
        raise ValueError('Certificate requires the multiclass OVR logistic control.')
    W = np.stack([est.coef_[0] for est in head.estimators_])
    b = np.array([est.intercept_[0] for est in head.estimators_])
    low, high = np.empty(len(M)), np.empty(len(M))
    margin_certified = np.empty(len(M), dtype=bool)
    eps_input = np.empty(len(M))
    for start in range(0, len(M), 8192):
        sl = slice(start, start + 8192)
        x = M[sl].astype(np.float64)
        delta = np.abs(Mq[sl].astype(np.float64) - x)
        logits, radius = x @ W.T + b, delta @ np.abs(W).T
        lo_v, hi_v = expit(logits - radius), expit(logits + radius)
        lo_q = lo_v / np.maximum(lo_v + hi_v.sum(axis=1, keepdims=True) - hi_v, 1e-300)
        hi_q = hi_v / np.maximum(hi_v + lo_v.sum(axis=1, keepdims=True) - lo_v, 1e-300)
        P = reference.align_proba(head, x, classes)
        pred = P.argmax(axis=1)
        others = hi_q.copy()
        others[np.arange(len(x)), pred] = -np.inf
        margin_certified[sl] = lo_q[np.arange(len(x)), pred] > others.max(axis=1)
        allowed = hi_q >= lo_q.max(axis=1, keepdims=True) - 1e-14
        eta = np.sqrt(np.mean((delta / scale[None, :]) ** 2, axis=1))
        lo_r, hi_r = np.full(len(x), np.inf), np.zeros(len(x))
        for c in range(len(classes)):
            rc = np.sqrt(np.mean(((x - prototypes[c]) / scale[None, :]) ** 2, axis=1))
            lo_r = np.minimum(lo_r, np.where(allowed[:, c], np.maximum(0.0, rc - eta), np.inf))
            hi_r = np.maximum(hi_r, np.where(allowed[:, c], rc + eta, 0.0))
        lower = np.column_stack([normalizers[0](1.0 - hi_q.max(axis=1)),
                                 normalizers[1](x[:, -1] - delta[:, -1]), normalizers[2](lo_r)])
        upper = np.column_stack([normalizers[0](1.0 - lo_q.max(axis=1)),
                                 normalizers[1](x[:, -1] + delta[:, -1]), normalizers[2](hi_r)])
        low[sl], high[sl] = combine(lower), combine(upper)
        eps_input[sl] = delta.max(axis=1)
    return low, high, margin_certified, eps_input


def quantization_audit(messages24, messages16, head, proto, scale, classes, data):
    raw24, pred24 = raw_components(messages24, head, proto, scale, classes)
    raw16, pred16 = raw_components(messages16, head, proto, scale, classes)
    score24, normalizers, _ = normalized_scores(raw24)
    score16 = {s: combine(np.column_stack([normalizers[j](a[:, j]) for j in range(3)])) for s, a in raw16.items()}
    cal_error = float(np.max(np.abs(score24['cal'] - score16['cal'])))
    reports = []
    for split in ['known', 'unknown']:
        lo, hi, certified, input_error = prediction_bounds(messages24[split], messages16[split], head, proto, scale, normalizers, classes)
        actual = score16[split]
        bound = np.maximum(np.abs(lo - score24[split]), np.abs(hi - score24[split]))
        # Floating-point evaluation tolerance is reported, not a mathematical claim of directed rounding.
        tolerance = 1e-8
        violations = int(np.sum((actual < lo - tolerance) | (actual > hi + tolerance)))
        class_changed = pred24[split] != pred16[split]
        if violations or np.any(certified & class_changed):
            raise AssertionError('Quantization interval/margin implementation failed its numerical check.')
        for alpha in LEVELS:
            t24, t16 = conformal_cutoff(score24['cal'], alpha), conformal_cutoff(score16['cal'], alpha)
            if not math.isfinite(t24) or not math.isfinite(t16):
                raise ValueError('Insufficient calibration observations for the requested precision audit.')
            dec24, dec16 = score24[split] > t24, score16[split] > t16
            possible = np.abs(score24[split] - t24) <= bound + cal_error + tolerance
            change = dec24 != dec16
            if np.any(change & ~possible) or abs(t24 - t16) > cal_error + tolerance:
                raise AssertionError('Decision-change/order-statistic bound failed.')
            reports.append({'population': split, 'alpha': alpha, 'n': len(actual),
                            'max_input_rounding_error': float(input_error.max()),
                            'max_score_change': float(np.max(np.abs(actual - score24[split]))),
                            'max_score_bound': float(bound.max()), 'mean_score_bound': float(bound.mean()),
                            'calibration_max_score_change': cal_error, 'threshold_shift': abs(t24 - t16),
                            'prediction_change_rate': float(class_changed.mean()),
                            'class_margin_certified_rate': float(certified.mean()),
                            'decision_change_rate': float(change.mean()),
                            'decision_change_bound_rate': float(possible.mean()),
                            'decision_rate_float32': float(dec24.mean()), 'decision_rate_binary16': float(dec16.mean()),
                            'numerical_bound_violations': violations, 'numerical_tolerance': tolerance,
                            'scope': 'fixed float32-trained head/prototypes/normalizers; calibration and test packets quantized; not separately retrained 16Q'})
    return reports
