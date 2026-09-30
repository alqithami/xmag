#!/usr/bin/env python3
"""Exact Round-2 integration points recovered from xmag_round3_inputs.txt.

The supported launcher installs these functions before running the extension.
Real-data preprocessing calls the user's SHA-256-verified executed preparer.
No old script or old result is modified. A separate fixture path exists only
for software tests; it is never allowed to stand in for scientific data.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import xlogy
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

import core as c
import reference

SNAPSHOT_VERSION = 'xmag_reviewer_extension_snapshot_v1.1'
EXPECTED = {
    'mdpi_revision_common.py': '0cf4e433f795196b99698cdeea8951fe3627af46475012c338b1bbd73c75b223',
    'xmag_round2_core.py': '9d1c6d046ca910298f49a9e13bfdadc14fd014314b2549ae8bda9b760b98ccbf',
    'xmag_round2_run.py': '8693726c9fe9c2946fc32d70298af2111c79ac32fc7824a90b4255f2cd99f24c',
}
_BASE_PREPARE = c.prepare
_BASE_FIT_HEAD = c.fit_head


def validation_positions(n, seed):
    """The executed audit used this permutation, NOT a stratified half split."""
    if n < 20:
        raise ValueError('Too few validation rows to separate normalization and calibration.')
    order = np.random.default_rng(int(seed) + 9017).permutation(n)
    return order[:n // 2], order[n // 2:]


def verify_and_load(root):
    root = Path(root).resolve()
    for name, digest in EXPECTED.items():
        p = root / 'scripts' / name
        if not p.is_file() or c.sha256(p) != digest:
            raise RuntimeError(f'Executed-source mismatch: {p}. Do not overwrite it or bypass this check; the supplied snapshot must match.')
    if c.sha256(reference.__file__) != EXPECTED['mdpi_revision_common.py']:
        raise RuntimeError('Bundled reference.py does not match the supplied reference core.')
    for h in c.HOLDOUTS:
        p = root / 'configs' / 'holdouts' / f'real_5g_nidd_{h.lower()}.yaml'
        if not p.is_file():
            raise FileNotFoundError(p)
        cfg = reference.load_yaml(p)
        if (cfg['dataset']['unknown_attack'] != h or
                cfg['dataset']['label_column'] != 'Attack Type' or
                int(cfg['agents']['n_agents']) != 8 or
                int(cfg['model']['n_estimators']) != 30 or
                float(cfg['experiment']['test_size']) != .30 or
                float(cfg['experiment']['validation_size']) != .20):
            raise RuntimeError(f'Unexpected holdout configuration: {p}')
    sys.path.insert(0, str(root / 'scripts'))
    name = '_xmag_verified_round2_execution'
    spec = importlib.util.spec_from_file_location(name, root / 'scripts' / 'xmag_round2_run.py')
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load the verified Round-2 preparer.')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if c.sha256(mod.ref.__file__) != EXPECTED['mdpi_revision_common.py']:
        raise RuntimeError('The executed preparer imported a different reference module.')
    return mod


def convert_prepared(data, seed):
    """Adapt the actual PreparedData object without changing row construction."""
    ni, ci = validation_positions(len(data.y_val), seed)
    ids = {'train': data.idx_train, 'norm': data.idx_val[ni], 'cal': data.idx_val[ci],
           'known': data.idx_known, 'unknown': data.idx_unknown}
    x = {'train': data.X_train, 'norm': data.X_val[ni], 'cal': data.X_val[ci],
         'known': data.X_known, 'unknown': data.X_unknown}
    y = {'train': data.y_train, 'norm': data.y_val[ni], 'cal': data.y_val[ci],
         'known': data.y_known, 'unknown': data.y_unknown}
    owners = {'train': data.a_train, 'norm': data.a_val[ni], 'cal': data.a_val[ci],
              'known': data.a_known, 'unknown': data.a_unknown}
    if any(not np.isfinite(v).all() for v in x.values()):
        raise ValueError('Nonfinite processed predictor.')
    all_ids = np.concatenate(list(ids.values()))
    if len(np.unique(all_ids)) != len(all_ids):
        raise AssertionError('The executed preparer produced overlapping splits.')
    active = sorted(np.unique(data.a_train).astype(int).tolist())
    if not set(np.unique(np.concatenate(list(owners.values())))).issubset(active):
        raise ValueError('An evaluated source has no trained owner model.')
    counts = {str(k): int(v) for k, v in __import__('pandas').Series(np.concatenate(list(y.values()))).value_counts().items()}
    return {'X': x, 'y': y, 'owners': owners, 'ids': ids, 'classes': list(data.classes),
            'active': active, 'preprocessor': data.preprocessor, 'features': list(data.feature_names),
            'raw_columns': list(map(str, data.preprocessor.feature_names_in_)),
            'holdout': data.unknown_attack, 'seed': int(seed), 'source': data.agent_source,
            'class_counts': counts}


def prepare_fixture(csv, holdout, seed, fixture=False):
    """Software-only constructor with the same audited validation permutation."""
    if not fixture:
        raise RuntimeError('Fixture preprocessing is not a real-data fallback.')
    d = _BASE_PREPARE(csv, holdout, seed, fixture=True)
    all_ids = np.concatenate(list(d['ids'].values()))
    labels = np.empty(int(all_ids.max()) + 1, dtype=d['y']['train'].dtype)
    for s in c.SPLITS:
        labels[d['ids'][s]] = d['y'][s]
    known = np.sort(np.concatenate([d['ids'][s] for s in ['train', 'norm', 'cal', 'known']]))
    tv, kt = train_test_split(known, test_size=.30, random_state=seed, stratify=labels[known])
    tr, va = train_test_split(tv, test_size=.20, random_state=seed, stratify=labels[tv])
    np.testing.assert_array_equal(tr, d['ids']['train'])
    np.testing.assert_array_equal(kt, d['ids']['known'])
    ni, ci = validation_positions(len(va), seed)
    old_ids = np.r_[d['ids']['norm'], d['ids']['cal']]
    order = np.argsort(old_ids)
    joined = {key: np.concatenate([d[key]['norm'], d[key]['cal']]) for key in ['X', 'y', 'owners']}
    for split, ii in [('norm', va[ni]), ('cal', va[ci])]:
        rows = order[np.searchsorted(old_ids[order], ii)]
        for key in joined:
            d[key][split] = joined[key][rows]
        d['ids'][split] = ii
    return d


def source_fields(data, models):
    """Preserve argmax class ties and the proxy's intermediate float32 cast."""
    fields = {}
    for split in c.SPLITS:
        X, owners = data['X'][split], data['owners'][split]
        n, d = X.shape
        cid, fid = np.empty(n, np.int32), np.empty(n, np.int32)
        prob, val = np.empty(n, np.float64), np.empty(n, np.float32)
        for a, model in models.items():
            ii_all = np.flatnonzero(owners == a)
            importance = reference.model_importance(model, d)
            for start in range(0, len(ii_all), 16384):
                ii = ii_all[start:start + 16384]
                P = reference.align_proba(model, X[ii], data['classes'])
                ci = np.argmax(P, axis=1)
                phi = X[ii].astype(np.float64) * importance[None, :]
                fi = np.argpartition(np.abs(phi), -1, axis=1)[:, -1]
                values = phi[np.arange(len(ii)), fi].astype(np.float32)
                # The old dense top-one proxy is zero everywhere when its selected
                # value rounds to zero; the later packet argmax then selects index 0.
                fid[ii] = np.where(values == 0, 0, fi)
                cid[ii], prob[ii], val[ii] = ci, P[np.arange(len(ii)), ci], values
        fields[split] = {'class_id': cid, 'probability': prob, 'feature_id': fid, 'contribution': val}
    return fields


def fit_prototypes(M, y, classes):
    M = np.asarray(M, np.float32)
    y = np.asarray(y, str)
    std = M.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.
    means = np.stack([M[y == str(k)].mean(axis=0, dtype=np.float64) for k in classes])
    if not np.isfinite(means).all() or not np.isfinite(std).all():
        raise ValueError('Invalid training prototypes/scales.')
    return means, std


def fit_head(M, y, kind, seed, jobs, quick=False):
    if kind != 'logistic':
        return _BASE_FIT_HEAD(M, y, kind, seed, jobs, quick)
    # An explicit seed is the one disclosed reproducibility improvement over
    # the old liblinear random_state=None. Every matched alternative uses it.
    head = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=1.,
               class_weight='balanced', max_iter=1000, random_state=int(seed) + 500), n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        head.fit(M, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('Coordinator did not converge; no successful result is recorded.')
    return head


def entropy(p):
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid probabilities in entropy.')
    return -xlogy(p, p).sum(axis=1) / np.log(max(2, p.shape[1]))


def quantization_audit(M24, M16, head, proto, std, classes, data):
    """Analytic propagation for calibration AND test; fixed learned functions."""
    raw24, pred24 = c.raw_components(M24, head, proto, std, classes)
    raw16, pred16 = c.raw_components(M16, head, proto, std, classes)
    s24, normalizers, _ = c.normalized_scores(raw24)
    s16 = {s: c.combine(np.column_stack([normalizers[j](v[:, j]) for j in range(3)])) for s, v in raw16.items()}
    bound, certified, input_error = {}, {}, {}
    tolerance = 1e-8
    for split in ['cal', 'known', 'unknown']:
        lo, hi, cert, error = c.prediction_bounds(M24[split], M16[split], head, proto, std, normalizers, classes)
        if (not np.isfinite(lo).all() or not np.isfinite(hi).all() or
                np.any(s16[split] < lo - tolerance) or np.any(s16[split] > hi + tolerance)):
            raise AssertionError(f'Quantization interval failed on {split}.')
        if np.any(cert & (pred24[split] != pred16[split])):
            raise AssertionError('Certified class changed after quantization.')
        bound[split] = np.maximum(abs(lo - s24[split]), abs(hi - s24[split]))
        certified[split], input_error[split] = cert, error
    empirical_cal_error = float(np.max(abs(s24['cal'] - s16['cal'])))
    analytic_cal_error = float(np.max(bound['cal']))
    reports = []
    for split in ['known', 'unknown']:
        for alpha in c.LEVELS:
            t24, t16 = c.conformal_cutoff(s24['cal'], alpha), c.conformal_cutoff(s16['cal'], alpha)
            if not math.isfinite(t24) or not math.isfinite(t16):
                raise ValueError('Insufficient calibration size for the requested diagnostic.')
            d24, d16 = s24[split] > t24, s16[split] > t16
            np.testing.assert_array_equal(d24, c.conformal_p(s24['cal'], s24[split]) <= alpha)
            np.testing.assert_array_equal(d16, c.conformal_p(s16['cal'], s16[split]) <= alpha)
            possible = abs(s24[split] - t24) <= bound[split] + analytic_cal_error + tolerance
            changed = d24 != d16
            if (np.any(changed & ~possible) or
                    abs(t24 - t16) > empirical_cal_error + tolerance or
                    empirical_cal_error > analytic_cal_error + tolerance):
                raise AssertionError('Calibrated decision-stability check failed.')
            reports.append({'population': split, 'alpha': alpha, 'n': len(d24),
                'max_input_rounding_error': float(input_error[split].max()),
                'max_score_change': float(np.max(abs(s24[split] - s16[split]))),
                'max_score_bound': float(bound[split].max()), 'mean_score_bound': float(bound[split].mean()),
                'calibration_max_score_change': empirical_cal_error,
                'calibration_analytic_max_score_bound': analytic_cal_error,
                'threshold_shift': abs(t24 - t16),
                'prediction_change_rate': float(np.mean(pred24[split] != pred16[split])),
                'class_margin_certified_rate': float(certified[split].mean()),
                'decision_change_rate': float(changed.mean()),
                'decision_change_bound_rate': float(possible.mean()),
                'decision_rate_float32': float(d24.mean()), 'decision_rate_binary16': float(d16.mean()),
                'numerical_bound_violations': 0, 'numerical_tolerance': tolerance,
                'scope': 'same float32-trained head/prototypes/normalizers; analytic calibration and test intervals; no model refit or identifier change'})
    return reports


def install(root, fixture=False):
    root = Path(root).resolve()
    if fixture:
        c.prepare = prepare_fixture
    else:
        r2 = verify_and_load(root)
        def prepare(csv, holdout, seed, fixture=False):
            if fixture:
                raise RuntimeError('A real-data run cannot switch to fixture mode.')
            cfg = root / 'configs' / 'holdouts' / f'real_5g_nidd_{holdout.lower()}.yaml'
            source = reference.load_yaml(cfg)
            target = Path(source['dataset']['path']).expanduser()
            if not target.is_absolute():
                target = root / target
            if target.resolve() != Path(csv).resolve():
                raise ValueError(f'Dataset path mismatch in {cfg}')
            data = convert_prepared(r2.prepare_round2_data(cfg, seed), seed)
            if data['class_counts'] != c.CLASS_COUNTS:
                raise ValueError('Unexpected primary dataset class counts.')
            return data
        c.prepare = prepare
    c.source_fields = source_fields
    c.fit_prototypes = fit_prototypes
    c.fit_head = fit_head
    c.quantization_audit = quantization_audit
    reference.entropy_score = entropy
    c.VERSION = SNAPSHOT_VERSION
    c.CONFIG.update({
        'normalization_calibration_split': 'default_rng(seed+9017).permutation(n_val); first floor(n/2) normalization, rest calibration',
        'local_class_ties': 'argmax, as executed in Round-2 packet_message',
        'proxy_rounding': 'selected value first stored float32, then packet scalar conversion',
        'prototype_accumulation': 'float64 std and mean from decoded float32 training messages',
        'entropy_zero_convention': 'scipy.special.xlogy with exact zero probabilities',
        'logistic_random_state': 'seed+500; explicit reproducibility improvement versus old None',
        'executed_round2_source_sha256': EXPECTED,
        'fixed_fpr_primary': 'identical realized integer false-positive count; predeclared row-position tie order',
        'fixed_fpr_secondary': 'boundary-randomized expected target and scalar strict/inclusive endpoints',
        'quantization_calibration_bound': 'analytic propagated interval, not only observed calibration drift',
    })
