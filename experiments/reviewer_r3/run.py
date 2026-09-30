#!/usr/bin/env python3
"""One-command, resumable reviewer extension for the existing Mac workspace."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

import core as c
import reference

HERE = Path(__file__).resolve().parent


def log(text):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + '  ' + text, flush=True)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def completed(folder, signature):
    marker = folder / 'COMPLETE.json'
    if not marker.exists():
        return False
    d = load_json(marker)
    if d.get('signature') != signature:
        raise RuntimeError(f'Fingerprint mismatch in {folder}; existing results were not overwritten.')
    for name, digest in d['files'].items():
        p = folder / name
        if not p.is_file() or c.sha256(p) != digest:
            raise RuntimeError(f'Missing/corrupt completed artifact: {p}')
    return True


def save_result(folder, method, scores, preds, data, payload, component, extra=None):
    p = folder / (method + '.json')
    arrays = folder / 'scores' / (method + '.npz')
    benign = data['y']['known'] == 'Benign'
    nominal, fixed = c.evaluate_scores(scores['cal'], scores['known'], scores['unknown'],
                                      benign, data['y']['known'], preds['known'])
    common = {'method': method, 'seed': int(data['seed']), 'held_out_attack': data['holdout'],
              'message_bytes_per_flow': payload, 'changed_component': component,
              'protocol': c.VERSION}
    for rows in (nominal, fixed):
        for row in rows:
            row.update(common)
    c.atomic_npz(arrays, calibration_score=scores['cal'], known_score=scores['known'],
                 unknown_score=scores['unknown'], benign_mask=benign,
                 known_pred=preds['known'], known_y=data['y']['known'])
    c.atomic_json(p, {'nominal': nominal, 'fixed_fpr': fixed, 'extra': extra or {},
                      'score_file_sha256': c.sha256(arrays)})
    log(f"  saved {method}: AUROC={nominal[-1]['unknown_auroc']:.6f}; matched-FPR rows={len(fixed)}")


def result_ready(folder, method):
    p = folder / (method + '.json')
    score = folder / 'scores' / (method + '.npz')
    if not p.exists():
        return False
    rec = load_json(p)
    if not score.exists() or c.sha256(score) != rec['score_file_sha256']:
        raise RuntimeError(f'Incomplete score cache {score}; retain the log rather than mixing results.')
    return True


def load_or_make(path, maker):
    if path.exists():
        return joblib.load(path)
    value = maker()
    c.atomic_joblib(path, value)
    return value


def collect_messages(data, fields, anomaly, layout):
    return {s: c.packet_roundtrip(fields[s], anomaly[s], layout,
                                 len(data['classes']), data['X']['train'].shape[1]) for s in c.SPLITS}


def evaluate_family_variant(folder, data, messages, head_name, method, component, args):
    head_path = folder / 'models' / (head_name + '.joblib')
    kind = 'et' if head_name == 'coordinator_et16' else 'logistic'
    t0 = time.monotonic()
    head = c.cached_model(head_path, lambda: c.fit_head(messages['train'], data['y']['train'], kind,
                                                        data['seed'], args.jobs, args.fixture))
    proto, scale = c.fit_prototypes(messages['train'], data['y']['train'], data['classes'])
    raw, preds = c.raw_components(messages, head, proto, scale, data['classes'])
    scores, norms, normalized = c.normalized_scores(raw)
    payload = 12 if method == 'control_12b' else 24 if method == 'control_24b' else 16
    if not result_ready(folder, method):
        save_result(folder, method, scores, preds, data, payload, component,
                    {'stage_seconds_this_invocation': time.monotonic() - t0,
                     'coordinator_serialized_bytes': head_path.stat().st_size})
    return head, proto, scale, raw, preds, scores, normalized


def run_trial(root, datafile, folder, holdout, seed, signature, args):
    folder.mkdir(parents=True, exist_ok=True)
    if completed(folder, signature):
        log(f'SKIP complete seed={seed} holdout={holdout}')
        return
    started = time.monotonic()
    fingerprint = folder / 'FINGERPRINT.json'
    if fingerprint.exists() and load_json(fingerprint)['signature'] != signature:
        raise RuntimeError(f'Changed experiment fingerprint: {folder}. Previous files left intact.')
    c.atomic_json(fingerprint, {'signature': signature, 'seed': seed, 'holdout': holdout})
    status = root / 'results' / args.output_name / 'STATUS.json'
    def stage(name):
        c.atomic_json(status, {'status': 'running', 'pid': os.getpid(), 'seed': seed,
                              'holdout': holdout, 'stage': name, 'updated': time.time()})
        log(f'seed={seed} holdout={holdout} | {name}')
    stage('training-only preprocessing and disjoint splits')
    data = c.prepare(datafile, holdout, seed, args.fixture)
    c.atomic_json(folder / 'split_manifest.json', c.split_manifest(data))
    c.atomic_npz(folder / 'split_ids.npz', **data['ids'])
    c.atomic_joblib(folder / 'models' / 'preprocessor.joblib', data['preprocessor'])
    stage('fit/reuse local RF and Isolation Forest')
    rf = c.cached_model(folder / 'models' / 'local_rf.joblib', lambda: c.fit_sources(data, 'rf', args.jobs, args.fixture))
    iso = c.cached_model(folder / 'models' / 'local_if.joblib', lambda: c.fit_sources(data, 'if', args.jobs, args.fixture))
    fields_rf = load_or_make(folder / 'cache_rf_fields.joblib', lambda: c.source_fields(data, rf['models']))
    anomaly_if, ifscale = load_or_make(folder / 'cache_if_fields.joblib', lambda: c.anomaly_fields(data, iso['models'], 'if'))
    c.atomic_json(folder / 'base_training.json', {'rf': rf['log'], 'if': iso['log'], 'if_scalar_normalizer': ifscale})

    stage('16Q reference and score/fusion replacements')
    M16 = collect_messages(data, fields_rf, anomaly_if, '16q')
    head16, proto16, scale16, raw16, pred16, scores16, norm16 = evaluate_family_variant(
        folder, data, M16, 'coordinator_logistic16', 'baseline_16q', 'none', args)
    for method, rule in [('fusion_mean_16q', 'mean'), ('fusion_max_16q', 'max')]:
        if not result_ready(folder, method):
            score = {s: c.combine(a, rule) for s, a in norm16.items()}
            save_result(folder, method, score, pred16, data, 16, 'fusion')
    if not result_ready(folder, 'uncertainty_16q'):
        save_result(folder, 'uncertainty_16q', {s: a[:, 0] for s, a in raw16.items()},
                    pred16, data, 16, 'rejection_score_control')
    if not result_ready(folder, 'residual_l1_16q'):
        raw_l1, pred_l1 = c.raw_components(M16, head16, proto16, scale16, data['classes'], 'l1')
        score_l1, _, _ = c.normalized_scores(raw_l1)
        save_result(folder, 'residual_l1_16q', score_l1, pred_l1, data, 16, 'prototype_residual_metric')
    stage('replace logistic coordinator by bounded Extra Trees')
    if not result_ready(folder, 'coordinator_extratrees_16q'):
        evaluate_family_variant(folder, data, M16, 'coordinator_et16', 'coordinator_extratrees_16q',
                                'coordinator_classifier', args)

    stage('same-split 12-byte and 24-byte controls')
    if not result_ready(folder, 'control_12b'):
        M12 = collect_messages(data, fields_rf, anomaly_if, '12b')
        evaluate_family_variant(folder, data, M12, 'coordinator_logistic12', 'control_12b', 'message_layout', args)
        del M12
    M24 = collect_messages(data, fields_rf, anomaly_if, '24b')
    head24, proto24, scale24, _, _, _, _ = evaluate_family_variant(
        folder, data, M24, 'coordinator_logistic24', 'control_24b', 'message_layout', args)
    stage('fixed-model float32-to-binary16 decision-stability diagnostic')
    if not (folder / 'quantization_diagnostics.json').exists():
        # The mathematical diagnostic evaluates its fixed numerical operator in
        # float64; primary message matrices remain the specified float32 decoder.
        Q24 = {s: (v if s == 'train' else v.astype(np.float64)) for s, v in M24.items()}
        Q16 = {s: (v if s == 'train' else v.astype(np.float64)) for s, v in M16.items()}
        quant = c.quantization_audit(Q24, Q16, head24, proto24, scale24, data['classes'], data)
        c.atomic_json(folder / 'quantization_diagnostics.json', quant)
        del Q24, Q16
    del M16, M24, raw16, norm16, scores16
    gc.collect()

    stage('replace local Random Forest by Extra Trees')
    if not result_ready(folder, 'local_extratrees_16q'):
        et = c.cached_model(folder / 'models' / 'local_et.joblib', lambda: c.fit_sources(data, 'et', args.jobs, args.fixture))
        fields_et = load_or_make(folder / 'cache_et_fields.joblib', lambda: c.source_fields(data, et['models']))
        Met = collect_messages(data, fields_et, anomaly_if, '16q')
        evaluate_family_variant(folder, data, Met, 'coordinator_local_et', 'local_extratrees_16q',
                                'local_classifier_and_its_corresponding_importance_proxy', args)
        c.atomic_json(folder / 'local_et_training.json', et['log'])
        del Met, fields_et, et
        gc.collect()

    stage('replace Isolation Forest by diagonal Gaussian mixture')
    if not result_ready(folder, 'anomaly_gmm_16q'):
        gmm = c.cached_model(folder / 'models' / 'local_gmm.joblib', lambda: c.fit_sources(data, 'gmm', args.jobs, args.fixture))
        anomaly_gmm, gscale = load_or_make(folder / 'cache_gmm_fields.joblib', lambda: c.anomaly_fields(data, gmm['models'], 'gmm'))
        Mgmm = collect_messages(data, fields_rf, anomaly_gmm, '16q')
        evaluate_family_variant(folder, data, Mgmm, 'coordinator_gmm', 'anomaly_gmm_16q', 'local_anomaly_detector', args)
        c.atomic_json(folder / 'gmm_training.json', {'models': gmm['log'], 'scalar_normalizer': gscale})
        del Mgmm, anomaly_gmm, gmm
        gc.collect()

    stage('centralized full-feature RF control and fixed-FPR comparisons')
    if not result_ready(folder, 'central_rf'):
        def fit_central():
            m = RandomForestClassifier(n_estimators=3 if args.fixture else 30, min_samples_leaf=2,
                    class_weight='balanced_subsample', random_state=seed, n_jobs=args.jobs)
            m.fit(data['X']['train'], data['y']['train'])
            m.set_params(n_jobs=1)
            return m
        central = c.cached_model(folder / 'models' / 'central_rf.joblib', fit_central)
        score, pred = {}, {}
        for s in ['norm', 'cal', 'known', 'unknown']:
            p = reference.align_proba(central, data['X'][s], data['classes'])
            score[s] = reference.entropy_score(p)
            pred[s] = np.asarray(data['classes'])[p.argmax(axis=1)]
        save_result(folder, 'central_rf', score, pred, data, 4 * data['X']['train'].shape[1], 'full_feature_control')
    for method in c.METHODS:
        if not result_ready(folder, method):
            raise RuntimeError('Missing required method: ' + method)
    files = [m + '.json' for m in c.METHODS] + ['split_manifest.json', 'quantization_diagnostics.json', 'base_training.json']
    c.atomic_json(folder / 'COMPLETE.json', {'signature': signature, 'seed': seed, 'holdout': holdout,
                    'methods': c.METHODS, 'elapsed_seconds_this_invocation': time.monotonic() - started,
                    'files': {name: c.sha256(folder / name) for name in files}})
    log(f'COMPLETE seed={seed} holdout={holdout}; {len(c.METHODS)} method/score conditions')


def report_status(root, name):
    out = root / 'results' / name
    runroot = root / 'runs' / name
    plan_path = out / 'PLAN.json'
    expected = len(load_json(plan_path)['tasks']) if plan_path.exists() else 40
    done = list(runroot.glob('seed*/**/COMPLETE.json'))
    print(f'Completed tasks: {len(done)}/{expected}')
    status = out / 'STATUS.json'
    if status.exists():
        print(status.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.cwd())
    ap.add_argument('--csv', type=Path, default=Path('data/5G-NIDD/Encoded.csv'))
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--pack-only', action='store_true')
    ap.add_argument('--fixture', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--output-name', default='mdpi_r3')
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    if args.jobs < 1 or args.jobs > 16:
        ap.error('--jobs must lie between 1 and 16')
    if not args.output_name.replace('_', '').isalnum():
        ap.error('output-name must be an alphanumeric name with optional underscores')
    if args.fixture and args.output_name == 'mdpi_r3':
        ap.error('Fixtures must have a separate output-name, never mdpi_r3')
    if args.status:
        report_status(root, args.output_name)
        return
    out, runroot = root / 'results' / args.output_name, root / 'runs' / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    runroot.mkdir(parents=True, exist_ok=True)
    lock = (out / '.run.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another extension process holds the output lock. Do not start duplicate runs.')
    if args.pack_only:
        import analysis
        analysis.finish(root, args.output_name, HERE)
        return
    csvpath = args.csv.expanduser()
    if not csvpath.is_absolute():
        csvpath = root / csvpath
    if not csvpath.is_file():
        raise SystemExit(f'Missing real dataset: {csvpath}')
    if not args.fixture and shutil.disk_usage(root).free < 10 * 1024**3:
        raise SystemExit('At least 10 GiB free space is required for resumable models and exact score arrays.')
    log('Fingerprinting the real CSV; no dataset download or modification')
    dhash = c.sha256(csvpath)
    if not args.fixture and dhash != c.DATASET_SHA256:
        raise SystemExit('CSV checksum does not match the completed Round-2 dataset. No experiment started; share this message rather than bypassing it.')
    source_hashes = {p.name: c.sha256(p) for p in HERE.glob('*.py')}
    env = {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
           'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__,
           'platform': platform.platform(), 'processor': platform.processor(), 'jobs': args.jobs}
    tasks = [(7, 'UDPFlood')] if args.fixture else [(s, h) for s in c.SEEDS for h in c.HOLDOUTS]
    description = {'version': c.VERSION, 'dataset_sha256': dhash, 'source_hashes': source_hashes,
                   'environment': env, 'configuration': c.CONFIG, 'methods': c.METHODS,
                   'tasks': [list(t) for t in tasks], 'fixture': args.fixture,
                   'protocol_note': 'Fresh matched extension. Shared components cached within each task. Historical tables are not overwritten and are not assumed bit-identical.'}
    signature = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
    description['signature'] = signature
    plan_path = out / 'PLAN.json'
    if plan_path.exists() and load_json(plan_path)['signature'] != signature:
        raise SystemExit('Code, data, settings, or environment changed since this suite began. Existing results are untouched; do not mix fingerprints.')
    c.atomic_json(plan_path, description)
    snap = out / 'code'
    snap.mkdir(exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in {'.py', '.txt', '.md'}:
            shutil.copy2(p, snap / p.name)
    # Preserve exact local execution sources without uploading data, models, or credentials.
    provenance = out / 'local_round2_source'
    provenance.mkdir(exist_ok=True)
    for p in sorted((root / 'scripts').glob('xmag_round2*.py')):
        shutil.copy2(p, provenance / p.name)
    for name in ['run_xmag_round2.sh', 'mdpi_revision_common.py']:
        p = root / 'scripts' / name
        if p.is_file():
            shutil.copy2(p, provenance / name)
    (out / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    try:
        with threadpool_limits(limits=max(1, min(args.jobs, 4))):
            for i, (seed, holdout) in enumerate(tasks, 1):
                log(f'TASK {i}/{len(tasks)} | seed={seed} holdout={holdout}')
                folder = runroot / f'seed{seed}' / holdout
                run_trial(root, csvpath, folder, holdout, seed, signature, args)
                gc.collect()
        import analysis
        analysis.finish(root, args.output_name, HERE)
        c.atomic_json(out / 'STATUS.json', {'status': 'complete', 'completed_tasks': len(tasks),
                                          'updated': time.time(), 'fixture': args.fixture})
        log(f'Completed. Share: results/{args.output_name}_review_text.txt')
        log(f'Full companion archive: results/{args.output_name}_reviewer_results.zip')
    except BaseException as exc:
        c.atomic_json(out / 'STATUS.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                 'error': str(exc), 'updated': time.time(), 'pid': os.getpid()})
        (out / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
