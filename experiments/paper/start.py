#!/usr/bin/env python3
"""Supported entry point: verify the executed snapshot, reuse scores, then extend.

All generated files belong to new runs/mdpi_r3 and results/mdpi_r3 paths.
Old runs, datasets, configurations and scripts are read-only inputs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import core as c
import run as runner
import snapshot_protocol as snapshot
from exact_fpr import empirical_point


def reuse_round2(root, csv, output_name):
    """Exactly retained composite scores; no source/classifier fit in this stage."""
    out = root / 'results' / output_name
    existing_plan = out / 'PLAN.json'
    if existing_plan.exists() and runner.load_json(existing_plan)['version'] != c.VERSION:
        raise RuntimeError('An older extension protocol already occupies this output directory. Do not mix versions.')
    if c.sha256(csv) != c.DATASET_SHA256:
        raise RuntimeError('The CSV differs from the completed Round-2 dataset; no training started.')
    inventory = []
    for seed in c.SEEDS:
        for h in c.HOLDOUTS:
            base = root / 'runs' / 'mdpi_r2' / f'seed{seed}' / f'real_5g_nidd_{h.lower()}'
            required = [base / 'round2_scores.npz', base / 'COMPLETE.json', base / 'protocol.json']
            if any(not p.is_file() for p in required):
                raise FileNotFoundError(f'Missing completed Round-2 inputs in {base}. No model has been trained by this extension.')
            marker = runner.load_json(required[1])
            expected = {'dataset_sha256': c.DATASET_SHA256,
                        'reference_core_sha256': snapshot.EXPECTED['mdpi_revision_common.py'],
                        'audit_core_sha256': snapshot.EXPECTED['xmag_round2_core.py'],
                        'runner_sha256': snapshot.EXPECTED['xmag_round2_run.py']}
            if any(marker.get(k) != v for k, v in expected.items()):
                raise RuntimeError(f'Unexpected provenance in {required[1]}')
            config = root / 'configs' / 'holdouts' / f'real_5g_nidd_{h.lower()}.yaml'
            if marker.get('configuration_sha256') != c.sha256(config):
                raise RuntimeError(f'Configuration changed since Round-2: {config}')
            inventory.append({'seed': seed, 'held_out_attack': h,
                              'path': str(required[0].relative_to(root)),
                              'sha256': c.sha256(required[0])})
    signature = hashlib.sha256(json.dumps({'input_scores': inventory, 'protocol': c.VERSION,
        'analysis_code': {name: c.sha256(Path(__file__).with_name(name)) for name in ['start.py', 'core.py', 'exact_fpr.py']}}, sort_keys=True).encode()).hexdigest()
    done = out / 'RETAINED_SCORE_REUSE_COMPLETE.json'
    if done.exists():
        d = runner.load_json(done)
        if d.get('signature') != signature:
            raise RuntimeError('The retained-score reuse fingerprint changed. Existing output is preserved.')
        if not all((out / f).is_file() and c.sha256(out / f) == digest for f, digest in d['outputs'].items()):
            raise RuntimeError('Retained-score analysis output is missing or changed.')
        runner.log('Retained-score analysis already complete: 40 archives reused, no training.')
        return
    runner.log('Reusing exact Round-2 score arrays for 12B, 16Q and 24B before new fitting.')
    y = pd.read_csv(csv, usecols=['Attack Type'], low_memory=False)['Attack Type'].astype(str).str.strip().to_numpy()
    ids = np.arange(len(y))
    fixed_rows, operating_rows, checks = [], [], []
    for i, rec in enumerate(inventory, 1):
        seed, h = rec['seed'], rec['held_out_attack']
        all_known = ids[y != h]
        tv, kt = train_test_split(all_known, test_size=.30, random_state=seed, stratify=y[all_known])
        _, va = train_test_split(tv, test_size=.20, random_state=seed, stratify=y[tv])
        _, ci = snapshot.validation_positions(len(va), seed)
        yk = y[kt]
        mask = yk == 'Benign'
        with np.load(root / rec['path'], allow_pickle=False) as scores:
            np.testing.assert_array_equal(scores['known_labels'].astype(str), yk)
            for layout in ['12B', '16Q', '24B']:
                cal, known, unknown = [c.clean_array(scores[f'{layout}_{s}']) for s in ['cal', 'known', 'unknown']]
                if len(cal) != len(ci) or len(known) != len(kt) or len(unknown) != int(np.sum(y == h)):
                    raise AssertionError('Retained score sizes do not match the executed split.')
                pk, pu = c.conformal_p(cal, known), c.conformal_p(cal, unknown)
                common = {'method': 'retained_round2_' + layout, 'seed': seed, 'held_out_attack': h,
                          'source': 'unmodified_saved_round2_scores', 'score_archive_sha256': rec['sha256']}
                for alpha in c.LEVELS:
                    operating_rows.append({**common, 'alpha': alpha,
                        'unknown_recall': float(np.mean(pu <= alpha)),
                        'known_false_rejection_rate': float(np.mean(pk <= alpha)),
                        'benign_false_rejection_rate': float(np.mean(pk[mask] <= alpha)),
                        'n_calibration': len(cal), 'n_known': len(known), 'n_unknown': len(unknown)})
                    for denominator, negative in [('all_known', known), ('benign_only', known[mask])]:
                        fixed_rows.append({**common, 'denominator': denominator,
                            **c.fixed_fpr(negative, unknown, alpha), **empirical_point(negative, unknown, alpha)})
        checks.append({**rec, 'known_row_order_checked_against_original_split': True,
                       'calibration_rows': len(ci), 'known_rows': len(kt), 'unknown_rows': int(np.sum(y == h))})
        runner.log(f'Retained score analysis {i}/40: seed={seed}, {h}')
    fixed = pd.DataFrame(fixed_rows)
    if len(fixed) != 720 or fixed.duplicated(['seed', 'held_out_attack', 'method', 'denominator', 'target_fpr']).any():
        raise AssertionError('Incomplete or duplicate retained fixed-FPR analysis.')
    spread = fixed.groupby(['seed', 'held_out_attack', 'denominator', 'target_fpr']).empirical_fpr.agg(['min', 'max'])
    if not (spread['min'] == spread['max']).all():
        raise AssertionError('Retained formats have unmatched realized FPR.')
    fixed.to_csv(out / 'retained_round2_fixed_fpr.csv', index=False)
    pd.DataFrame(operating_rows).to_csv(out / 'retained_round2_nominal.csv', index=False)
    summary = fixed.groupby(['method', 'denominator', 'target_fpr'])[['empirical_fpr', 'empirical_recall', 'expected_recall']].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(k) for k in summary.columns]
    summary.reset_index().to_csv(out / 'retained_round2_fixed_fpr_summary.csv', index=False)
    c.atomic_json(out / 'retained_round2_score_inventory.json', checks)
    names = ['retained_round2_fixed_fpr.csv', 'retained_round2_nominal.csv',
             'retained_round2_fixed_fpr_summary.csv', 'retained_round2_score_inventory.json']
    c.atomic_json(done, {'signature': signature, 'score_archives': 40, 'fixed_fpr_rows': 720,
                       'new_models_trained_for_this_stage': 0,
                       'outputs': {n: c.sha256(out / n) for n in names}})
    runner.log('Existing-score comparison complete. New fitting is needed only for replacement controls and diagnostics.')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--csv', type=Path, default=Path('data/5G-NIDD/Encoded.csv'))
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--pack-only', action='store_true')
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--output-name', default='mdpi_r3')
    args, _ = parser.parse_known_args()
    if '-h' in sys.argv or '--help' in sys.argv:
        runner.main()
        return
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f'Workspace does not exist: {root}')
    os.chdir(root)
    if args.status:
        runner.main()
        return
    if not args.output_name.replace('_', '').isalnum() or (args.fixture and args.output_name == 'mdpi_r3'):
        raise SystemExit('Invalid output scope; fixtures must use a separate output name.')
    out = root / 'results' / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.launcher.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A snapshot-matched extension is already running in this output scope.')
    try:
        snapshot.install(root, fixture=args.fixture)
        if not args.fixture:
            c.CONFIG['local_holdout_configuration_sha256'] = {
                h: c.sha256(root / 'configs' / 'holdouts' / f'real_5g_nidd_{h.lower()}.yaml') for h in c.HOLDOUTS}
            if not args.pack_only:
                csv = args.csv.expanduser()
                if not csv.is_absolute():
                    csv = root / csv
                reuse_round2(root, csv, args.output_name)
        runner.main()
    except BaseException:
        import traceback
        (out / 'launcher_failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
