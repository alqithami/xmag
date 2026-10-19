#!/usr/bin/env python3
"""Completeness, equal-realized-FPR statistics, reconciliation, and packaging."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import core as c
from exact_fpr import empirical_point, TIE_SEED


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty(len(p))
    out[order] = np.minimum(1.0, adjusted)
    return out


def paired_statistics(fixed, plan):
    rows = []
    for denom in c.CONFIG['fixed_fpr_denominators']:
        for family_name, methods in [('primary_replacements', c.REPLACEMENTS), ('reference_controls', c.CONTROLS)]:
            for method in methods:
                for target in c.LEVELS:
                    f = fixed[(fixed.denominator == denom) & np.isclose(fixed.target_fpr, target)]
                    base = f[f.method == 'baseline_16q'].set_index(['seed', 'held_out_attack'])
                    alt = f[f.method == method].set_index(['seed', 'held_out_attack'])
                    if set(base.index) != set(alt.index):
                        raise ValueError('Unpaired trial coverage in fixed-FPR comparison.')
                    base, alt = base.sort_index(), alt.sort_index()
                    if not np.array_equal(base.empirical_fpr.to_numpy(), alt.empirical_fpr.to_numpy()):
                        raise ValueError('Compared methods do not have identical actual FPR.')
                    delta = (alt.empirical_recall - base.empirical_recall).round(12)
                    x = delta.to_numpy()
                    block = delta.groupby(level='held_out_attack').mean().to_numpy()
                    key = f'{denom}/{method}/{target}'
                    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
                    ci_trial = np.quantile(x[rng.integers(0, len(x), size=(20000, len(x)))].mean(axis=1), [.025, .975])
                    ci_block = np.quantile(block[rng.integers(0, len(block), size=(20000, len(block)))].mean(axis=1), [.025, .975])
                    nz = x[x != 0]
                    if not len(nz):
                        pw, pt, effect = 1.0, 1.0, 0.0
                    else:
                        pw = float(stats.wilcoxon(x, zero_method='wilcox', method='auto').pvalue)
                        pt = float(stats.ttest_1samp(x, 0).pvalue) if len(x) > 1 and np.std(x) > 0 else 0.0
                        ranks = stats.rankdata(np.abs(nz))
                        effect = float(np.sum(ranks * np.sign(nz)) / ranks.sum())
                    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(block))))
                    psign = float(np.mean(np.abs((signs * block).mean(axis=1)) >= abs(block.mean()) - 1e-14))
                    rows.append({'comparison_family': family_name, 'denominator': denom,
                                 'operating_metric': 'empirical_recall_at_identical_realized_fpr',
                                 'method_minus_baseline': method, 'target_fpr': target,
                                 'n_pairs': len(x), 'n_holdout_blocks': len(block),
                                 'mean_delta_recall': float(x.mean()), 'median_delta_recall': float(np.median(x)),
                                 'trial_ci_low': float(ci_trial[0]), 'trial_ci_high': float(ci_trial[1]),
                                 'family_ci_low': float(ci_block[0]), 'family_ci_high': float(ci_block[1]),
                                 'wilcoxon_p_raw': pw, 'paired_t_p_raw': pt, 'family_signflip_p_raw': psign,
                                 'rank_biserial': effect, 'n_positive': int((x > 0).sum()),
                                 'n_negative': int((x < 0).sum()), 'n_zero': int((x == 0).sum()),
                                 'exploratory_fixture': plan['fixture']})
    df = pd.DataFrame(rows)
    for _, ix in df.groupby(['comparison_family', 'denominator']).groups.items():
        for col in ['wilcoxon_p_raw', 'paired_t_p_raw', 'family_signflip_p_raw']:
            df.loc[ix, col.replace('_raw', '_holm')] = holm(df.loc[ix, col])
        df.loc[ix, 'multiplicity_family_size'] = len(ix)
    return df


def reconcile(root, nominal, out):
    """Compare regenerated controls without asserting historical equivalence."""
    p = root / 'results/mdpi_r2/all_metrics_round2.csv'
    old, source = None, None
    if p.exists():
        old, source = pd.read_csv(p), str(p.relative_to(root))
    else:
        zpath = root / 'results/xmag_round2_audit_results.zip'
        if zpath.exists():
            with zipfile.ZipFile(zpath) as z:
                names = [n for n in z.namelist() if n.endswith('/all_metrics_round2.csv')]
                if len(names) == 1:
                    old = pd.read_csv(io.BytesIO(z.read(names[0])))
                    source = str(zpath.relative_to(root)) + ':' + names[0]
    if old is None:
        c.atomic_json(out / 'round2_reconciliation.json', {'status': 'historical_rows_not_available_locally',
             'interpretation': 'Use new internally matched comparisons; do not relabel historical results.'})
        return
    mappings = {'baseline_16q': ('X-MAG-COS-16Q', 'composite'),
                'control_12b': ('Class+anomaly-12B', 'composite'),
                'control_24b': ('X-MAG-COS-24B', 'composite'), 'central_rf': ('Central-RF-full', 'entropy')}
    results = []
    for new_method, (old_method, old_score) in mappings.items():
        a = nominal[nominal.method == new_method]
        b = old[(old.method == old_method) & (old.score == old_score)]
        keys = ['seed', 'held_out_attack', 'alpha']
        if b.duplicated(keys).any():
            raise ValueError('Duplicate historical keys; reconciliation refused.')
        joined = a.merge(b, on=keys, suffixes=('_extension', '_round2'), validate='one_to_one')
        for _, row in joined.iterrows():
            rec = {k: row[k] for k in keys}
            rec['method'] = new_method
            for metric in ['known_macro_f1', 'unknown_auroc', 'unknown_recall', 'known_false_rejection_rate']:
                rec[metric + '_extension'] = float(row[metric + '_extension'])
                rec[metric + '_round2'] = float(row[metric + '_round2'])
                rec[metric + '_delta'] = rec[metric + '_extension'] - rec[metric + '_round2']
            results.append(rec)
    pd.DataFrame(results).to_csv(out / 'round2_control_reconciliation.csv', index=False)
    c.atomic_json(out / 'round2_reconciliation.json', {'status': 'reported_without_forcing_agreement',
        'historical_source': source, 'paired_rows': len(results),
        'interpretation': 'Fresh matched extension. Examine discrepancies before manuscript integration; equality is not assumed.'})


def finish(root, name, code_dir):
    out, runroot = root / 'results' / name, root / 'runs' / name
    plan = read_json(out / 'PLAN.json')
    nominal, fixed, quant, splitinfo = [], [], [], []
    print('Verifying saved scores and constructing identical-realized-FPR comparisons...', flush=True)
    for seed, holdout in plan['tasks']:
        folder = runroot / f'seed{seed}' / holdout
        marker = folder / 'COMPLETE.json'
        if not marker.exists():
            raise RuntimeError(f'Task incomplete: seed={seed} holdout={holdout}; no complete archive generated.')
        record = read_json(marker)
        if record['signature'] != plan['signature']:
            raise RuntimeError('Mixed experiment fingerprints.')
        for relative, expected in record['files'].items():
            if c.sha256(folder / relative) != expected:
                raise RuntimeError(f'Changed completed result: {folder / relative}')
        for method in c.METHODS:
            rec = read_json(folder / (method + '.json'))
            score_path = folder / 'scores' / (method + '.npz')
            if not score_path.exists() or c.sha256(score_path) != rec['score_file_sha256']:
                raise RuntimeError(f'Exact score archive missing/corrupt: {score_path}')
            nominal.extend(rec['nominal'])
            with np.load(score_path, allow_pickle=False) as scores:
                known, unknown = scores['known_score'], scores['unknown_score']
                benign = scores['benign_mask'].astype(bool)
                for row in rec['fixed_fpr']:
                    negative = known if row['denominator'] == 'all_known' else known[benign]
                    fixed.append({**row, **empirical_point(negative, unknown, row['target_fpr'])})
        for row in read_json(folder / 'quantization_diagnostics.json'):
            quant.append({'seed': seed, 'held_out_attack': holdout, **row})
        splitinfo.append(read_json(folder / 'split_manifest.json'))
    nominal, fixed, quant = pd.DataFrame(nominal), pd.DataFrame(fixed), pd.DataFrame(quant)
    nk = ['seed', 'held_out_attack', 'method', 'alpha']
    fk = ['seed', 'held_out_attack', 'method', 'denominator', 'target_fpr']
    expected_n = len(plan['tasks']) * len(c.METHODS) * len(c.LEVELS)
    if nominal.duplicated(nk).any() or fixed.duplicated(fk).any() or len(nominal) != expected_n or len(fixed) != 2 * expected_n:
        raise ValueError('Duplicate/missing nominal or fixed-FPR rows.')
    if np.max(np.abs(fixed.expected_fpr - fixed.target_fpr)) > 1e-12:
        raise ValueError('Expected randomized FPR differs from target.')
    if not ((fixed.strict_fpr <= fixed.target_fpr + 1e-12) & (fixed.inclusive_fpr >= fixed.target_fpr - 1e-12)).all():
        raise ValueError('Invalid deterministic FPR brackets.')
    spreads = fixed.groupby(['seed', 'held_out_attack', 'denominator', 'target_fpr']).empirical_fpr.agg(['min', 'max'])
    spread = float((spreads['max'] - spreads['min']).max())
    if spread != 0:
        raise ValueError('Methods were not compared at identical actual empirical FPR.')
    nominal.to_csv(out / 'all_nominal_metrics.csv', index=False)
    fixed.to_csv(out / 'all_fixed_fpr_metrics.csv', index=False)
    quant.to_csv(out / 'quantization_diagnostics.csv', index=False)
    c.atomic_json(out / 'all_split_manifests.json', splitinfo)
    cols = ['known_macro_f1', 'unknown_auroc', 'unknown_recall', 'known_false_rejection_rate',
            'benign_false_rejection_rate', 'unknown_average_precision', 'standardized_partial_auroc_fpr01']
    summary = nominal.groupby(['method', 'alpha'])[cols].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(out / 'nominal_summary.csv', index=False)
    metrics = ['empirical_recall', 'empirical_fpr', 'expected_recall', 'expected_fpr',
               'strict_fpr', 'strict_recall', 'inclusive_fpr', 'inclusive_recall', 'boundary_probability']
    fsum = fixed.groupby(['method', 'denominator', 'target_fpr'])[metrics].agg(['mean', 'std', 'min', 'max'])
    fsum.columns = ['_'.join(x) for x in fsum.columns]
    fsum.reset_index().to_csv(out / 'fixed_fpr_summary.csv', index=False)
    byfamily = fixed.groupby(['held_out_attack', 'method', 'denominator', 'target_fpr'])[['empirical_recall', 'empirical_fpr', 'expected_recall']].agg(['mean', 'std', 'count'])
    byfamily.columns = ['_'.join(x) for x in byfamily.columns]
    byfamily.reset_index().to_csv(out / 'fixed_fpr_by_family.csv', index=False)
    paired_statistics(fixed, plan).to_csv(out / 'paired_fixed_fpr_statistics.csv', index=False)
    reconcile(root, nominal, out)
    c.atomic_json(out / 'RUN_COMPLETENESS.json', {'status': 'software_fixture_only' if plan['fixture'] else 'complete_for_prespecified_extension',
        'expected_tasks': len(plan['tasks']), 'completed_tasks': len(splitinfo),
        'conditions_per_task': len(c.METHODS), 'nominal_rows': len(nominal), 'fixed_fpr_rows': len(fixed),
        'denominators': c.CONFIG['fixed_fpr_denominators'], 'levels': c.LEVELS,
        'max_expected_fpr_error': float(np.max(np.abs(fixed.expected_fpr - fixed.target_fpr))),
        'max_between_method_empirical_fpr_spread': spread, 'fixed_tie_seed': TIE_SEED,
        'quantization_numerical_violations': int(quant.numerical_bound_violations.sum()),
        'raw_scores_retained_locally': True, 'models_retained_locally': True,
        'historical_results_modified': False, 'fingerprint': plan['signature']})
    notes = '''# Interpretation of the reviewer extension\n\nThe primary fixed-FPR comparison uses exactly the same realized false-positive count floor(target*n_negative) for every method within a seed/holdout/denominator. Scores are ordered first; ties are ordered by a predeclared pseudorandom row-position key, identical across methods. The realized FPR is floor(target*n_negative)/n_negative, with its exact denominator and count exported. A requested percentage may not be an attainable finite-sample fraction, but the actual fraction is identical across methods. The tie seed is fixed at 20260930 and is not selected after seeing results. These are retrospective test-ROC boundaries, not deployment thresholds fitted without test data.\n\nCompanion columns report an ordinary strict-score threshold, an inclusive threshold, and boundary randomization. The expected randomized FPR is exactly the target, even when the integer target is unattainable; expected recall and deterministic endpoints are all retained. It is not mislabeled as an observed deterministic result. The primary paired statistics use realized empirical recall, not interpolated recall.\n\nBoth all-known and Benign-only denominators are evaluated. Independent-calibration operating results remain separate. Test observations are never used to fit source models, classifiers, prototypes, or normalizers. Unknown labels do not select weights or replacement configurations. Benign open-set rejection is only one part of the entire IDS false-alert rate.\n\nEach replacement changes one named block. A new local forest also has its own importance proxy. The residual replacement changes standardized RMS distance to standardized mean absolute distance with the same centers. Mean and maximum fusion use the same normalized components. GMM uses at most 16,384 training-only rows per source and eight diagonal-covariance components; the cap is prespecified.\n\nShared reference components are refitted once per task and cached. This extension does not assume undocumented historical model caches have compatible parameters or splits. Inspect round2_control_reconciliation.csv before replacing old manuscript numbers. Differences in prediction accumulation, environment, and the explicit validation-halving rule can matter; all new competitive results remain internally matched.\n\nThe pure quantization check freezes the float32-trained head, prototypes, and normalizers, preserves integer fields, and quantizes both calibration and test packet scalars. The fixed diagnostic operator is evaluated in float64. Analytic sigmoid/OVR intervals, possible predicted-class prototypes, and monotone fusion bound score changes; calibration order-statistic movement and decision-margin coverage are checked. The tolerance is 1e-8, not a directed-rounding machine proof. This is not a pure quantization interpretation of different-content formats or separately retrained heads.\n\nStatistical replication reuses one dataset. The primary multiplicity family consists of six replacements times three FPR levels, separately for each denominator and test. Four controls form separate exploratory families. Family-block intervals and sign-flip tests retain holdout structure. Nonsignificance does not establish equivalence.\n'''
    (out / 'INTERPRETATION.md').write_text(notes, encoding='utf-8')
    excluded = {'STATUS.json', 'failure_traceback.txt'}
    files = [p for p in sorted(out.rglob('*')) if p.is_file() and p.name not in excluded and not p.name.startswith('.')]
    files = [p for p in files if p.suffix in {'.csv', '.json', '.md', '.txt', '.py', '.sh', '.yaml', '.yml'} and p.name != 'MANIFEST.json']
    c.atomic_json(out / 'MANIFEST.json', {'files': {str(p.relative_to(out)): c.sha256(p) for p in files},
                    'excluded': ['raw dataset', 'fitted models', 'per-flow score NPZ files', 'split-ID NPZ files'],
                    'exact_local_scores': str(runroot.relative_to(root))})
    review = root / 'results' / (name + '_review_text.txt')
    review_files = [out / 'RUN_COMPLETENESS.json', out / 'INTERPRETATION.md'] + [p for p in files if p.suffix in {'.csv', '.json'} and p.name != 'RUN_COMPLETENESS.json']
    with review.open('w', encoding='utf-8') as f:
        for p in review_files:
            f.write('\n' + '=' * 80 + '\nFILE: ' + str(p.relative_to(out)) + '\n' + '=' * 80 + '\n')
            f.write(p.read_text(encoding='utf-8'))
            f.write('\n')
    archive = root / 'results' / (name + '_reviewer_results.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files + [out / 'MANIFEST.json']:
            z.write(p, Path(name) / p.relative_to(out))
        z.write(review, review.name)
    print('Verified complete results:', archive, flush=True)
    print('Direct-readable review text:', review, flush=True)
