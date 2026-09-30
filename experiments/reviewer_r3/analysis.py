#!/usr/bin/env python3
"""Completeness checks, paired statistics, reconciliation, and review packaging."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import core as c


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
                        raise ValueError('Unpaired trial coverage in the fixed-FPR comparison.')
                    base, alt = base.sort_index(), alt.sort_index()
                    delta = (alt.expected_recall - base.expected_recall).round(12)
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
    """Compare regenerated controls without pretending historical equivalence."""
    p = root / 'results/mdpi_r2/all_metrics_round2.csv'
    old = None
    source = None
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
             'interpretation': 'Use new internally matched extension comparisons. Do not overwrite or relabel historical results.'})
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
        'interpretation': 'This extension is a fresh matched execution. Examine discrepancies before updating manuscript tables; equality is not assumed.'})


def finish(root, name, code_dir):
    out, runroot = root / 'results' / name, root / 'runs' / name
    plan = read_json(out / 'PLAN.json')
    nominal, fixed, quant, splitinfo = [], [], [], []
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
            fixed.extend(rec['fixed_fpr'])
        for row in read_json(folder / 'quantization_diagnostics.json'):
            quant.append({'seed': seed, 'held_out_attack': holdout, **row})
        splitinfo.append(read_json(folder / 'split_manifest.json'))
    nominal, fixed, quant = pd.DataFrame(nominal), pd.DataFrame(fixed), pd.DataFrame(quant)
    nk = ['seed', 'held_out_attack', 'method', 'alpha']
    fk = ['seed', 'held_out_attack', 'method', 'denominator', 'target_fpr']
    expected_n = len(plan['tasks']) * len(c.METHODS) * len(c.LEVELS)
    if nominal.duplicated(nk).any() or fixed.duplicated(fk).any() or len(nominal) != expected_n or len(fixed) != 2 * expected_n:
        raise ValueError('Duplicate/missing nominal or fixed-FPR result rows.')
    if np.max(np.abs(fixed.expected_fpr - fixed.target_fpr)) > 1e-12:
        raise ValueError('At least one matched expected FPR differs from target.')
    if not ((fixed.strict_fpr <= fixed.target_fpr + 1e-12) & (fixed.inclusive_fpr >= fixed.target_fpr - 1e-12)).all():
        raise ValueError('Missing deterministic bracket around fixed FPR.')
    nominal.to_csv(out / 'all_nominal_metrics.csv', index=False)
    fixed.to_csv(out / 'all_fixed_fpr_metrics.csv', index=False)
    quant.to_csv(out / 'quantization_diagnostics.csv', index=False)
    c.atomic_json(out / 'all_split_manifests.json', splitinfo)
    cols = ['known_macro_f1', 'unknown_auroc', 'unknown_recall', 'known_false_rejection_rate',
            'benign_false_rejection_rate', 'unknown_average_precision', 'standardized_partial_auroc_fpr01']
    summary = nominal.groupby(['method', 'alpha'])[cols].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(out / 'nominal_summary.csv', index=False)
    fsum = fixed.groupby(['method', 'denominator', 'target_fpr'])[['expected_recall', 'expected_fpr', 'strict_fpr', 'strict_recall', 'inclusive_fpr', 'inclusive_recall', 'boundary_probability']].agg(['mean', 'std', 'min', 'max'])
    fsum.columns = ['_'.join(x) for x in fsum.columns]
    fsum.reset_index().to_csv(out / 'fixed_fpr_summary.csv', index=False)
    fixed.groupby(['held_out_attack', 'method', 'denominator', 'target_fpr']).expected_recall.agg(['mean', 'std', 'count']).reset_index().to_csv(out / 'fixed_fpr_by_family.csv', index=False)
    paired_statistics(fixed, plan).to_csv(out / 'paired_fixed_fpr_statistics.csv', index=False)
    reconcile(root, nominal, out)
    c.atomic_json(out / 'RUN_COMPLETENESS.json', {'status': 'software_fixture_only' if plan['fixture'] else 'complete_for_prespecified_extension',
        'expected_tasks': len(plan['tasks']), 'completed_tasks': len(splitinfo),
        'conditions_per_task': len(c.METHODS), 'nominal_rows': len(nominal), 'fixed_fpr_rows': len(fixed),
        'denominators': c.CONFIG['fixed_fpr_denominators'], 'levels': c.LEVELS,
        'max_expected_fpr_error': float(np.max(np.abs(fixed.expected_fpr - fixed.target_fpr))),
        'quantization_numerical_violations': int(quant.numerical_bound_violations.sum()),
        'raw_scores_retained_locally': True, 'models_retained_locally': True,
        'historical_results_modified': False, 'fingerprint': plan['signature']})
    notes = '''# Interpretation of the reviewer extension\n\nThis package contains newly measured component substitutions, matched expected-FPR comparisons, independent-calibration operating results, and fixed-model quantization diagnostics.\n\nFixed-FPR recall is retrospective ROC evaluation: reject above an empirical negative-score boundary and randomize at that boundary with the exported probability. Its FPR equals the target in expectation over randomization. It is not an observed deterministic threshold and it is not a threshold selected independently of the test sample. The strict and inclusive endpoints report the actual attainable deterministic brackets. Separate all-known and Benign-only denominators are retained.\n\nFor deployable operating points, use the independent known calibration rows, not test-FPR boundaries. All detector fitting, upstream refitting after replacements, and component normalization exclude calibration and test observations. Models are not selected by test recall.\n\nEach replacement changes one named component block. Replacing the local forest also changes the importance proxy belonging to that forest; it does not hold that model-dependent quantity artificially fixed. Replacing the residual keeps the same decoded input, classifier, and class prototype centers and changes standardized RMS distance to standardized mean absolute distance. Mean and maximum fusion consume the same three normalized evidence components without additional model training. GMM uses at most 16,384 training-only rows per source and eight diagonal-covariance components; this cap is prespecified, not optimized on held-out labels.\n\nThe extension re-fits shared reference components once per task and caches them. It does not assume undocumented historical model caches have compatible splits or parameters. Inspect round2_control_reconciliation.csv before replacing old manuscript numbers. Model-training randomness, prediction accumulation, and the explicitly recorded validation-halving rule may cause differences; new comparisons must remain internally matched.\n\nQuantization checks freeze the float32-trained head, prototypes, and normalizers. The same integer identities are retained while both calibration and test scalar fields are encoded in binary16. The diagnostic numerical operator is evaluated in float64. Analytic sigmoid intervals, normalized OVR probability intervals, possible predicted-class prototype sets, and monotone fusion bound score changes. Calibration order-statistic movement is bounded by the maximum calibration-score perturbation. Decision changes must lie within the exported score-margin band. Numerical validation has a stated 1e-8 tolerance and is not a directed-rounding machine proof. This comparison is distinct from separately refitting the 16Q head, and does not establish a quantization theorem for different message content.\n\nTests and bootstrap/sign-flip analyses do not create independent deployment datasets. The primary multiplicity family has six replacements times three FPR levels, separately for each denominator and statistical test. Four reference comparisons form separate exploratory families. No equivalence conclusion follows from nonsignificance.\n'''
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
