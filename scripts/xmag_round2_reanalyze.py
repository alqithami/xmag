#!/usr/bin/env python3
"""Round-2 reanalysis of retained trial metrics; never trains or changes scores."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

METRICS = ['known_macro_f1', 'unknown_auroc', 'unknown_recall_conformal05']
KEYS = ['held_out_attack', 'seed']

def holm(values):
    p = np.asarray(values, float)
    order = np.argsort(p, kind='stable')
    ans = np.empty(len(p)); bound = 0.0
    for i, j in enumerate(order):
        bound = max(bound, (len(p)-i)*p[j]); ans[j] = min(1.0, bound)
    return ans

def boot(values, seed=42, draws=50000):
    x = np.asarray(values, float); rng = np.random.default_rng(seed)
    return np.quantile(x[rng.integers(0, len(x), (draws, len(x)))].mean(axis=1), [.025, .975])

def paired(a, b, metric):
    assert a.index.equals(b.index), 'Unmatched trial identities'
    d = (a[metric]-b[metric]).astype(float)
    if not np.isfinite(d).all(): raise ValueError('Nonfinite paired difference')
    nz = d.to_numpy()[d.to_numpy()!=0]
    if not len(nz):
        wstat, wp, ts, tp, rb = 0., 1., 0., 1., 0.
    else:
        w = stats.wilcoxon(d.to_numpy(), zero_method='wilcox', alternative='two-sided', method='auto')
        t = stats.ttest_1samp(d, 0.)
        ranks = stats.rankdata(np.abs(nz), method='average')
        rb = (ranks[nz>0].sum()-ranks[nz<0].sum())/ranks.sum()
        wstat, wp, ts, tp = w.statistic, w.pvalue, t.statistic, t.pvalue
    lo, hi = boot(d)
    fam = d.groupby(level='held_out_attack').mean()
    flo, fhi = boot(fam)
    signs = np.array(list(itertools.product([-1,1], repeat=len(fam))))
    fp = np.mean(np.abs(signs@fam.to_numpy()/len(fam)) >= abs(fam.mean())-1e-15)
    return {'metric':metric,'n_pairs':len(d),'n_families':len(fam),
            'mean_difference':d.mean(),'median_difference':d.median(),
            'ci95_trial_low':lo,'ci95_trial_high':hi,
            'ci95_family_block_low':flo,'ci95_family_block_high':fhi,
            'paired_t_stat':ts,'paired_t_p':tp,'wilcoxon_stat':wstat,'wilcoxon_p':wp,
            'rank_biserial_tie_corrected':rb,'family_signflip_p':fp,
            'positive_pairs':int((d>0).sum()),'negative_pairs':int((d<0).sum()),'zero_pairs':int((d==0).sum())}

def get(df, method, score):
    x = df[(df.method==method)&(df.score==score)].copy()
    if x.duplicated(KEYS).any(): raise ValueError('Duplicate paired keys')
    return x.set_index(KEYS).sort_index()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--primary-metrics',required=True)
    ap.add_argument('--secondary-metrics')
    ap.add_argument('--outdir',required=True)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    src=Path(args.primary_metrics); df=pd.read_csv(src)
    a=get(df,'X-MAG-COS-16Q','composite'); b=get(df,'Class+anomaly-12B','composite')
    if len(a)!=40 or set(a.index.get_level_values('seed'))!={7,21,42,84,123}:
        raise ValueError('Expected 40 matched primary seed-family trials')
    pairs=pd.DataFrame([paired(a,b,m) for m in METRICS])
    for typ in ['wilcoxon_p','paired_t_p','family_signflip_p']:
        pairs[typ+'_holm3']=holm(pairs[typ])
    pairs.to_csv(out/'paired_16q_minus_12b.csv',index=False)
    (a[METRICS]-b[METRICS]).reset_index().to_csv(out/'paired_16q_minus_12b_by_trial.csv',index=False)
    (a[METRICS]-b[METRICS]).groupby(level=0).mean().to_csv(out/'paired_16q_minus_12b_by_family.csv')
    q24=get(df,'X-MAG-COS-24B','composite')
    pd.DataFrame([paired(a,q24,m) for m in METRICS]).to_csv(out/'paired_16q_minus_24b.csv',index=False)
    comparators=[('X-MAG-DH-24B','coordinator_uncertainty'),('Distributed-logit-average','entropy+anomaly'),
                 ('Central-RF-full','entropy'),('FedAvg-linear','entropy'),('Central-ExtraTrees-full','entropy')]
    table=[]
    for method, score in comparators:
        other=get(df,method,score)
        for metric in METRICS+['tpr_at_test_fpr05']:
            r=paired(q24,other,metric);r.update(comparator=method,
                interpretation='legacy_all_slot_comparator_under_audit' if method=='Distributed-logit-average' else 'submitted_protocol_reanalysis')
            table.append(r)
    tests=pd.DataFrame(table)
    for typ in ['wilcoxon_p','paired_t_p']:
        tests[typ+'_holm20']=holm(tests[typ])
    tests.to_csv(out/'complete_paired_tests_recomputed.csv',index=False)
    crf=get(df,'Central-RF-full','entropy')
    (q24[METRICS]-crf[METRICS]).groupby(level=0).mean().to_csv(out/'central_rf_differences_by_family.csv')
    la=get(df,'X-MAG-COS-24B','local_anomaly')
    cols=['unknown_auroc','unknown_recall_q95','unknown_recall_conformal05','known_fpr_q95','known_fpr_conformal05']
    la[cols].reset_index().to_csv(out/'anomaly_decision_rules_by_trial.csv',index=False)
    la[cols].agg(['mean','std','min','max']).T.to_csv(out/'anomaly_decision_rule_summary.csv')
    agg_cols=METRICS+['known_fpr_conformal05']
    for name, frame in [('5gnidd',df)]+([('ciciot',pd.read_csv(args.secondary_metrics))] if args.secondary_metrics else []):
        g=frame.groupby(['method','score','message_bytes_per_flow'])[agg_cols].agg(['mean','std','min','max'])
        g.columns=['_'.join(x) for x in g.columns]
        g.to_csv(out/f'{name}_means_and_dispersion.csv')
    h=a.xs('UDPFlood',level='held_out_attack').unknown_auroc
    t=stats.ttest_1samp(h,.5,alternative='less')
    ci=stats.t.interval(.95,len(h)-1,loc=h.mean(),scale=stats.sem(h))
    pd.DataFrame([dict(n_seeds=len(h),mean_auroc=h.mean(),sd_auroc=h.std(),
        all_seeds_below_chance=bool((h<.5).all()),minimum=h.min(),maximum=h.max(),
        t_stat=t.statistic,one_sided_p=t.pvalue,seed_variation_ci_low=ci[0],seed_variation_ci_high=ci[1],
        family_rows=457340,total_rows=1215890,dataset_fraction=457340/1215890,
        qualification='Conditional seed variability on one fixed dataset; not independent deployment replications'
    )]).to_csv(out/'udpflood_below_chance.csv',index=False)
    manifest={'purpose':'Reanalysis of retained observations, not a rerun',
              'primary_rows':len(df),'primary_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
              'bootstrap_draws':50000,'resampling_seed':42,
              'software':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':__import__('scipy').__version__},
              'warnings':['Nonsignificance is not equivalence.',
                          '40 repeated seed-family trials are not 40 independent datasets.',
                          'The legacy all-slot comparator is retained for audit only, not a superiority claim.']}
    if args.secondary_metrics: manifest['secondary_sha256']=hashlib.sha256(Path(args.secondary_metrics).read_bytes()).hexdigest()
    (out/'reanalysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pairs.to_string(index=False)); print(f'Saved completed statistical analyses: {out}')
if __name__=='__main__':main()
