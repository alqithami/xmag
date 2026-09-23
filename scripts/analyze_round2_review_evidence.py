#!/usr/bin/env python3
"""Reanalyse author-supplied Round-2 evidence; no training or test-label tuning.

Inputs are lossless selected-column projections of the author's text export.
They are not substitutes for the full audit archive or executed training code.
"""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import rankdata, ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/mdpi_r2/review_evidence/encoding_pairs_alpha005.csv'
OUT = ROOT / 'results/mdpi_r2/review_evidence/derived'
METRICS = ['known_macro_f1', 'unknown_auroc', 'unknown_recall', 'known_false_rejection_rate']
SEEDS = {7, 21, 42, 84, 123}
FAMILIES = {'HTTPFlood','ICMPFlood','SYNFlood','SYNScan','SlowrateDoS','TCPConnectScan','UDPFlood','UDPScan'}
EXPECTED_MEANS = {
 'known_macro_f1_12B': 0.9988055471163794,
 'known_macro_f1_16Q': 0.9987938275875425,
 'unknown_auroc_12B': 0.8998768027171952,
 'unknown_auroc_16Q': 0.9054290486482147,
 'unknown_recall_12B': 0.7046460408989582,
 'unknown_recall_16Q': 0.7241284986731309,
 'known_false_rejection_rate_12B': 0.047416648593506215,
 'known_false_rejection_rate_16Q': 0.049868453898031406,
}

def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    out = np.empty(len(values))
    running = 0.0
    for k, idx in enumerate(order):
        running = max(running, (len(values)-k)*values[idx])
        out[idx] = min(1.0, running)
    return out.tolist()

def bootstrap_mean_ci(d, seed):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(50000,len(d)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def peer_summary():
    path = INPUT.with_name('peer_ideal_loss1_alpha005.csv')
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    keys = [(int(r['seed']),r['held_out_attack'],r['method'],r['channel']) for r in rows]
    expected=set(itertools.product(SEEDS, {'UDPFlood','SlowrateDoS'}, {'peer_context','owner_entropy_control'}, {'ideal','loss_only_1pct'}))
    assert len(keys)==40 and set(keys)==expected
    results=[]
    for family, method, channel in itertools.product(sorted({'UDPFlood','SlowrateDoS'}),['owner_entropy_control','peer_context'],['ideal','loss_only_1pct']):
        selected=[r for r in rows if (r['held_out_attack'],r['method'],r['channel'])==(family,method,channel)]
        rec={'held_out_attack':family,'method':method,'channel':channel,'n_seeds':5}
        for metric in ['unknown_auroc','unknown_recall_conditional','unknown_recall_end_to_end']:
            x=np.array([float(r[metric]) for r in selected])
            rec[metric+'_mean']=float(x.mean())
            rec[metric+'_std']=float(x.std(ddof=1))
        results.append(rec)
    write_csv(OUT/'peer_ideal_loss1_summary.csv',results)
    diagnostics=[]
    for family in ['UDPFlood','SlowrateDoS']:
        x={ (int(r['seed']),r['method'],r['channel']): float(r['unknown_auroc']) for r in rows if r['held_out_attack']==family }
        change=np.array([x[s,'peer_context','loss_only_1pct']-x[s,'peer_context','ideal'] for s in sorted(SEEDS)])
        peer_minus_owner=np.array([x[s,'peer_context','ideal']-x[s,'owner_entropy_control','ideal'] for s in sorted(SEEDS)])
        diagnostics.append({'family':family,'max_absolute_loss1_auroc_change':float(np.abs(change).max()),'mean_peer_minus_owner_ideal_auroc':float(peer_minus_owner.mean())})
    return {'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'summaries':results,'diagnostics':diagnostics}

def main():
    with INPUT.open(newline='', encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    keys=[(int(r['seed']), r['held_out_attack']) for r in rows]
    assert len(rows)==40 and len(set(keys))==40
    assert set(keys)==set(itertools.product(SEEDS,FAMILIES))
    cols={k:np.array([float(r[k]) for r in rows]) for k in EXPECTED_MEANS}
    for k, expected in EXPECTED_MEANS.items():
        np.testing.assert_allclose(cols[k].mean(), expected, rtol=0, atol=1e-12, err_msg=k)
    OUT.mkdir(parents=True, exist_ok=True)
    stats=[]
    for i, metric in enumerate(METRICS):
        a,b=cols[metric+'_16Q'],cols[metric+'_12B']
        d=a-b
        dr=np.round(d,12)
        nonzero=dr[dr!=0]
        w=wilcoxon(dr, zero_method='wilcox', alternative='two-sided', method='approx') if len(nonzero) else None
        ranks=rankdata(np.abs(nonzero),method='average')
        rbc=float(np.sum(ranks*np.sign(nonzero))/ranks.sum()) if len(nonzero) else 0.0
        fammeans=np.array([d[[r['held_out_attack']==h for r in rows]].mean() for h in sorted(FAMILIES)])
        signs=np.array(list(itertools.product([-1,1],repeat=8)))
        exact=float(np.mean(np.abs((signs*fammeans).mean(axis=1))>=abs(fammeans.mean())-1e-15))
        ci=bootstrap_mean_ci(d,271828+i)
        fci=bootstrap_mean_ci(fammeans,314159+i)
        tt=ttest_1samp(d,0.0)
        stats.append(dict(metric=metric,n_pairs=40,n_family_blocks=8,mean_12B=float(b.mean()),mean_16Q=float(a.mean()),mean_difference=float(d.mean()),sd_difference=float(d.std(ddof=1)),bootstrap_low=ci[0],bootstrap_high=ci[1],family_block_bootstrap_low=fci[0],family_block_bootstrap_high=fci[1],paired_t=float(tt.statistic),paired_t_p=float(tt.pvalue),wilcoxon_statistic=float(w.statistic) if w else 0.0,wilcoxon_p=float(w.pvalue) if w else 1.0,rank_biserial=rbc,family_signflip_p=exact,positive=int(np.sum(dr>0)),negative=int(np.sum(dr<0)),zero=int(np.sum(dr==0))))
    for p, adj in [('paired_t_p','paired_t_holm'),('wilcoxon_p','wilcoxon_holm'),('family_signflip_p','family_signflip_holm')]:
        for row,value in zip(stats,holm([r[p] for r in stats])): row[adj]=value
    write_csv(OUT/'paired_16Q_minus_12B.csv',stats)
    family_rows=[]
    for h in sorted(FAMILIES):
        mask=np.array([r['held_out_attack']==h for r in rows])
        row={'held_out_attack':h,'n_seeds':int(mask.sum())}
        for metric in METRICS:
            for layout in ['12B','16Q']:
                x=cols[metric+'_'+layout][mask]
                row[metric+'_'+layout+'_mean']=float(x.mean())
                row[metric+'_'+layout+'_std']=float(x.std(ddof=1))
        family_rows.append(row)
    write_csv(OUT/'per_family_encoding_summary.csv',family_rows)
    report={'status':'validated_selected_evidence','input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'round2_dataset_sha256':'7c238e2d5dabbc1afcd01c50db91372d6f6808f7572bb92baeb2df03a18af90c','scope':'40 paired encoding trials and 40 selected peer rows at nominal alpha 0.05; no new training; selected columns only','test_family':'four displayed metrics, Holm separately by test type','caution':'Seed-holdout pairs reuse one dataset. Trial-wise tests are descriptive; family-block sensitivity is also reported. Nonsignificance is not equivalence.','statistics':stats,'per_family':family_rows,'peer':peer_summary()}
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('ROUND2_ANALYSIS_BEGIN')
    print(json.dumps(report,indent=2))
    print('ROUND2_ANALYSIS_END')

if __name__=='__main__': main()
