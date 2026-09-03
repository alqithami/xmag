#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def holm(pvals):
    m=len(pvals);order=np.argsort(pvals);adj=np.empty(m,float);running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*pvals[idx];running=max(running,val);adj[idx]=min(1.0,running)
    return adj

def rank_biserial(x,y):
    d=np.asarray(x)-np.asarray(y);d=d[d!=0]
    if not len(d): return 0.0
    ranks=np.arange(1,len(d)+1,dtype=float)
    order=np.argsort(np.abs(d));rr=np.empty_like(ranks);rr[order]=ranks
    pos=rr[d>0].sum();neg=rr[d<0].sum()
    return float((pos-neg)/(pos+neg)) if pos+neg else 0.0

def bootstrap_ci(d,seed=42,B=10000):
    rng=np.random.default_rng(seed);d=np.asarray(d,float)
    means=np.mean(rng.choice(d,size=(B,len(d)),replace=True),axis=1)
    return np.quantile(means,[0.025,0.975])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs-root",default="runs/mdpi_r1")
    ap.add_argument("--out",default="results/mdpi_r1/paired_statistics.csv")
    args=ap.parse_args()
    frames=[]
    for p in Path(args.runs_root).glob("**/metrics.csv"):
        df=pd.read_csv(p);frames.append(df)
    if not frames: raise SystemExit("No metrics.csv files found.")
    df=pd.concat(frames,ignore_index=True)
    key=["seed","held_out_attack"]
    target=df[(df.method=="X-MAG-COS-24B")&(df.score=="composite")].set_index(key)
    comparators=["X-MAG-DH-24B","Distributed-logit-average","Central-RF-full","FedAvg-linear","Central-ExtraTrees-full"]
    metrics=["known_macro_f1","unknown_auroc","unknown_recall_conformal05","tpr_at_test_fpr05"]
    rows=[]
    for comp in comparators:
        c=df[df.method==comp].copy()
        if comp=="Central-RF-full": c=c[c.score=="entropy"]
        if comp=="X-MAG-DH-24B": c=c[c.score=="coordinator_uncertainty"]
        c=c.drop_duplicates(key).set_index(key)
        common=target.index.intersection(c.index)
        if len(common)<3: continue
        for metric in metrics:
            x=target.loc[common,metric].astype(float).to_numpy()
            y=c.loc[common,metric].astype(float).to_numpy()
            d=x-y
            try: w=wilcoxon(x,y,zero_method="wilcox",alternative="two-sided")
            except Exception: w=type("R",(),{"statistic":np.nan,"pvalue":1.0})()
            t=ttest_rel(x,y,nan_policy="omit")
            lo,hi=bootstrap_ci(d)
            rows.append({
                "comparison":f"X-MAG-COS-24B vs {comp}","metric":metric,"n_pairs":len(common),
                "xmag_mean":np.mean(x),"baseline_mean":np.mean(y),"mean_difference":np.mean(d),
                "bootstrap_ci_low":lo,"bootstrap_ci_high":hi,
                "paired_t_stat":t.statistic,"paired_t_p":t.pvalue,
                "wilcoxon_stat":w.statistic,"wilcoxon_p":w.pvalue,
                "rank_biserial":rank_biserial(x,y),
            })
    out=pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_p_holm"]=holm(out["wilcoxon_p"].fillna(1).to_numpy())
        out["paired_t_p_holm"]=holm(out["paired_t_p"].fillna(1).to_numpy())
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);out.to_csv(p,index=False)
    print(out.to_string(index=False));print(f"\nSaved {p}")
if __name__=="__main__":main()
