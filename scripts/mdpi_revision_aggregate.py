#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs-root",default="runs/mdpi_r1")
    ap.add_argument("--outdir",default="results/mdpi_r1")
    args=ap.parse_args()
    outdir=Path(args.outdir);(outdir/"figures").mkdir(parents=True,exist_ok=True)
    frames=[]
    for p in Path(args.runs_root).glob("**/metrics.csv"):
        frames.append(pd.read_csv(p))
    if not frames: raise SystemExit("No metrics.csv files found.")
    allm=pd.concat(frames,ignore_index=True)
    allm.to_csv(outdir/"all_metrics.csv",index=False)

    summary=(allm.groupby(["method","score","message_bytes_per_flow"],dropna=False)
             .agg(n=("unknown_auroc","size"),
                  known_f1_mean=("known_macro_f1","mean"),known_f1_std=("known_macro_f1","std"),
                  auroc_mean=("unknown_auroc","mean"),auroc_std=("unknown_auroc","std"),
                  auroc_min=("unknown_auroc","min"),
                  recall_mean=("unknown_recall_conformal05","mean"),
                  recall_std=("unknown_recall_conformal05","std"),
                  recall_min=("unknown_recall_conformal05","min"),
                  fpr_mean=("known_fpr_conformal05","mean"),
                  fpr_max=("known_fpr_conformal05","max"),
                  latency_us_mean=("per_flow_predict_us","mean"),
                  model_size_bytes_mean=("model_size_bytes","mean"))
             .reset_index())
    summary.to_csv(outdir/"protocol_matched_summary.csv",index=False)

    pareto=summary[summary.method.isin([
        "Class+anomaly-12B","Class+proxy-18B","X-MAG-COS-16Q","X-MAG-COS-20B",
        "X-MAG-COS-24B","X-MAG-COS-30B","Distributed-logit-average","Central-RF-full"
    ])].copy()
    pareto.to_csv(outdir/"message_pareto.csv",index=False)

    fig,ax=plt.subplots(figsize=(7.2,4.8))
    finite=pareto[np.isfinite(pareto.message_bytes_per_flow)]
    ax.scatter(finite.message_bytes_per_flow,finite.auroc_mean,s=60)
    for _,r in finite.iterrows():
        ax.annotate(r.method,(r.message_bytes_per_flow,r.auroc_mean),xytext=(4,4),
                    textcoords="offset points",fontsize=7)
    ax.set_xlabel("Evidence or feature bytes per flow")
    ax.set_ylabel("Mean unknown AUROC")
    ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(outdir/"figures"/"pareto_bytes_auroc.pdf")
    fig.savefig(outdir/"figures"/"pareto_bytes_auroc.png",dpi=250);plt.close(fig)

    x=allm[(allm.method=="X-MAG-COS-24B")&(allm.score=="composite")]
    fam=x.groupby("held_out_attack").agg(auroc=("unknown_auroc","mean"),
                                         recall=("unknown_recall_conformal05","mean")).reset_index()
    fig,ax=plt.subplots(figsize=(8,4.8))
    pos=np.arange(len(fam));w=.38
    ax.bar(pos-w/2,fam.auroc,width=w,label="AUROC")
    ax.bar(pos+w/2,fam.recall,width=w,label="Recall at conformal 5%")
    ax.set_xticks(pos,fam.held_out_attack,rotation=35,ha="right")
    ax.set_ylim(0,1.05);ax.legend();ax.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(outdir/"figures"/"per_family_open_set.pdf")
    fig.savefig(outdir/"figures"/"per_family_open_set.png",dpi=250);plt.close(fig)
    print(summary.to_string(index=False))
    print(f"\nSaved tables and figures under {outdir}")
if __name__=="__main__":main()
