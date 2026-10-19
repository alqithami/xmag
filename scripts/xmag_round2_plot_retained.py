#!/usr/bin/env python3
"""Plot retained 16Q per-family means and genuine across-seed dispersion."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metrics-csv',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    df=pd.read_csv(args.metrics_csv)
    x=df[(df.method=='X-MAG-COS-16Q')&(df.score=='composite')]
    if len(x)!=40 or x.duplicated(['seed','held_out_attack']).any():
        raise SystemExit('Expected exactly 40 distinct retained 16Q seed-family rows')
    if not x.groupby('held_out_attack').seed.nunique().eq(5).all():
        raise SystemExit('Expected five seeds for each family')
    a=x.groupby('held_out_attack')[['unknown_auroc','unknown_recall_conformal05']].agg(['mean','std'])
    fig,ax=plt.subplots(figsize=(8.2,4.3));pos=np.arange(len(a));w=.37
    ax.bar(pos-w/2,a[('unknown_auroc','mean')],w,yerr=a[('unknown_auroc','std')],capsize=3,label='AUROC')
    ax.bar(pos+w/2,a[('unknown_recall_conformal05','mean')],w,yerr=a[('unknown_recall_conformal05','std')],capsize=3,label='Recall: nominal 5% rank rule')
    ax.set_xticks(pos,a.index,rotation=32,ha='right')
    ax.set_ylim(-.02,1.08);ax.set_ylabel('Retained 16Q mean ± seed standard deviation')
    ax.set_yticks(np.linspace(0,1,6));ax.legend(loc='lower left',fontsize=8)
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,bbox_inches='tight');fig.savefig(p.with_suffix('.png'),dpi=240,bbox_inches='tight')
    a.to_csv(p.with_name(p.stem+'_data.csv'));plt.close(fig)
    print(p)
if __name__=='__main__':main()
