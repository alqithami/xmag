#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdpi_revision_common import composite_scores, metrics_for_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs/mdpi_r1")
    ap.add_argument("--out", default="results/mdpi_r1/hyperparameter_sensitivity.csv")
    ap.add_argument("--grid", nargs="+", type=float, default=[0.0,0.25,0.5,0.75,1.0])
    args = ap.parse_args()
    rows=[]
    for npz_path in sorted(Path(args.runs_root).glob("**/score_components.npz")):
        z=np.load(npz_path, allow_pickle=True)
        classes=[str(x) for x in z["classes"]]
        for beta in args.grid:
            for lam in args.grid:
                for gamma in args.grid:
                    sv=composite_scores(z["val_u"],z["val_a"],z["val_r"],z["val_u"],z["val_a"],z["val_r"],beta,lam,gamma)
                    sk=composite_scores(z["val_u"],z["val_a"],z["val_r"],z["known_u"],z["known_a"],z["known_r"],beta,lam,gamma)
                    su=composite_scores(z["val_u"],z["val_a"],z["val_r"],z["unknown_u"],z["unknown_a"],z["unknown_r"],beta,lam,gamma)
                    yk=z["y_known"].astype(str); pk=z["pred_known"].astype(str)
                    from sklearn.metrics import f1_score
                    ybin=np.r_[np.zeros(len(sk),int),np.ones(len(su),int)]
                    score=np.r_[sk,su]
                    from sklearn.metrics import roc_auc_score
                    thr=float(np.quantile(sv,0.95,method="higher"))
                    rows.append({
                        "run":npz_path.parent.name,"beta":beta,"lambda":lam,"gamma":gamma,
                        "known_macro_f1":float(f1_score(yk,pk,average="macro",zero_division=0)),
                        "unknown_auroc":float(roc_auc_score(ybin,score)),
                        "unknown_recall_q95":float(np.mean(su>=thr)),
                        "known_fpr_q95":float(np.mean(sk>=thr)),
                    })
    if not rows: raise SystemExit("No score_components.npz files found.")
    out=pd.DataFrame(rows)
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);out.to_csv(p,index=False)
    summary=(out.groupby(["beta","lambda","gamma"])
             .agg(mean_auroc=("unknown_auroc","mean"),min_auroc=("unknown_auroc","min"),
                  mean_recall=("unknown_recall_q95","mean"),min_recall=("unknown_recall_q95","min"),
                  mean_fpr=("known_fpr_q95","mean"),max_fpr=("known_fpr_q95","max"))
             .reset_index()
             .sort_values(["min_auroc","min_recall"],ascending=False))
    sp=p.with_name(p.stem+"_summary.csv");summary.to_csv(sp,index=False)
    print(summary.head(30).to_string(index=False))
    print(f"\nSaved {p}\nSaved {sp}")
if __name__=="__main__": main()
