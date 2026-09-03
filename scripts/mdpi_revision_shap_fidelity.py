#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mdpi_revision_common import *

def extract_shap(explainer, X, model, classes):
    values=explainer.shap_values(X)
    pred_local=np.argmax(model.predict_proba(X),axis=1)
    if isinstance(values,list):
        return np.stack([values[c][i] for i,c in enumerate(pred_local)])
    arr=np.asarray(values)
    if arr.ndim==3 and arr.shape[-1]==len(model.classes_):
        return np.stack([arr[i,:,c] for i,c in enumerate(pred_local)])
    if arr.ndim==2: return arr
    raise ValueError(f"Unsupported SHAP shape {arr.shape}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True);ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",required=True);ap.add_argument("--sample",type=int,default=2000)
    args=ap.parse_args()
    try: import shap
    except ImportError: raise SystemExit("Install SHAP first: python -m pip install shap")
    data=prepare_data(args.config,args.seed)
    models=train_local_classifiers(data,args.seed)
    rng=np.random.default_rng(args.seed)
    pool_X=np.r_[data.X_known,data.X_unknown]
    pool_a=np.r_[data.a_known,data.a_unknown]
    ids=rng.choice(len(pool_X),size=min(args.sample,len(pool_X)),replace=False)
    X=pool_X[ids];owners=pool_a[ids]
    all_proxy=[];all_shap=[];t_proxy=0.;t_shap=0.
    for aid,model in enumerate(models):
        mask=owners==aid
        if not np.any(mask) or not hasattr(model,"feature_importances_"): continue
        Xi=X[mask]
        t=time.perf_counter()
        proxy=Xi.astype(float)*model_importance(model,Xi.shape[1])[None,:]
        t_proxy+=time.perf_counter()-t
        t=time.perf_counter()
        explainer=shap.TreeExplainer(model)
        sv=extract_shap(explainer,Xi,model,data.classes)
        t_shap+=time.perf_counter()-t
        all_proxy.append(proxy);all_shap.append(sv)
    proxy=np.concatenate(all_proxy);sv=np.concatenate(all_shap)
    abs_p=np.abs(proxy);abs_s=np.abs(sv)
    top1=np.mean(np.argmax(abs_p,axis=1)==np.argmax(abs_s,axis=1))
    top3=[];rho=[]
    for p,s in zip(abs_p,abs_s):
        P=set(np.argpartition(p,-min(3,len(p)))[-3:])
        S=set(np.argpartition(s,-min(3,len(s)))[-3:])
        top3.append(len(P&S)/len(P|S))
        r=spearmanr(p,s).statistic
        if np.isfinite(r): rho.append(r)
    result=pd.DataFrame([{
        "holdout":data.unknown_attack,"seed":args.seed,"n_samples":len(proxy),
        "top1_agreement":top1,"top3_jaccard_mean":np.mean(top3),
        "absolute_rank_spearman_mean":np.mean(rho),
        "proxy_seconds_total":t_proxy,"treeshap_seconds_total":t_shap,
        "speedup_treeshap_over_proxy":t_shap/max(t_proxy,1e-12),
    }])
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);result.to_csv(out,index=False)
    print(result.to_string(index=False));print(f"\nSaved {out}")
if __name__=="__main__":main()
