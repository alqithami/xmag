#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mdpi_revision_common import *
from mdpi_revision_trial import fit_message_variant

def evaluate_variant(data, mainv, Mv, Mk, Mu, scenario, bytes_per_flow=24):
    Mtr=mainv["Mtr"]
    coord=fit_coordinator(Mtr,data.y_train)
    pv=align_proba(coord,Mv,data.classes);pk=align_proba(coord,Mk,data.classes);pu=align_proba(coord,Mu,data.classes)
    uv,uk,uu=uncertainty_score(pv),uncertainty_score(pk),uncertainty_score(pu)
    av,ak,au=mainv["a"]
    rv,proto,rs=prototype_residual(Mtr,data.y_train,Mv,pv,data.classes)
    rk=residual_with_proto(Mk,pk,data.classes,proto,rs);ru=residual_with_proto(Mu,pu,data.classes,proto,rs)
    sv=composite_scores(uv,av,rv,uv,av,rv);sk=composite_scores(uv,av,rv,uk,ak,rk);su=composite_scores(uv,av,rv,uu,au,ru)
    met=metrics_for_score(data.y_known,pk,data.classes,sv,sk,su)
    return {"scenario":scenario,"bytes_per_flow":bytes_per_flow,**met}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True);ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    data=prepare_data(args.config,args.seed)
    cls=train_local_classifiers(data,args.seed);ifs=train_local_anomaly(data,args.seed)
    mainv=fit_message_variant(data,cls,ifs,m=1,k=1,include_class=True,include_proxy=True,include_anomaly=True)
    rows=[]
    scenarios=[("ideal",0.0,0),("edge",0.01,3),("congested",0.05,20),("severe",0.10,50)]
    for name,loss,jitter in scenarios:
        ctr,dtr=peer_context(mainv["Mtr"],data.a_train,data.idx_train,loss,jitter,args.seed)
        cv,dv=peer_context(mainv["Mv"],data.a_val,data.idx_val,loss,jitter,args.seed+1)
        ck,dk=peer_context(mainv["Mk"],data.a_known,data.idx_known,loss,jitter,args.seed+2)
        cu,du=peer_context(mainv["Mu"],data.a_unknown,data.idx_unknown,loss,jitter,args.seed+3)
        Mtr=np.c_[mainv["Mtr"]*dtr[:,None],ctr]
        Mv=np.c_[mainv["Mv"]*dv[:,None],cv]
        Mk=np.c_[mainv["Mk"]*dk[:,None],ck]
        Mu=np.c_[mainv["Mu"]*du[:,None],cu]
        coord=fit_coordinator(Mtr,data.y_train)
        pv=align_proba(coord,Mv,data.classes);pk=align_proba(coord,Mk,data.classes);pu=align_proba(coord,Mu,data.classes)
        score_v=entropy_score(pv);score_k=entropy_score(pk);score_u=entropy_score(pu)
        met=metrics_for_score(data.y_known,pk,data.classes,score_v,score_k,score_u)
        rows.append({"experiment":"network","scenario":name,"loss_rate":loss,"jitter_steps":jitter,
                     "known_delivery":float(dk.mean()),"unknown_delivery":float(du.mean()),**met})
    K=len(data.classes);D=data.transformed_feature_count
    for frac in [0.10,0.20,0.30]:
        for attack in ["class_suppression","proxy_replacement","anomaly_suppression","replay","combined"]:
            Mk,bad=targeted_attack(mainv["Mk"],data.a_known,data.classes,data.benign_class,frac,attack,
                                   args.seed+10,K,D)
            Mu,_=targeted_attack(mainv["Mu"],data.a_unknown,data.classes,data.benign_class,frac,attack,
                                 args.seed+10,K,D)
            r=evaluate_variant(data,mainv,mainv["Mv"],Mk,Mu,f"{attack}_{frac}")
            r.update({"experiment":"active_attack","attack":attack,"compromised_fraction":frac,
                      "compromised_agents":";".join(map(str,bad.tolist()))})
            rows.append(r)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).to_string(index=False));print(f"\nSaved {out}")
if __name__=="__main__":main()
