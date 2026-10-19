#!/usr/bin/env python3
"""Targeted real-data audits for mathematics-4541032, round 2.

Run from the existing xmag workspace. Old scripts, runs and results are untouched.
No synthetic data substitute is used. Completed matching trials are resumed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
sys.path.insert(0,str(Path(__file__).resolve().parent))
import mdpi_revision_common as ref
from xmag_round2_core import (active_average, align_probabilities, entropy, scaler, scale,
    composite, pvalues, operating_metrics, calibration_parts, fit_head, prototypes,
    residual, packet_message, event_context, classification_f1)

SEEDS=[7,21,42,84,123]
HOLDS=['HTTPFlood','ICMPFlood','SYNFlood','SYNScan','SlowrateDoS','TCPConnectScan','UDPFlood','UDPScan']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf8')


def atomic_csv(df,path):
    path=Path(path);tmp=path.with_suffix('.tmp.csv');df.to_csv(tmp,index=False);tmp.replace(path)


def slot_table(owners,labels,nominal,holdout,seed,split):
    owners=np.asarray(owners,int);labels=np.asarray(labels,str);n=len(owners);out=[]
    for slot in range(nominal):
        m=owners==slot
        out.append(dict(holdout=holdout,seed=seed,split=split,slot=slot,flow_count=int(m.sum()),
                        share=float(m.mean()) if n else 0.,distinct_labels=int(len(np.unique(labels[m])))))
    return out


def prepare_round2_data(config_path,seed):
    """Original row splits with a training-only feature screen (no test inspection)."""
    cfg=ref.load_yaml(config_path)
    df=pd.read_csv(cfg['dataset']['path'],low_memory=False)
    df=df.loc[:,~df.columns.astype(str).str.startswith('Unnamed:')].copy()
    label=ref.infer_label(df.columns,cfg['dataset'].get('label_column'))
    y=df[label].astype(str).str.strip().to_numpy(); h=str(cfg['dataset']['unknown_attack'])
    if h not in set(y):raise ValueError(f'Unknown holdout label {h!r}')
    labels=[str(v) for v in cfg['dataset'].get('benign_labels',['Benign'])]
    benign=next((b for b in labels if b in set(y)),None)
    if benign is None:raise ValueError('No declared benign label is present')
    agent=cfg['agents'].get('agent_column')
    if not agent:agent=next((c for c in ref.AGENT_CANDIDATES if c in df.columns),None)
    idx=np.arange(len(df));known=idx[y!=h];unknown=idx[y==h]
    tv,kt=train_test_split(known,test_size=float(cfg['experiment']['test_size']),random_state=seed,stratify=y[known])
    tr,va=train_test_split(tv,test_size=float(cfg['experiment']['validation_size']),random_state=seed,stratify=y[tv])
    drops=set(ref.leakage_columns(df,cfg,label))
    if agent:drops.add(agent)
    X=df.drop(columns=list(drops),errors='ignore').replace([np.inf,-np.inf],np.nan)
    count=X.iloc[tr].nunique(dropna=False)
    X=X.loc[:,count[count>1].index]
    pre=ref.make_preprocessor(X.iloc[tr]); Xt=np.asarray(pre.fit_transform(X.iloc[tr]),np.float32)
    Xv,Xk,Xu=[np.asarray(pre.transform(X.iloc[i]),np.float32) for i in (va,kt,unknown)]
    names=list(map(str,pre.get_feature_names_out()))
    N=int(cfg['agents']['n_agents'])
    owners=[];source=None
    for i in (tr,va,kt,unknown):
        o,source=ref.stable_agents(df.iloc[i][agent] if agent else None,i,N);owners.append(o)
    return ref.PreparedData(Xt,y[tr],owners[0],tr,Xv,y[va],owners[1],va,
        Xk,y[kt],owners[2],kt,Xu,y[unknown],owners[3],unknown,
        sorted(np.unique(y[tr]).tolist()),names,label,source,h,benign,int(X.shape[1]),int(Xt.shape[1]),pre)


def metadata_audit(csv,configs,seeds,out):
    c0=yaml.safe_load(configs[0].read_text());cols=pd.read_csv(csv,nrows=0).columns
    label=c0['dataset']['label_column'];agent=c0['agents'].get('agent_column')
    if not agent: agent=next((x for x in ref.AGENT_CANDIDATES if x in cols),None)
    nominal=int(c0['agents']['n_agents']);d=pd.read_csv(csv,usecols=[label]+([agent] if agent else []),low_memory=False)
    y=d[label].astype(str).str.strip().to_numpy();ids=np.arange(len(d))
    own,_=ref.stable_agents(d[agent] if agent else None,ids,nominal)
    rows=slot_table(own,y,nominal,'ALL',-1,'entire_dataset')
    classrows=[];compromise=[]
    for cfgpath in configs:
        cfg=yaml.safe_load(cfgpath.read_text());h=str(cfg['dataset']['unknown_attack']);known=ids[y!=h];unk=ids[y==h]
        for seed in seeds:
            tv,kt=train_test_split(known,test_size=cfg['experiment']['test_size'],random_state=seed,stratify=y[known])
            tr,va=train_test_split(tv,test_size=cfg['experiment']['validation_size'],random_state=seed,stratify=y[tv])
            ni,ci=calibration_parts(len(va),seed)
            for name,ii in [('train',tr),('validation',va),('normalization',va[ni]),('calibration',va[ci]),('known_test',kt),('unknown_test',unk)]:
                rows+=slot_table(own[ii],y[ii],nominal,h,seed,name)
                for sl in range(nominal):
                    vc=pd.Series(y[ii][own[ii]==sl]).value_counts()
                    for label_name,nc in vc.items():classrows.append(dict(holdout=h,seed=seed,split=name,slot=sl,label=label_name,flow_count=int(nc)))
            # Exact legacy slot-selection rule, including split-specific slot range.
            for name,ii in [('known_test',kt),('unknown_test',unk)]:
                nslots=int(own[ii].max(initial=0)+1)
                for fraction in [.1,.2,.3]:
                    nbad=max(1,int(np.ceil(fraction*nslots)))
                    bad=np.random.default_rng(seed+10).choice(np.arange(nslots),size=nbad,replace=False)
                    affected=np.isin(own[ii],bad)
                    for cl in ['ALL']+sorted(np.unique(y[ii]).tolist()):
                        mask=np.ones(len(ii),bool) if cl=='ALL' else y[ii]==cl
                        compromise.append(dict(holdout=h,seed=seed,split=name,requested_fraction=fraction,
                            legacy_slot_range=nslots,chosen_slots=';'.join(map(str,bad.tolist())),class_label=cl,
                            flow_count=int(mask.sum()),affected_count=int(np.sum(mask&affected)),
                            affected_share=float(np.mean(affected[mask])) if mask.any() else 0.))
    tab=pd.DataFrame(rows);atomic_csv(tab,out/'monitor_slot_distribution.csv')
    atomic_csv(pd.DataFrame(classrows),out/'monitor_slot_class_by_partition.csv')
    atomic_csv(pd.DataFrame(compromise),out/'legacy_compromise_affected_flows.csv')
    byclass=pd.crosstab(pd.Series(own,name='slot'),pd.Series(y,name='label')).reindex(range(nominal),fill_value=0)
    byclass.to_csv(out/'monitor_slot_by_label.csv')
    counts=np.bincount(own,minlength=nominal);weights=counts[counts>0]/counts.sum()
    report=dict(agent_column=agent or 'stable_row_index_hash',nominal_slots=nominal,active_slots=np.flatnonzero(counts).tolist(),
        counts=counts.tolist(),total_rows=len(d),largest_slot_share=float(counts.max()/counts.sum()),
        effective_slots_inverse_hhi=float(1/np.sum(weights**2)),
        caution='sVid values are metadata categories, not verified physical monitor identities; missing values follow the submitted string-hash conversion')
    dump(out/'ownership_audit.json',report);return report


def residual_scores(Ms,probs,train_y,classes):
    proto=prototypes(Ms[0],train_y,classes)
    return [residual(m,p,proto) for m,p in zip(Ms[1:],probs)]


def main_metric_rows(data,model_name,score_name,cal,known,unknown,pknown,alpha,extra):
    rows=operating_metrics(cal,known,unknown,data.y_known,data.benign_class,alpha)
    for r in rows:r.update(method=model_name,score=score_name,known_macro_f1=classification_f1(data.y_known,pknown,data.classes),**extra)
    return rows


def peer_audit(data,Ms,seed,nominal,calidx):
    """Refit once for each head on IDEAL train input; freeze across test channels."""
    Mtr,Mv,Mk,Mu=Ms
    Ftr,_,_=event_context(Mtr,data.a_train,data.idx_train,nominal,seed+300)
    Fv,_,_=event_context(Mv,data.a_val,data.idx_val,nominal,seed+400)
    peer=fit_head(Ftr,data.y_train);control=fit_head(Mtr,data.y_train)
    cal_peer=entropy(align_probabilities(peer,Fv,data.classes))[calidx]
    cal_owner=entropy(align_probabilities(control,Mv,data.classes))[calidx]
    mixM=np.r_[Mk,Mu];mixowners=np.r_[data.a_known,data.a_unknown];mixids=np.r_[data.idx_known,data.idx_unknown]
    nk=len(Mk); rows=[]
    settings=[('ideal',0.,0),('loss_only_1pct',.01,0),('delay_only_3',0.,3),('loss1pct_delay3',.01,3),('loss5pct_delay20',.05,20)]
    for setting,loss,delay in settings:
        F,delivered,peercount=event_context(mixM,mixowners,mixids,nominal,seed+500,loss,delay)
        dk=delivered[:nk];du=delivered[nk:]
        for name,model,inputs,cal in [('peer_context',peer,F,cal_peer),('owner_entropy_control',control,mixM,cal_owner)]:
            pk=align_probabilities(model,inputs[:nk][dk],data.classes)
            pu=align_probabilities(model,inputs[nk:][du],data.classes)
            ms=operating_metrics(cal,entropy(pk),entropy(pu),data.y_known[dk],data.benign_class)
            for x in ms:
                x.update(method=name,channel=setting,loss=loss,delay_steps=delay,
                    known_delivery=float(dk.mean()),unknown_delivery=float(du.mean()),
                    known_macro_f1_delivered=classification_f1(data.y_known[dk],pk,data.classes),
                    unknown_end_to_end_recall=float(x['unknown_recall']*du.mean()),
                    peer_available_fraction=float(np.mean(peercount[delivered]>0)),
                    model_refitted_per_channel=False,label_separated_test_streams=False,
                    interpretation='Synthetic event order; dependent scores, empirical calibration only')
                rows.append(x)
    return rows


def run_trial(cfgpath,seed,trial,out,alphas):
    started=time.perf_counter();print(f'  Preparing {cfgpath.name}, seed {seed}',flush=True)
    data=prepare_round2_data(cfgpath,seed)
    cfg=ref.load_yaml(cfgpath);nominal=int(cfg['agents']['n_agents'])
    print('  Training source classifiers and anomaly models once',flush=True)
    cls=ref.train_local_classifiers(data,seed,int(cfg['model']['n_estimators']))
    ifs=ref.train_local_anomaly(data,seed)
    Xs=[data.X_train,data.X_val,data.X_known,data.X_unknown]
    owners=[data.a_train,data.a_val,data.a_known,data.a_unknown]
    p=[ref.owner_proba(cls,X,o,data.classes) for X,o in zip(Xs,owners)]
    proxies=[ref.owner_proxy(cls,X,o,k=1) for X,o in zip(Xs,owners)]
    a_raw=[ref.owner_anomaly(ifs,X,o) for X,o in zip(Xs,owners)]
    astats=scaler(a_raw[0]);avalues=[scale(x,astats) for x in a_raw]
    del a_raw
    normidx,calidx=calibration_parts(len(data.y_val),seed)
    info=dict(seed=seed,held_out_attack=data.unknown_attack,
              protocol='round2_training_only_screen_packet_decoded_independent_norm7_cal7',
              transformed_d=data.transformed_feature_count,K=len(data.classes),nominal_slots=nominal,
              normalization_n=len(normidx),calibration_n=len(calidx))
    counts=np.bincount(data.a_train,minlength=len(cls));active=np.flatnonzero(counts>0)
    classmap=[dict(slot=i,n_train=int(counts[i]),trained_on_local_data=bool(counts[i]>0),
                   local_classes=list(map(str,m.classes_)),global_classes=data.classes,
                   included_in_corrected_average=bool(counts[i]>0)) for i,m in enumerate(cls)]
    dump(trial/'class_alignment_audit.json',classmap)
    rows=[];score_cache={};Ms24=None;curve_rows=[];hist_rows=[];assignment_rows=[]
    for layout in ['12B','16Q','24B']:
        print(f'  Evaluating packet-decoded {layout}',flush=True)
        Ms=[packet_message(pp,phi,aa,layout)[0] for pp,phi,aa in zip(p,proxies,avalues)]
        head=fit_head(Ms[0],data.y_train);probs=[align_probabilities(head,m,data.classes) for m in Ms[1:]]
        u=[1-x.max(axis=1) for x in probs];r=residual_scores(Ms,probs,data.y_train,data.classes)
        received_a=[m[:,-1].astype(float) for m in Ms[1:]]
        # All three score normalizers are fitted on NORMALIZATION rows only.
        refs=[scaler(u[0][normidx]),scaler(received_a[0][normidx]),scaler(r[0][normidx])]
        uu=[scale(x,refs[0]) for x in u];aa=[scale(x,refs[1]) for x in received_a];rr=[scale(x,refs[2]) for x in r]
        s=[composite(x,y,z) for x,y,z in zip(uu,aa,rr)]
        method='Class+anomaly-12B' if layout=='12B' else 'X-MAG-COS-'+layout
        extra={**info,'message_bytes_per_flow':int(layout.replace('B','').replace('Q','')),
               'message_dimension':Ms[0].shape[1],'active_training_slots':len(active)}
        rows+=main_metric_rows(data,method,'composite',s[0][calidx],s[1],s[2],probs[1],alphas,extra)
        score_cache.update({f'{layout}_cal':s[0][calidx],f'{layout}_known':s[1],f'{layout}_unknown':s[2]})
        if data.unknown_attack in ['UDPFlood','SlowrateDoS']:
            ybin=np.r_[np.zeros(len(s[1]),int),np.ones(len(s[2]),int)]
            fpr,tpr,_=roc_curve(ybin,np.r_[s[1],s[2]])
            grid=np.unique(np.r_[0.,np.logspace(-6,-1,600),np.linspace(.1,1,600)])
            interp=np.interp(grid,fpr,tpr)
            for fx,ty in zip(grid,interp):curve_rows.append(dict(seed=seed,held_out_attack=data.unknown_attack,layout=layout,fpr=fx,tpr_interpolated=ty))
            for population,values in [('known',s[1]),('unknown',s[2])]:
                count,edges=np.histogram(values,bins=np.linspace(0,1,101))
                for lo,hi,n in zip(edges[:-1],edges[1:],count):hist_rows.append(dict(seed=seed,held_out_attack=data.unknown_attack,layout=layout,population=population,bin_left=lo,bin_right=hi,count=int(n)))
            pred=np.asarray(data.classes)[probs[2].argmax(axis=1)]
            pu=pvalues(s[0][calidx],s[2])
            for alpha in alphas:
                accepted=pu>alpha
                for label_name in data.classes:assignment_rows.append(dict(seed=seed,held_out_attack=data.unknown_attack,layout=layout,alpha=alpha,
                    assigned_known_class=label_name,count=int(np.sum(accepted&(pred==label_name))),accepted_total=int(accepted.sum()),unknown_total=len(s[2])))

        if layout=='24B':
            Ms24=Ms
            for name,arrs in [('uncertainty',u),('anomaly_only',aa),('residual_only',rr),
                              ('uncertainty_anomaly',[.5*x+.5*y for x,y in zip(uu,aa)])]:
                rows+=main_metric_rows(data,'X-MAG-COS-24B',name,arrs[0][calidx],arrs[1],arrs[2],probs[1],alphas,extra)
            assert all(np.array_equal(composite(x,y,z,0,0,0),y) for x,y,z in zip(uu,aa,rr))
            # Same anomaly score, different decisions: retain boundary ties explicitly.
            tie=[]
            for alpha in alphas:
                cal=aa[0][calidx];tau=float(np.quantile(cal,1-alpha,method='higher'))
                tie.append(dict(alpha=alpha,threshold=tau,calibration_equal_threshold=int(np.sum(cal==tau)),
                    calibration_greater_threshold=int(np.sum(cal>tau)),
                    recall_quantile_ge=float(np.mean(aa[2]>=tau)),
                    recall_conformal=float(np.mean(pvalues(cal,aa[2])<=alpha)),
                    matched_zero_weight_scores_identical=True))
            atomic_csv(pd.DataFrame(tie),trial/'anomaly_tie_audit.csv')
    print('  Auditing active-source versus legacy all-slot probability averages',flush=True)
    bp=[active_average(cls,X,data.classes,counts) for X in Xs[1:]]
    legacy=[ref.average_proba(cls,X,data.classes) for X in Xs[1:]]
    # Invariance to global class order; catches index misuse without assuming class balance.
    probe=Xs[1][:min(256,len(Xs[1]))]
    q=active_average(cls,probe,data.classes,counts)
    qr=active_average(cls,probe,list(reversed(data.classes)),counts)[:,::-1]
    if not np.allclose(q,qr,atol=1e-12,rtol=0):raise RuntimeError('Class-index alignment failed')
    for name,preds in [('Active-source-probability-average',bp),('Legacy-all-slot-probability-average',legacy)]:
        e=[entropy(x) for x in preds]
        es=scaler(e[0][normidx]);as_=scaler(avalues[1][normidx])
        s=[.5*scale(x,es)+.5*scale(y,as_) for x,y in zip(e,avalues[1:])]
        extra={**info,'message_bytes_per_flow':4*len(data.classes)*(len(active) if name.startswith('Active') else len(cls))+4,
               'message_dimension':len(data.classes)+1,'active_training_slots':len(active),
               'n_classifiers_averaged':len(active) if name.startswith('Active') else len(cls),
               'class_permutation_max_error':float(np.max(abs(q-qr)))}
        rows+=main_metric_rows(data,name,'entropy_anomaly',s[0][calidx],s[1],s[2],preds[1],alphas,extra)
    # Strong centralized control uses the exact same calibration row identities.
    central=ref.rf_model(seed,int(cfg['model']['n_estimators'])).fit(data.X_train,data.y_train)
    cp=[align_probabilities(central,X,data.classes) for X in Xs[1:]];ce=[entropy(x) for x in cp]
    rows+=main_metric_rows(data,'Central-RF-full','entropy',ce[0][calidx],ce[1],ce[2],cp[1],alphas,
                         {**info,'message_bytes_per_flow':4*data.transformed_feature_count,'message_dimension':data.transformed_feature_count,
                          'active_training_slots':len(active)})
    if data.unknown_attack in ['UDPFlood','SlowrateDoS']:
        print('  Replaying mixed test events for the paired channel controls',flush=True)
        pe=pd.DataFrame(peer_audit(data,Ms24,seed,nominal,calidx))
        pe['seed']=seed;pe['held_out_attack']=data.unknown_attack
        atomic_csv(pe,trial/'peer_context_metrics.csv')
    for name,records in [('diagnostic_curves.csv',curve_rows),('diagnostic_histograms.csv',hist_rows),('accepted_assignments.csv',assignment_rows)]:
        if records:atomic_csv(pd.DataFrame(records),trial/name)
    score_cache['known_labels']=np.asarray(data.y_known,str)
    np.savez_compressed(trial/'round2_scores.npz',**score_cache)
    atomic_csv(pd.DataFrame(rows),trial/'metrics.csv')
    dump(trial/'protocol.json',{**info,'source_classes':classmap,
        'active_slots':active.tolist(),'elapsed_seconds':time.perf_counter()-started,
        'old_runs_overwritten':False,'anomaly_input':'decoded packet field only',
        'normalization_calibration_overlap':int(len(set(normidx)&set(calidx)))})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',default='data/5G-NIDD/Encoded.csv')
    ap.add_argument('--configs',default='configs/holdouts')
    ap.add_argument('--outdir',default='results/mdpi_r2')
    ap.add_argument('--runs-root',default='runs/mdpi_r2')
    ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    ap.add_argument('--holdouts',nargs='+',default=HOLDS)
    ap.add_argument('--metadata-only',action='store_true')
    args=ap.parse_args();csv=Path(args.csv)
    if not csv.is_file():raise SystemExit(f'Missing original real dataset: {csv}. No synthetic substitution is permitted.')
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    configs=[]
    for h in args.holdouts:
        path=Path(args.configs)/f'real_5g_nidd_{h.lower()}.yaml'
        if not path.is_file():raise SystemExit(f'Missing holdout configuration: {path}')
        config=yaml.safe_load(path.read_text())
        if Path(config['dataset']['path']).resolve()!=csv.resolve():
            raise SystemExit(f'{path} points to a different dataset. Align its dataset.path with --csv before running.')
        configs.append(path)
    fingerprint=dict(dataset_sha256=sha(csv),reference_core_sha256=sha(ref.__file__),
        audit_core_sha256=sha(Path(__file__).with_name('xmag_round2_core.py')),runner_sha256=sha(__file__),
        python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=__import__('sklearn').__version__)
    if (out/'environment_and_provenance.json').exists():
        previous=json.loads((out/'environment_and_provenance.json').read_text())
        if previous!=fingerprint:raise SystemExit('Output directory has a different audit fingerprint. Choose fresh --outdir and --runs-root; existing evidence has been preserved.')
    dump(out/'environment_and_provenance.json',fingerprint)
    print('[1/3] Counting all eight slots and each train/validation/test split',flush=True)
    metadata_audit(csv,configs,args.seeds,out)
    if args.metadata_only:
        print('Metadata audit complete; no model trained.');return
    print('[2/3] Corrected evidence, active-source baseline, and channel evaluation',flush=True)
    complete=[]
    for seed in args.seeds:
        for path in configs:
            trial=Path(args.runs_root)/f'seed{seed}'/path.stem
            tag={**fingerprint,'seed':seed,'configuration_sha256':sha(path)}
            done=trial/'COMPLETE.json'
            if done.is_file() and json.loads(done.read_text())==tag and (trial/'metrics.csv').is_file():
                print(f'Reusing completed {trial}',flush=True);complete.append(trial);continue
            if done.is_file():
                raise SystemExit(f'Cached trial has a different code/data fingerprint: {trial}. Choose another --runs-root; old results have been preserved.')
            trial.mkdir(parents=True,exist_ok=True)
            run_trial(path,seed,trial,out,[.05,.01,.001]);dump(done,tag);complete.append(trial)
    print('[3/3] Collecting results and an upload archive (no raw traffic)',flush=True)
    metric=pd.concat([pd.read_csv(t/'metrics.csv') for t in complete],ignore_index=True)
    atomic_csv(metric,out/'all_metrics_round2.csv')
    numeric=['known_macro_f1','unknown_auroc','unknown_recall','known_false_rejection_rate','benign_false_rejection_rate',
             'unknown_average_precision','standardized_partial_auroc_fpr01']
    summary=metric.groupby(['method','score','alpha'])[numeric].agg(['mean','std','min','max']);summary.to_csv(out/'summary_round2.csv')
    for filename in ['anomaly_tie_audit.csv','peer_context_metrics.csv','diagnostic_curves.csv','diagnostic_histograms.csv','accepted_assignments.csv']:
        frames=[]
        for t in complete:
            if (t/filename).is_file():
                v=pd.read_csv(t/filename)
                if 'seed' not in v:v['seed']=int(t.parent.name[4:])
                if 'held_out_attack' not in v:v['held_out_attack']=t.name
                frames.append(v)
        if frames:atomic_csv(pd.concat(frames,ignore_index=True),out/filename)
    auditrows=[]
    for t in complete:
        for record in json.loads((t/'class_alignment_audit.json').read_text()):
            auditrows.append({**record,'trial':str(t)})
    dump(out/'class_alignment_all_trials.json',auditrows)
    dump(out/'RUN_COMPLETENESS.json',dict(completed_trials=len(complete),expected_trials=len(configs)*len(args.seeds),
        requested_seeds=args.seeds,requested_holdouts=args.holdouts,status='complete_for_requested_scope',
        full_40_trial_scope=(set(args.seeds)==set(SEEDS) and set(args.holdouts)==set(HOLDS))))
    archive=out.parent/'xmag_round2_audit_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file() and f.suffix in ['.csv','.json','.txt','.md']:z.write(f,f.relative_to(out.parent))
    print(f'Completed. Share: {archive}',flush=True)
if __name__=='__main__':main()
