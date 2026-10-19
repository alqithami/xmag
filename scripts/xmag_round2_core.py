#!/usr/bin/env python3
"""Audited round-2 operations. No imports from data-dependent plotting code.

These helpers do not overwrite round-1 functions or results. The new peer path
never receives labels, uses one mixed test stream, separates loss from delay,
and records dropped messages as missing decisions rather than zero observations.
"""
from __future__ import annotations
import numpy as np
from scipy.special import xlogy
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def align_probabilities(model, X, classes):
    classes=list(map(str,classes)); local=list(map(str,model.classes_))
    if len(set(classes)) != len(classes) or len(set(local)) != len(local):
        raise ValueError('Class labels must be unique')
    if set(local)-set(classes): raise ValueError('Model contains an unknown class label')
    raw=np.asarray(model.predict_proba(X),float)
    if raw.shape!=(len(X),len(local)): raise ValueError('Bad probability matrix shape')
    if not np.isfinite(raw).all() or (raw<0).any(): raise ValueError('Invalid probabilities')
    out=np.zeros((len(X),len(classes)),float)
    for col, name in enumerate(local): out[:,classes.index(name)]=raw[:,col]
    total=out.sum(axis=1,keepdims=True)
    if (total<=0).any(): raise ValueError('Zero probability row')
    return out/total


def active_average(models, X, classes, counts):
    counts=np.asarray(counts,int)
    if len(counts)!=len(models): raise ValueError('Counts must index exactly the model slots')
    active=np.flatnonzero(counts>0)
    if not len(active): raise ValueError('No trained source available')
    acc=np.zeros((len(X),len(classes)),float)
    for slot in active: acc+=align_probabilities(models[slot],X,classes)
    return acc/len(active)


def entropy(p):
    p=np.asarray(p,float)
    return -xlogy(p,p).sum(axis=1)/np.log(max(2,p.shape[1]))


def scaler(ref):
    ref=np.asarray(ref,float)
    if not len(ref) or not np.isfinite(ref).all(): raise ValueError('Invalid normalization sample')
    lo,hi=np.percentile(ref,[5,95]); den=hi-lo
    if den<=1e-12: den=ref.std()
    if den<=1e-12: den=1.
    return float(lo),float(den)


def scale(x, fitted):
    x=np.asarray(x,float)
    if not np.isfinite(x).all(): raise ValueError('Invalid score')
    return np.clip((x-fitted[0])/fitted[1],0,1)


def composite(u,a,r,beta=.25,lam=.5,gamma=.25):
    if not all(0<=x<=1 for x in (beta,lam,gamma)): raise ValueError('Weights outside [0,1]')
    s1=beta*r+(1-beta)*u; s2=lam*u+(1-lam)*a
    return gamma*np.maximum(s1,s2)+(1-gamma)*s2


def pvalues(cal, scores):
    cal=np.asarray(cal,float); scores=np.asarray(scores,float)
    if not len(cal) or not np.isfinite(cal).all() or not np.isfinite(scores).all():
        raise ValueError('Calibration/test scores must be finite and nonempty')
    ordered=np.sort(cal)
    return (1+len(ordered)-np.searchsorted(ordered,scores,side='left'))/(len(ordered)+1)


def operating_metrics(cal, known, unknown, yknown, benign, alphas=(.05,.01,.001)):
    known=np.asarray(known,float); unknown=np.asarray(unknown,float)
    yknown=np.asarray(yknown,str)
    if not len(known) or not len(unknown) or len(known)!=len(yknown):
        raise ValueError('Invalid held-out evaluation arrays')
    y=np.r_[np.zeros(len(known),int),np.ones(len(unknown),int)]
    score=np.r_[known,unknown]
    pk=pvalues(cal,known); pu=pvalues(cal,unknown)
    auc=roc_auc_score(y,score); ap=average_precision_score(y,score)
    partial=roc_auc_score(y,score,max_fpr=.01)
    benign_mask=(yknown==str(benign)); rows=[]
    for alpha in alphas:
        rejected_k=pk<=alpha; rejected_u=pu<=alpha
        rows.append(dict(alpha=alpha,unknown_auroc=float(auc),unknown_average_precision=float(ap),
            standardized_partial_auroc_fpr01=float(partial),
            unknown_prevalence=len(unknown)/len(y),n_calibration=len(cal),
            n_known=len(known),n_benign=int(benign_mask.sum()),n_unknown=len(unknown),
            unknown_recall=float(rejected_u.mean()),known_false_rejection_rate=float(rejected_k.mean()),
            benign_false_rejection_rate=float(rejected_k[benign_mask].mean()) if benign_mask.any() else float('nan'),
            n_benign_rejected=int(rejected_k[benign_mask].sum()),
            minimum_pvalue=1/(len(cal)+1)))
    return rows


def calibration_parts(n,seed):
    if n<20: raise ValueError('Too few validation samples to separate normalization and calibration')
    order=np.random.default_rng(int(seed)+9017).permutation(n)
    return order[:n//2],order[n//2:]


def fit_head(M,y):
    base=LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=1000)
    return OneVsRestClassifier(base,n_jobs=1).fit(M,y)


def prototypes(M,y,classes):
    M=np.asarray(M,np.float32); y=np.asarray(y,str)
    std=M.std(axis=0,dtype=np.float64); std[std<1e-8]=1.
    means=np.stack([M[y==str(c)].mean(axis=0,dtype=np.float64) for c in classes])
    if not np.isfinite(means).all(): raise ValueError('Prototype class without training observations')
    return means,std


def residual(M,probs,proto,chunk=4096):
    means,std=proto; pred=np.argmax(probs,axis=1)
    out=np.empty(len(M),float)
    for start in range(0,len(M),chunk):
        end=min(start+chunk,len(M))
        z=(M[start:end]-means[pred[start:end]])/std
        out[start:end]=np.sqrt(np.mean(z*z,axis=1))
    return out


def packet_message(p,proxy,anomaly,layout):
    """Serialize then decode the scored evidence; no full-precision anomaly bypass.

    Wire layouts are little-endian, packed (no platform-dependent alignment).
    12B is a declared two-byte-header design for the class/anomaly ablation.
    16Q has four header bytes, nine evidence bytes and three reserved bytes.
    """
    p=np.asarray(p); proxy=np.asarray(proxy); anomaly=np.asarray(anomaly)
    n,K=p.shape; D=proxy.shape[1]
    if proxy.shape[0]!=n or anomaly.shape!=(n,): raise ValueError('Evidence row mismatch')
    c=np.argmax(p,axis=1); j=np.argmax(np.abs(proxy),axis=1); rows=np.arange(n)
    if layout=='12B':
        dtype=np.dtype([('header','<u2'),('class_id','<u2'),('prob','<f4'),('anomaly','<f4')])
    elif layout=='16Q':
        dtype=np.dtype([('header','<u4'),('class_id','u1'),('prob','<f2'),('proxy','<f2'),
                        ('anomaly','<f2'),('feature_id','<u2'),('reserved','u1',(3,))])
    elif layout=='24B':
        dtype=np.dtype([('header','<u8'),('class_id','<u2'),('prob','<f4'),('feature_id','<u2'),
                        ('proxy','<f4'),('anomaly','<f4')])
    else: raise ValueError('Unknown layout')
    if K>256 or D>65536: raise ValueError('Evidence identifiers exceed declared limits')
    if not all(np.isfinite(x).all() for x in (p,proxy,anomaly)): raise ValueError('Nonfinite input evidence')
    if layout=='16Q' and np.abs(proxy[rows,j]).max(initial=0)>65504:
        raise ValueError('Attribution proxy overflows binary16; do not silently clip research inputs')
    wire=np.zeros(n,dtype=dtype);wire['class_id']=c;wire['prob']=p[rows,c];wire['anomaly']=anomaly
    if layout!='12B': wire['feature_id']=j;wire['proxy']=proxy[rows,j]
    # Actual serialized bytes are read back before downstream scoring.
    received=np.frombuffer(wire.tobytes(),dtype=dtype)
    m=np.zeros((n,K+1 if layout=='12B' else K+D+1),np.float32)
    m[rows,received['class_id'].astype(int)]=received['prob']
    if layout!='12B': m[rows,K+received['feature_id'].astype(int)]=received['proxy']
    m[:,-1]=received['anomaly']
    return m,dtype.itemsize


def event_context(M,owners,ids,nominal_slots=8,seed=42,loss=0.,delay_steps=0):
    """One label-blind replay, preserving row identity after delayed arrivals.

    `ids` are original row identities, not capture timestamps. A random simulated
    send order is explicitly used; no physical-time interpretation is implied.
    A dropped current message has no decision. Context contains prior delivered
    messages from OTHER owners only. It never depends on known/unknown labels.
    """
    M=np.asarray(M,np.float32); owners=np.asarray(owners,int); ids=np.asarray(ids,np.int64)
    n=len(M)
    if n!=len(owners) or n!=len(ids) or len(np.unique(ids))!=n: raise ValueError('Duplicate/misaligned event identities')
    if (owners<0).any() or (owners>=nominal_slots).any(): raise ValueError('Owner outside monitor slots')
    if not 0<=loss<1 or delay_steps<0: raise ValueError('Bad channel setting')
    # Input-order invariance: random draws are first indexed by canonical identities.
    canonical=np.argsort(ids,kind='stable')
    send_order=canonical[np.random.default_rng(seed).permutation(n)]
    delivered=np.ones(n,bool)
    mask=np.random.default_rng(seed+1111).random(n)>=loss
    delivered[canonical]=mask
    delays=np.zeros(n,int)
    if delay_steps: delays[canonical]=np.random.default_rng(seed+2222).integers(0,delay_steps+1,n)
    send_rank=np.empty(n,int);send_rank[send_order]=np.arange(n)
    arrival=send_rank+delays
    order=np.lexsort((ids,send_rank,arrival))
    state=np.zeros((nominal_slots,M.shape[1]),np.float32); seen=np.zeros(nominal_slots,bool)
    pooled=np.zeros_like(M); count=np.zeros(n,int)
    for i in order:
        if not delivered[i]: continue
        owner=int(owners[i]); peers=seen.copy();peers[owner]=False
        count[i]=int(peers.sum())
        if count[i]: pooled[i]=state[peers].mean(axis=0)
        state[owner]=M[i];seen[owner]=True
    features=np.c_[M,pooled,(count/max(1,nominal_slots-1)).astype(np.float32)]
    return features,delivered,count


def classification_f1(y,p,classes):
    return float(f1_score(np.asarray(y,str),np.asarray(classes)[np.argmax(p,axis=1)],
                          labels=list(classes),average='macro',zero_division=0))
