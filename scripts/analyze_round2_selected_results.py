#!/usr/bin/env python3
"""Reanalyse author-supplied R2 observations without retraining or score tuning.

Primary layout family: known macro-F1, AUROC, operational recall at alpha=0.05.
Wilcoxon p-values are exact conditional sign-permutation tails of the nonzero
average ranks (doubled to preserve half-integer ties). Round paired differences
to 14 decimal places before ranking. Family-mean tests are a separate sensitivity
family, not 40 independent replications of the underlying data collection.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_1samp


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p))
    level = 0.0
    for i, j in enumerate(order):
        level = max(level, (len(p)-i)*p[j])
        out[j] = min(1.0, level)
    return out


def signed_rank_exact(d):
    d = np.round(np.asarray(d, dtype=float), 14)
    d = d[d != 0]
    if len(d) == 0:
        return {"n_nonzero": 0, "wilcoxon_W": 0.0, "wilcoxon_p": 1.0, "rank_biserial": 0.0}
    w = np.rint(2*rankdata(np.abs(d), method="average")).astype(int)
    total = int(w.sum())
    plus = int(w[d > 0].sum())
    small = min(plus, total-plus)
    ways = [0]*(total+1)
    ways[0] = 1
    reached = 0
    for value in w:
        value = int(value)
        for s in range(reached, -1, -1):
            ways[s+value] += ways[s]
        reached += value
    p = min(1.0, 2*sum(ways[:small+1])/(2**len(d)))
    return {"n_nonzero": len(d), "wilcoxon_W": small/2, "wilcoxon_p": p,
            "rank_biserial": (2*plus-total)/total}


def bootstrap_mean(d, seed=20260923, n=50000):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    b = rng.choice(d, size=(n,len(d)), replace=True).mean(axis=1)
    low, high = np.quantile(b, [0.025,0.975])
    return float(low), float(high)


def compare(d):
    d = np.asarray(d, dtype=float)
    lo, hi = bootstrap_mean(d)
    t = ttest_1samp(d, 0.0) if np.std(d) > 0 else None
    return {"n_pairs": len(d), "mean_difference": float(d.mean()),
            "std_difference": float(d.std(ddof=1)), "bootstrap_low": lo, "bootstrap_high": hi,
            "t_stat": float(t.statistic) if t else 0.0,
            "t_p": float(t.pvalue) if t else 1.0, **signed_rank_exact(d)}


def validate_inputs(df, peer, provenance):
    expected = set(itertools.product([7,21,42,84,123],
        ["HTTPFlood","ICMPFlood","SYNFlood","SYNScan","SlowrateDoS","TCPConnectScan","UDPFlood","UDPScan"]))
    assert len(df) == 40 and not df.duplicated(["seed","holdout"]).any()
    assert set(zip(df.seed, df.holdout)) == expected
    for col, mean in provenance["expected_layout_means"].items():
        assert np.isclose(df[col].mean(),mean,rtol=0,atol=2e-14), (col,df[col].mean(),mean)
        assert np.isfinite(df[col]).all() and df[col].between(0,1).all()
    assert len(peer) == 10 and not peer.duplicated(["seed","holdout"]).any()
    assert set(zip(peer.seed,peer.holdout)) == set(itertools.product([7,21,42,84,123],["SlowrateDoS","UDPFlood"]))
    assert np.isclose(signed_rank_exact([1,2,3,4,5])["wilcoxon_p"], 0.0625)
    assert signed_rank_exact([0,0])["wilcoxon_p"] == 1
    assert signed_rank_exact([1,-1])["wilcoxon_p"] == 1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",default="results/mdpi_r2_review")
    ap.add_argument("--output-dir",default="results/mdpi_r2_review/derived")
    args=ap.parse_args()
    root=Path(args.input_dir)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(root/"layout_pairs_alpha005.csv")
    peer=pd.read_csv(root/"peer_pairs_alpha005.csv")
    provenance=json.loads((root/"provenance.json").read_text())
    validate_inputs(df,peer,provenance)
    rows=[]
    for unit in ["seed_holdout", "family_mean"]:
        for metric in ["f1","auc","recall"]:
            tmp=df[["holdout"]].copy()
            tmp["d"]=df[metric+"_16"]-df[metric+"_12"]
            d=tmp.d.to_numpy() if unit=="seed_holdout" else tmp.groupby("holdout").d.mean().to_numpy()
            rows.append({"comparison":"16Q minus 12B", "alpha":0.05, "unit":unit,
                         "metric":metric, **compare(d)})
    stats=pd.DataFrame(rows)
    for unit in stats.unit.unique():
        idx=stats.index[stats.unit==unit]
        stats.loc[idx,"wilcoxon_p_holm_three_metrics"]=holm(stats.loc[idx,"wilcoxon_p"])
        stats.loc[idx,"t_p_holm_three_metrics"]=holm(stats.loc[idx,"t_p"])
    stats.to_csv(out/"paired_16q_minus_12b.csv",index=False)
    family=df.groupby("holdout")[["f1_16","auc_16","recall_16","auc_12","recall_12"]].agg(["mean","std","min","max"])
    family.columns=["_".join(c) for c in family.columns]
    family.to_csv(out/"per_family_alpha005.csv")
    ps=peer.groupby("holdout").agg({c:["mean","std"] for c in peer.select_dtypes("number").columns if c!="seed"})
    ps.columns=["_".join(c) for c in ps.columns]
    ps.to_csv(out/"peer_context_alpha005.csv")
    peer_tests=[]
    for name,g in peer.groupby("holdout"):
        for metric in ["auc","recall"]:
            peer_tests.append({"holdout":name,"comparison":"peer minus matched owner", "metric":metric,
                               **compare(g["peer_"+metric]-g["owner_"+metric])})
    pd.DataFrame(peer_tests).to_csv(out/"peer_paired_sensitivity.csv",index=False)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.csv")}
    report={"scope_validated": True,"layout_pairs":40,"peer_pairs":10,"input_sha256":hashes,
            "statistics":stats.to_dict(orient="records"),
            "per_family":family.reset_index().to_dict(orient="records"),
            "peer_summary":ps.reset_index().to_dict(orient="records"),
            "peer_tests":peer_tests,
            "notes":["No equivalence test or analyst-utility evaluation was performed.",
                     "Postprocessing is not a fresh experiment; inputs are supplied observed metrics.",
                     "Seed-level tests do not remove dependence between trials sharing one dataset.",
                     "Peer scores are empirically calibrated under synthetic order, not exchangeability-certified."]}
    (out/"analysis.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("ROUND2_POSTPROCESS_VALIDATED")
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=="__main__": main()
