# MDPI Mathematics Round-1 Result Summary

## Matched 5G-NIDD protocol

All principal results use five seeds and eight leave-one-attack-family-out tasks, giving forty matched trials per method.

### Full-evidence message frontier

| Method | Bytes/flow | Known macro-F1 | Unknown AUROC | Unknown recall | Mean FPR |
|---|---:|---:|---:|---:|---:|
| Class + anomaly | 12 | 0.998736 | 0.900658 | 0.711659 | 0.047471 |
| X-MAG-COS-16Q | 16 | 0.998748 | 0.902236 | 0.723842 | 0.049728 |
| Class + proxy | 18 | 0.998769 | 0.898034 | 0.700373 | 0.041680 |
| X-MAG-COS-20B | 20 | 0.998749 | 0.902230 | 0.723818 | 0.049742 |
| X-MAG-COS-24B | 24 | 0.998749 | 0.902230 | 0.723818 | 0.049742 |
| X-MAG-COS-30B | 30 | 0.998764 | 0.899142 | 0.688294 | 0.049549 |

The 16-byte quantized payload retains the behavior of the 20- and 24-byte references. It is therefore the compact operating point in the revision; this is an empirical Pareto choice, not a proof of universal optimality.

### Baseline interpretation

- Central Random Forest with entropy rejection: AUROC 0.887291 and recall 0.798623 over the same forty tasks.
- FedAvg-linear: AUROC 0.900359 and recall 0.660022, but lower known macro-F1 (0.901329).
- Distributed all-source logit averaging: AUROC 0.850527 and recall 0.510383, with poor known-class performance under the source-aware non-IID ownership partition.
- X-MAG-DH coordinator uncertainty: AUROC 0.905045 and recall 0.704205.

After Holm correction, the evidence supports superiority over the distributed logit-average baseline, but it does not support a broad claim of superiority over the centralized Random Forest, FedAvg-linear, or the earlier DH head.

## Attribution-proxy fidelity

| Holdout | Top-1 agreement | Top-3 Jaccard | Spearman rank | Speed-up |
|---|---:|---:|---:|---:|
| UDPFlood | 0.270570 | 0.650105 | 0.936730 | 156.7× |
| SlowrateDoS | 0.408968 | 0.416099 | 0.907938 | 73.1× |

The proxy has high absolute-rank agreement with TreeSHAP but only moderate exact-feature agreement. It should be described as an attribution proxy, not a SHAP-equivalent explanation.

## CICIoT2023 stress test

The 16-, 20-, and 24-byte full-evidence variants were nearly indistinguishable:

```text
known macro-F1 ≈ 0.72236
unknown AUROC  ≈ 0.70148
unknown recall ≈ 0.20609
```

The distributed logit-average baseline achieved AUROC 0.764014. CICIoT2023 therefore establishes a limitation and does not support a broad cross-dataset generalization claim.

## Files

- `protocol_matched_summary.csv`: principal aggregate table.
- `all_metrics.csv`: trial-level results.
- `paired_statistics.csv`: paired tests, effect sizes, and confidence intervals.
- `hyperparameter_sensitivity*.csv`: score-weight sensitivity.
- `network_attack_*.csv`: network and targeted manipulation experiments.
- `shap_fidelity_*.csv`: attribution-proxy validation.
- `message_pareto.csv`: communication frontier.
