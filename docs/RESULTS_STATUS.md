# Results status

The current main method is **X-MAG-COS-24B**.

## Frozen method

```text
message = top-1 class evidence + top-1 explanation evidence + local anomaly scalar
k = 1
top_m = 1
score = linear_max_resid_owner_b025__uncert_anomaly_b05_gamma_0.25
message_bytes_per_flow = 24
```

## Three-seed stability result

From `results/tables/table_seed_stability_cos24_overall.csv`:

```text
known_macro_f1_mean  = 0.998053
known_macro_f1_std   = 0.000496
unknown_auroc_mean   = 0.943859
unknown_auroc_std    = 0.093135
unknown_recall_mean  = 0.839499
unknown_recall_std   = 0.279391
false_alarm_mean     = 0.050306
false_alarm_std      = 0.000836
worst_unknown_auroc  = 0.709369
worst_unknown_recall = 0.218584
message_bytes        = 24
```

## By-attack interpretation

Most holdouts are stable and strong. The difficult cases are:

```text
SlowrateDoS:
unknown_auroc_mean  = 0.840089
unknown_recall_mean = 0.447108

UDPFlood:
unknown_auroc_mean  = 0.746750
unknown_recall_mean = 0.301454
```

The seed-stability result is more conservative than the single-seed seed-42 result. The paper should therefore report the three-seed mean plus standard deviation as the main result and keep the seed-42 tables as method-selection/ablation evidence.

## Why COS replaced DH

The earlier X-MAG-DH-24B design used the same 24-byte message but relied on coordinator uncertainty for open-set scoring. It failed on UDPFlood in the seed-42 run:

```text
DH-24B UDPFlood AUROC  = 0.489338
DH-24B UDPFlood recall = 0.013294
```

X-MAG-COS-24B recovered this failure case under seed 42:

```text
COS-24B UDPFlood AUROC  = 0.789850
COS-24B UDPFlood recall = 0.424966
```

Across three seeds, UDPFlood remains the limiting case, with worst observed AUROC 0.709369 and worst observed recall 0.218584.

## Next required validation

The next experiment should be noisy-agent robustness and then cross-dataset validation. The seed-stability sweep is now complete.