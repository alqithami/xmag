# Seed-stability sweep

The seed-stability sweep is complete.

## Scope

```text
random_state in {7, 42, 123}
holdouts = all eight 5G-NIDD attack families
variant = X-MAG-COS-24B
```

The script uses:

```text
k = 1
top_m = 1
gamma = 0.25
score = linear_max_resid_owner_b025__uncert_anomaly_b05_gamma_0.25
message_bytes_per_flow = 24
```

## Run

```bash
bash scripts/run_seed_stability_cos24.sh
python scripts/aggregate_seed_stability.py
```

## Outputs

```text
results/tables/table_seed_stability_cos24_all_rows.csv
results/tables/table_seed_stability_cos24_by_attack.csv
results/tables/table_seed_stability_cos24_overall.csv
```

## Overall result

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

## Interpretation

The result supports the main method while making the final claim more conservative. Known-class macro-F1 and false-alarm rate are highly stable across seeds. Unknown detection remains strong on average, but UDPFlood and SlowrateDoS produce most of the variance. The final manuscript should report mean plus standard deviation and should not rely only on the seed-42 table.