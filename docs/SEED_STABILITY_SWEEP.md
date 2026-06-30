# Seed-stability sweep

The current main result is single-seed. Before final manuscript claims, run a small stability sweep over:

```text
random_state in {7, 42, 123}
holdouts = all eight 5G-NIDD attack families
variant = X-MAG-COS-24B
```

## Run

```bash
bash scripts/run_seed_stability_cos24.sh
python scripts/aggregate_seed_stability.py
```

The script uses:

```text
k = 1
top_m = 1
gamma = 0.25
score = linear_max_resid_owner_b025__uncert_anomaly_b05_gamma_0.25
```

## Outputs

```text
results/tables/table_seed_stability_cos24_all_rows.csv
results/tables/table_seed_stability_cos24_by_attack.csv
results/tables/table_seed_stability_cos24_overall.csv
```

## Interpretation

Use the overall table to report mean ± standard deviation for known macro-F1, unknown AUROC, unknown recall, false-alarm rate, and worst-case unknown recall.
