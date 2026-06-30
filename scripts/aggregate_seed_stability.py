#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import pandas as pd

TARGET_SCORE = "linear_max_resid_owner_b025__uncert_anomaly_b05_gamma_0.25"
rows = []

for p in sorted(Path("runs").glob("seed*_cos24_real_5g_nidd_*/composite_open_score_diagnostic.csv")):
    parent = p.parent.name
    seed = int(parent.split("_cos24_")[0].replace("seed", ""))
    attack = parent.split("real_5g_nidd_")[-1]
    df = pd.read_csv(p)
    keep = df[(df["score"] == TARGET_SCORE) & (df["k"] == 1) & (df["top_m"] == 1)].copy()
    if keep.empty:
        continue
    keep.insert(0, "seed", seed)
    keep.insert(1, "held_out_attack", attack)
    rows.append(keep)

if not rows:
    raise SystemExit("No seed-stability COS-24B result files found.")

all_rows = pd.concat(rows, ignore_index=True)
cols = [
    "seed", "held_out_attack", "method", "score", "k", "top_m",
    "known_macro_f1", "unknown_auroc", "unknown_recall_at_threshold",
    "known_false_alarm_rate_at_threshold", "message_bytes_per_flow", "seconds",
]
all_rows = all_rows[[c for c in cols if c in all_rows.columns]]

metric_cols = [
    "known_macro_f1",
    "unknown_auroc",
    "unknown_recall_at_threshold",
    "known_false_alarm_rate_at_threshold",
    "seconds",
]

by_attack = all_rows.groupby("held_out_attack")[metric_cols].agg(["mean", "std", "min", "max"])
by_attack.columns = ["_".join(c).strip() for c in by_attack.columns]
by_attack = by_attack.reset_index()

overall = pd.DataFrame([{
    "scope": "MEAN_OVER_ALL_ROWS",
    "known_macro_f1_mean": all_rows["known_macro_f1"].mean(),
    "known_macro_f1_std": all_rows["known_macro_f1"].std(),
    "unknown_auroc_mean": all_rows["unknown_auroc"].mean(),
    "unknown_auroc_std": all_rows["unknown_auroc"].std(),
    "unknown_recall_mean": all_rows["unknown_recall_at_threshold"].mean(),
    "unknown_recall_std": all_rows["unknown_recall_at_threshold"].std(),
    "false_alarm_mean": all_rows["known_false_alarm_rate_at_threshold"].mean(),
    "false_alarm_std": all_rows["known_false_alarm_rate_at_threshold"].std(),
    "worst_unknown_auroc": all_rows["unknown_auroc"].min(),
    "worst_unknown_recall": all_rows["unknown_recall_at_threshold"].min(),
    "message_bytes_per_flow": all_rows["message_bytes_per_flow"].max(),
}])

Path("results/tables").mkdir(parents=True, exist_ok=True)
all_rows.to_csv("results/tables/table_seed_stability_cos24_all_rows.csv", index=False)
by_attack.to_csv("results/tables/table_seed_stability_cos24_by_attack.csv", index=False)
overall.to_csv("results/tables/table_seed_stability_cos24_overall.csv", index=False)

print("OVERALL")
print(overall.to_string(index=False))
print("\nSaved:")
print("results/tables/table_seed_stability_cos24_all_rows.csv")
print("results/tables/table_seed_stability_cos24_by_attack.csv")
print("results/tables/table_seed_stability_cos24_overall.csv")
