# Results status

The committed result tables are preliminary real 5G-NIDD tables from the extracted `Encoded.csv` file.

## Main compact setting: X-MAG-DH-24B

`results/tables/table_main_xmag_dh_24b_all_holdouts.csv` summarizes the current compact setting:

```text
method = xmag_top1_class_token_anomaly_fusion
k = 1
top_m = 1
alpha = 1.0
message = 24 bytes per flow
```

Summary:

```text
mean known_macro_f1 = 0.997936
mean unknown_auroc = 0.903635
mean unknown_recall_at_threshold = 0.744105
max false alarm = 0.050912
message = 24 bytes per flow
```

Failure case:

```text
UDPFlood unknown:
unknown_auroc = 0.489338
unknown_recall_at_threshold = 0.013294
```

## Baseline comparison

`results/tables/table_xmag_vs_baseline_all_holdouts.csv` compares the 24-byte X-MAG setting with the 36-byte logit-average-plus-local-anomaly baseline.

Summary:

```text
baseline mean AUROC = 0.918934
X-MAG mean AUROC = 0.903635
baseline mean recall = 0.833425
X-MAG mean recall = 0.744105
message reduction = 33.33%
```

## Interpretation

Current evidence supports a nuanced paper claim:

> Compact hybrid explanation messages preserve known-attack classification and reduce communication cost, but UDPFlood exposes a failure mode where a 24-byte message is insufficient for robust open-set detection.

The next experiment is a targeted failure-case sweep over `UDPFlood` and `SlowrateDoS`.
