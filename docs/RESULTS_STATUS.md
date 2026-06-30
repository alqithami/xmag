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

## Main all-holdout summary

From `results/tables/table_xmag_cos_all_holdouts_summary.csv`:

```text
X-MAG-COS-24B
mean known_macro_f1     = 0.997936
min known_macro_f1      = 0.997021
mean unknown_auroc      = 0.950247
min unknown_auroc       = 0.789850
mean unknown_recall     = 0.851704
min unknown_recall      = 0.424966
max false alarm         = 0.051147
message size            = 24 bytes
mean runtime            = 10.34 s
```

## Why COS replaced DH

The earlier X-MAG-DH-24B design used the same 24-byte message but relied on coordinator uncertainty for open-set scoring. It failed on UDPFlood:

```text
DH-24B UDPFlood AUROC  = 0.489338
DH-24B UDPFlood recall = 0.013294
```

X-MAG-COS-24B recovers this failure case:

```text
COS-24B UDPFlood AUROC  = 0.789850
COS-24B UDPFlood recall = 0.424966
```

## Next required validation

Run seed stability over random states `7`, `42`, and `123`:

```bash
bash scripts/run_seed_stability_cos24.sh
python scripts/aggregate_seed_stability.py
```

The manuscript should not make final statistical claims until this stability table is available.
