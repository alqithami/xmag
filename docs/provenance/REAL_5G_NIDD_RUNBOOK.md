# Real 5G-NIDD runbook

This document records the exact real-data path used for the current X-MAG-DH experiments.

## 1. Dataset files

Download `Encoded.zip` and extract it so that the following file exists:

```bash
data/5G-NIDD/Encoded.csv
```

The expected real file is roughly 467 MB and has about 1,215,891 lines including the header.

## 2. Audit

```bash
python xmag_pipeline.py audit --csv data/5G-NIDD/Encoded.csv --outdir runs/real_5g_nidd_audit
```

Expected labels:

```text
Benign: 477737
HTTPFlood: 140812
ICMPFlood: 1155
SYNFlood: 9721
SYNScan: 20043
SlowrateDoS: 73124
TCPConnectScan: 20052
UDPFlood: 457340
UDPScan: 15906
```

Use the exact label names above. The config must use `SYNFlood`, not `SYN Flood`; and `UDPScan`, not `UDP Scan`.

## 3. Current compact method

The current compact candidate is:

```text
method = xmag_top1_class_token_anomaly_fusion
k = 1
top_m = 1
alpha = 1.0
message = 24 bytes per flow
```

This setting is compact and strong for most holdouts, but it fails on `UDPFlood`. It should be treated as a compact baseline and not as the final method.

## 4. Failure-focused sweep

```bash
bash scripts/run_failure_grid.sh
```

The sweep checks `UDPFlood` and `SlowrateDoS` over multiple alpha and top-m values.
