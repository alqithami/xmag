# X-MAG-IDS

**X-MAG-IDS** is a reproducible research repository for **explanation-aware multi-agent intrusion detection** in 5G-enabled IoT networks.

The current working method is a compact hybrid agent message:

```text
X-MAG-DH / hybrid message = top-m class evidence + top-k explanation evidence + local anomaly scalar
```

The implementation supports 5G-NIDD schema auditing, leave-one-attack-family-out evaluation, open-set scoring, communication-cost accounting, and diagnostic sweeps over compact agent-message designs.

## Current status

The repository has moved beyond the synthetic smoke test. The committed tables under `results/tables/` are **real 5G-NIDD runs** using the extracted `Encoded.csv` file. The current 24-byte X-MAG-DH setting gives strong known-class performance but exposes two failure cases, especially `UDPFlood`, which is now the target of a focused failure-case sweep.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

The public repository still keeps the original one-file smoke pipeline in `xmag_pipeline.py`. The hybrid-message scripts are reviewer artifacts used with the modular local codebase developed during experimentation.

## Dataset setup

The dataset is not redistributed here. Download `Encoded.zip` from the 5G-NIDD dataset page, then place/extract it locally:

```bash
mkdir -p data/5G-NIDD
unzip -o data/5G-NIDD/Encoded.zip -d data/5G-NIDD/encoded_extracted
cp data/5G-NIDD/encoded_extracted/Encoded.csv data/5G-NIDD/Encoded.csv
```

Confirm:

```bash
ls -lh data/5G-NIDD/Encoded.csv
wc -l data/5G-NIDD/Encoded.csv
```

Expected line count for the real encoded file is about `1,215,891`, including the header.

## Audit the real dataset

```bash
python xmag_pipeline.py audit --csv data/5G-NIDD/Encoded.csv --outdir runs/real_5g_nidd_audit
```

Expected attack labels include:

```text
Benign, HTTPFlood, ICMPFlood, SYNFlood, SYNScan, SlowrateDoS, TCPConnectScan, UDPFlood, UDPScan
```

Use the exact label names. The real dataset uses `SYNFlood`, not `SYN Flood`; and `UDPScan`, not `UDP Scan`.

## Current compact result

The current compact setting is:

```text
method = xmag_top1_class_token_anomaly_fusion
k = 1
top_m = 1
alpha = 1.0
message = 24 bytes per flow
```

Summary from `results/tables/table_main_xmag_dh_24b_all_holdouts.csv`:

```text
mean known_macro_f1 = 0.997936
mean unknown_auroc = 0.903635
mean unknown_recall_at_threshold = 0.744105
message_bytes_per_flow = 24
```

Failure case:

```text
UDPFlood unknown: unknown_auroc = 0.489338, recall = 0.013294
```

## Next experiment

Run the focused failure-case sweep over `UDPFlood` and `SlowrateDoS`:

```bash
bash scripts/run_failure_grid.sh
```

This checks whether increasing `top_m` or changing alpha recovers the failure cases without giving up lightweight communication.

## Scientific caution

Do not report synthetic smoke-test numbers as paper results. Only tables produced from the real extracted `data/5G-NIDD/Encoded.csv` should be used, and current tables should be interpreted together with the documented UDPFlood failure mode.
