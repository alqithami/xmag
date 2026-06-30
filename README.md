# X-MAG-IDS

**X-MAG-IDS** is a reproducible research repository for **explanation-aware multi-agent intrusion detection** in 5G-enabled IoT networks.

The current frozen method is:

```text
X-MAG-COS-24B = top-1 class evidence + top-1 explanation evidence + local anomaly scalar
```

Each local agent sends a 24-byte message. A coalition classifier predicts known attack classes, and a separate composite open-set head detects unknown attacks using message residual, owner-agent uncertainty, and anomaly evidence.

## Current status

The repository has moved beyond synthetic smoke testing. The current main result uses real 5G-NIDD `Encoded.csv` with leave-one-attack-family-out evaluation across eight attack holdouts and a three-seed stability sweep over random states `7`, `42`, and `123`.

Three-seed stability result from `results/tables/table_seed_stability_cos24_overall.csv`:

```text
X-MAG-COS-24B
known macro-F1       = 0.998053 +/- 0.000496
unknown AUROC        = 0.943859 +/- 0.093135
unknown recall       = 0.839499 +/- 0.279391
false-alarm rate     = 0.050306 +/- 0.000836
worst unknown AUROC  = 0.709369
worst unknown recall = 0.218584
message size         = 24 bytes per flow
```

The earlier single-seed result for seed `42` was stronger on the hardest UDPFlood case, but the three-seed sweep gives the more conservative paper claim. UDPFlood remains the limiting holdout.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

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

The real encoded file used in our run has about `1,215,891` lines including the header.

## Audit the real dataset

```bash
python xmag_pipeline.py audit --csv data/5G-NIDD/Encoded.csv --outdir runs/real_5g_nidd_audit
```

Expected attack labels:

```text
Benign, HTTPFlood, ICMPFlood, SYNFlood, SYNScan, SlowrateDoS, TCPConnectScan, UDPFlood, UDPScan
```

Use the exact label names. The real dataset uses `SYNFlood`, not `SYN Flood`; and `UDPScan`, not `UDP Scan`.

## Reproduce the frozen all-holdout COS run

```bash
for CFG in configs/holdouts/real_5g_nidd_*.yaml
do
  NAME=$(basename "$CFG" .yaml)
  echo "Running composite $NAME"

  python scripts/xmag_composite_open_score_diagnostic.py \
    --config "$CFG" \
    --out "runs/composite_${NAME}" \
    --top-m 1 2 \
    --k-values 1 \
    --gamma 0.25
done
```

Use `X-MAG-COS-24B` as the main method: `top_m=1`, `k=1`, and score `linear_max_resid_owner_b025__uncert_anomaly_b05_gamma_0.25`.

## Seed-stability sweep

Run the complete small stability experiment:

```bash
bash scripts/run_seed_stability_cos24.sh
python scripts/aggregate_seed_stability.py
```

This runs all eight 5G-NIDD holdouts for random states `7`, `42`, and `123`, then writes:

```text
results/tables/table_seed_stability_cos24_all_rows.csv
results/tables/table_seed_stability_cos24_by_attack.csv
results/tables/table_seed_stability_cos24_overall.csv
```

## Paper files

The current manuscript draft is in:

```text
manuscript/main.tex
```

It has been updated to frame the method as **X-MAG-COS: Composite Open-Set Scoring for Explanation-Aware Multi-Agent IDS** and now includes the three-seed stability analysis.

## Scientific caution

Do not report synthetic smoke-test numbers as paper results. Only tables produced from the real extracted `data/5G-NIDD/Encoded.csv` should be used. The seed-stability result is the current conservative result; single-seed tables should be described as method-selection or ablation evidence.