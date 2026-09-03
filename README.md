# X-MAG-IDS

X-MAG-IDS studies **communication-constrained, source-partitioned open-set intrusion detection** for 5G/IoT traffic. Local source monitors encode compact class, attribution-proxy, and anomaly evidence; a known-class head and a separately calibrated open-set head operate on that evidence.

## Current confirmatory evidence

The current MDPI Mathematics revision uses one protocol across all methods:

- 5G-NIDD leave-one-attack-family-out evaluation over eight attack families;
- five seeds: `7, 21, 42, 84, 123`;
- forty matched seed–holdout trials per principal method;
- source ownership derived from leakage-excluded `sVid` metadata;
- split-conformal operation at a nominal 5% known-traffic false-positive level;
- the same training/validation/test construction for methods, baselines, and ablations.

The most compact full-evidence configuration is `X-MAG-COS-16Q`: a 16-byte message using binary16 evidence values. Across the forty matched trials it obtained:

```text
known macro-F1       0.998748 ± 0.000294
unknown AUROC        0.902236 ± 0.176277
unknown recall       0.723842 ± 0.371420
mean known FPR       0.049728
maximum known FPR    0.051040
```

The 16-, 20-, and 24-byte full-evidence variants are essentially equivalent in this experiment. The repository therefore treats 16 bytes as the compact operating point and 24 bytes as the float32 reference, rather than claiming that 24 bytes is uniquely optimal.

These results also establish important limits:

- coordinator uncertainty was not significantly inferior or superior to the composite head after multiplicity correction;
- the source-aware all-agent logit average was significantly weaker;
- centralized full-feature Random Forest remained a strong upper bound;
- UDPFlood remained a difficult open-set family;
- CICIoT2023 remained a stress test rather than a generalization success.

## Repository layout

```text
configs/                         Holdout configurations
scripts/mdpi_revision_*.py       Protocol-matched revision experiments
scripts/run_mdpi_round1_all.sh   Resumable full revision suite
scripts/export_mdpi_hard_holdout_support.py
                                 No-heredoc ROC/histogram/confusion exporter
results/mdpi_r1/                 Matched 5G-NIDD results and figures
results/mdpi_r1_ciciot/          CICIoT2023 stress-test results
results/archives/                Exact result/code archives
docs/MDPI_ROUND1_RUNBOOK.md      Reproduction and raw-support instructions
docs/MDPI_ROUND1_RESULTS.md      Result interpretation
```

Raw datasets are intentionally not redistributed.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-mdpi-r1.txt
```

The real 5G-NIDD encoded file is expected at:

```text
data/5G-NIDD/Encoded.csv
```

## Reproduce the full matched revision suite

```bash
bash scripts/run_mdpi_round1_all.sh
```

The runner is resumable and skips completed trials that already contain `metrics.csv`.

## Generate the remaining hard-holdout support

Do not use a shell heredoc. Run the checked-in exporter:

```bash
python scripts/export_mdpi_hard_holdout_support.py --runs-root runs/mdpi_r1
```

It creates:

```text
results/mdpi_r1/mdpi_hard_holdout_support.zip
```

If the local matched-run folders no longer contain `score_components.npz`, rerun only the two required seed-42 hard holdouts, then export:

```bash
python scripts/mdpi_revision_trial.py --config configs/holdouts/real_5g_nidd_udpflood.yaml --seed 42 --skip-heavy-baselines --out runs/mdpi_r1_raw/seed42/real_5g_nidd_udpflood
python scripts/mdpi_revision_trial.py --config configs/holdouts/real_5g_nidd_slowratedos.yaml --seed 42 --skip-heavy-baselines --out runs/mdpi_r1_raw/seed42/real_5g_nidd_slowratedos
python scripts/export_mdpi_hard_holdout_support.py --runs-root runs/mdpi_r1_raw
```

## Data and result policy

Only derived metrics, figures, configurations, and code are committed. Obtain 5G-NIDD and CICIoT2023 from their official distribution sources and follow their terms of use. The exact matched revision result archives are retained under `results/archives/`.
