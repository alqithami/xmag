# X-MAG-COS

X-MAG-COS is a reproducible software and derived-results repository for **communication-constrained, source-partitioned open-set intrusion detection** in 5G/IoT traffic. Source monitors encode class evidence, an attribution proxy, and anomaly evidence; a known-class head and a separately calibrated open-set head operate on the compact message.

## Repository scope

This repository contains the implementation, experiment configurations, tests, derived aggregate results, statistical analyses, and figure-generation code. The journal manuscript, response letters, publisher templates, and other publication assets are intentionally maintained outside this repository.

Raw third-party datasets and full local run arrays are also excluded. They can be regenerated from the public datasets and checked-in scripts.

## Confirmatory protocol

The current evidence uses one matched protocol across all principal methods and reproducible baselines:

- 5G-NIDD leave-one-attack-family-out evaluation over eight attack families;
- five seeds: `7, 21, 42, 84, 123`;
- forty matched seed-holdout trials per principal method;
- source ownership derived from leakage-excluded `sVid` metadata;
- split-conformal operation at a nominal 5% known-traffic false-positive level;
- the same train/validation/test construction for methods, baselines, and ablations.

The compact operating point is **X-MAG-COS-16Q**, a 16-byte binary16 evidence encoding. Across the forty matched trials it obtained:

```text
known macro-F1       0.998748 +/- 0.000294
unknown AUROC        0.902236 +/- 0.176277
unknown recall       0.723842 +/- 0.371420
mean known FPR       0.049728
maximum known FPR    0.051040
```

The 16-, 20-, and 24-byte full-evidence variants are detection-equivalent in this experiment. The repository therefore treats 16 bytes as the most compact evaluated full-content point and 24 bytes as the float32 reference; it does not claim that 24 bytes is uniquely optimal.

The results also establish important limits:

- coordinator uncertainty is not significantly inferior or superior to the composite head after multiplicity correction;
- the source-aware all-agent logit average is significantly weaker;
- centralized full-feature Random Forest remains a strong upper bound;
- UDPFlood remains a decisive open-set failure case;
- CICIoT2023 remains a transfer stress test rather than a generalization success;
- the attribution field is a lightweight proxy and is not equivalent to full SHAP attribution;
- targeted source-message manipulation requires authentication, trust weighting, or redundant observation.

## Hard-family diagnostics

The per-flow analysis is complete for UDPFlood and SlowrateDoS:

- SlowrateDoS retains useful ranking (`AUROC 0.8994 +/- 0.0185`) but only `0.3035 +/- 0.1162` recall at the validation 95th-percentile rule. Of 254,662 missed seed-trial decisions, 99.691% were assigned to HTTPFlood.
- UDPFlood is below random ranking on average (`AUROC 0.4533 +/- 0.0146`) and is almost never rejected (`0.0032 +/- 0.0044` recall). All 2,279,445 missed seed-trial decisions were assigned to Benign.

## Repository layout

```text
configs/                              Dataset and holdout configurations
scripts/mdpi_revision_*.py            Protocol-matched experiments and analyses
scripts/run_mdpi_round1_all.sh        Resumable five-seed revision suite
scripts/export_mdpi_hard_holdout_support.py
                                      Per-flow ROC/histogram/confusion exporter
scripts/plot_mdpi_manuscript_figures.py
                                      Publication-quality plotting from saved results
scripts/generate_mdpi_remaining_figures.sh
                                      One-command figure and archive workflow
scripts/verify_repository_complete.py Repository integrity and completeness check
results/mdpi_r1/                      Matched 5G-NIDD summary results and figure data
results/mdpi_r1_ciciot/               CICIoT2023 transfer-stress summaries
results/tables/                        Earlier documented experiment tables
tests/                                Smoke and regression tests
docs/                                 Reproduction and result interpretation
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-mdpi-r1.txt
```

The real 5G-NIDD encoded file is expected locally at:

```text
data/5G-NIDD/Encoded.csv
```

## Reproduce the matched suite

```bash
bash scripts/run_mdpi_round1_all.sh
```

The runner is resumable and skips completed trials containing `metrics.csv`.

## Reproduce the diagnostic figures

```bash
bash scripts/generate_mdpi_remaining_figures.sh
```

This command reuses existing per-flow score arrays and regenerates only missing UDPFlood or SlowrateDoS support.

## Verify repository completeness

```bash
python scripts/verify_repository_complete.py
pytest -q
```

The same checks run automatically on Python 3.11 and 3.12 through GitHub Actions.

## Data and artifact policy

Only code, configurations, derived summary metrics, and figure data are committed. Obtain 5G-NIDD and CICIoT2023 from their official distribution sources and follow their terms of use. Raw datasets, full per-flow arrays, generated archives, manuscript files, and publisher materials remain outside the repository.
