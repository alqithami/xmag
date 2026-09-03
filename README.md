# X-MAG-COS

X-MAG-COS studies **communication-constrained, source-partitioned open-set intrusion detection** for 5G/IoT traffic. Each source monitor encodes compact class, attribution-proxy, and anomaly evidence. A known-class head and a separately calibrated open-set head operate on that evidence.

## Current confirmatory evidence

The MDPI *Mathematics* revision uses one protocol across all principal methods and reproducible baselines:

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

The 16-, 20-, and 24-byte full-evidence variants are detection-equivalent in this experiment. The repository therefore treats 16 bytes as the most compact evaluated full-content point and 24 bytes as the float32 diagnostic reference; it does not claim that 24 bytes is uniquely optimal.

The results also establish important limits:

- coordinator uncertainty is not significantly inferior or superior to the composite head after multiplicity correction;
- the source-aware all-agent logit average is significantly weaker;
- centralized full-feature Random Forest remains a strong upper bound;
- UDPFlood remains a decisive open-set failure case;
- CICIoT2023 remains a transfer stress test rather than a generalization success;
- the attribution field is a lightweight proxy and is not equivalent to full SHAP attribution;
- targeted source-message manipulation requires authentication, trust weighting, or redundant observation.

## Hard-family diagnostics

The reviewer-requested per-flow analysis is complete for UDPFlood and SlowrateDoS:

- SlowrateDoS retains useful ranking (`AUROC 0.8994 +/- 0.0185`) but only `0.3035 +/- 0.1162` recall at the validation 95th-percentile rule. Of 254,662 missed seed-trial decisions, 99.691% were assigned to HTTPFlood.
- UDPFlood is below random ranking on average (`AUROC 0.4533 +/- 0.0146`) and is almost never rejected (`0.0032 +/- 0.0044` recall). All 2,279,445 missed seed-trial decisions were assigned to Benign.

The manuscript distinguishes semantic absorption for SlowrateDoS from a more fundamental representation-and-rejection failure for UDPFlood.

## Repository layout

```text
configs/                              Holdout configurations
scripts/mdpi_revision_*.py            Protocol-matched revision experiments
scripts/run_mdpi_round1_all.sh        Resumable full revision suite
scripts/export_mdpi_hard_holdout_support.py
                                      Per-flow ROC/histogram/confusion exporter
scripts/plot_mdpi_manuscript_figures.py
                                      Publication-ready plotting from saved results
scripts/generate_mdpi_remaining_figures.sh
                                      One-command figure and archive workflow
results/mdpi_r1/                      Matched 5G-NIDD result tables and figure data
results/mdpi_r1_ciciot/               CICIoT2023 transfer-stress result tables
manuscript/                           Revised split LaTeX source and bibliography
docs/MDPI_ROUND1_FIGURE_RESULTS.md    Diagnostic interpretation and reproduction notes
```

Raw datasets and local run directories are intentionally not redistributed. Final binary figure PDFs and response-letter documents are contained in the author resubmission package and are reproducible from the committed code and derived results.

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

## Reproduce the matched revision suite

```bash
bash scripts/run_mdpi_round1_all.sh
```

The runner is resumable and skips completed trials containing `metrics.csv`.

## Reproduce the manuscript figures

```bash
bash scripts/generate_mdpi_remaining_figures.sh
```

This command reuses existing per-flow score arrays, regenerates only missing UDPFlood or SlowrateDoS raw support, and writes:

```text
results/mdpi_r1/manuscript_figures/
results/mdpi_r1/xmag_mdpi_remaining_figures.zip
```

## Data and result policy

Only code, configurations, derived summary metrics, figure data, and manuscript source are committed. Obtain 5G-NIDD and CICIoT2023 from their official distribution sources and follow their terms of use. Raw third-party datasets and full local run arrays remain outside the repository.
