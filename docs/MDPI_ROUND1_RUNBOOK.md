# MDPI Round-1 Reproduction Guide

## Repository checkout and experiment workspace

A directory containing the datasets and completed runs does not have to be a Git checkout. Keep raw datasets in the existing experiment workspace and use a clean repository clone for versioned code when needed:

```bash
git clone https://github.com/alqithami/xmag.git xmag_github
```

The checked-in scripts can also be copied into an existing non-Git workspace. Do not run `git init` inside a dataset directory merely to obtain updates.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-mdpi-r1.txt
```

## Full protocol-matched suite

```bash
bash scripts/run_mdpi_round1_all.sh
```

The runner is resumable and skips trials that already contain `metrics.csv`.

## Hard-holdout figure support

Generate all UDPFlood and SlowrateDoS ROC, score-distribution, and accepted-unknown analyses with:

```bash
bash scripts/generate_mdpi_remaining_figures.sh
```

The workflow first reuses existing `score_components.npz` files. When a required array is absent, it reruns only that seed-holdout pair with the heavy comparison models disabled.

## Repository verification

```bash
python scripts/verify_repository_complete.py
pytest -q
```

## Data policy

Raw 5G-NIDD and CICIoT2023 files, complete per-flow arrays, generated ZIP archives, manuscripts, and publisher materials are not tracked. Only reproducible code, configurations, derived summary data, and documentation are committed.
