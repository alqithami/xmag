# MDPI Mathematics Round-1 Reproduction and Raw-Support Guide

## Why the terminal displayed `heredoc>`

`heredoc>` is not model execution. It means `zsh` received a command beginning with a here-document such as `<<'PY'` but did not receive the exact closing line `PY`. Press **Control+C** once to cancel it. The repository now contains a normal Python file, so no here-document is required.

## A. Update the local checkout

```bash
cd /Users/alqithami/Desktop/2026/July/xmag/xmag_repo
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements-mdpi-r1.txt
```

## B. Try the no-rerun export first

If the completed matched-run folders still contain `score_components.npz`, this command generates the requested ROC curves, known-versus-unknown histograms, accepted-unknown confusion analysis, metadata manifest, and one upload archive:

```bash
python scripts/export_mdpi_hard_holdout_support.py --runs-root runs/mdpi_r1
```

Success ends with:

```text
Share this file:
results/mdpi_r1/mdpi_hard_holdout_support.zip
```

Upload that single ZIP.

## C. Only if the exporter reports that score arrays are absent

Rerun the two hard holdouts without the heavy comparison models:

```bash
python scripts/mdpi_revision_trial.py --config configs/holdouts/real_5g_nidd_udpflood.yaml --seed 42 --skip-heavy-baselines --out runs/mdpi_r1_raw/seed42/real_5g_nidd_udpflood
```

```bash
python scripts/mdpi_revision_trial.py --config configs/holdouts/real_5g_nidd_slowratedos.yaml --seed 42 --skip-heavy-baselines --out runs/mdpi_r1_raw/seed42/real_5g_nidd_slowratedos
```

Then export:

```bash
python scripts/export_mdpi_hard_holdout_support.py --runs-root runs/mdpi_r1_raw
```

Upload:

```text
results/mdpi_r1/mdpi_hard_holdout_support.zip
```

No other experiment is currently required for the hard-holdout figures.

## D. Full matched suite

```bash
bash scripts/run_mdpi_round1_all.sh
```

The runner is resumable and skips any trial that already has `metrics.csv`.

## E. Author-supplied administrative information

The verified NCA grant/award number is still required if the award has one. Do not leave `CRPG-00-0000` in the revised manuscript. If the award formally has no number, state the funding body without inventing one.
