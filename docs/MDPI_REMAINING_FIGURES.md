# Generate the Remaining MDPI Revision Figures

This workflow generates the figures that require per-flow score arrays for the two difficult 5G-NIDD holdouts: `UDPFlood` and `SlowrateDoS`. It also regenerates the communication frontier and per-family summary figure from the matched five-seed experiment.

No shell heredoc is required.

## Required local state

Run from the experiment workspace that contains:

```text
data/5G-NIDD/Encoded.csv
runs/mdpi_r1/
configs/holdouts/
scripts/mdpi_revision_trial.py
scripts/mdpi_revision_common.py
```

The workspace does not need to be a Git checkout.

## Download the latest figure scripts

```bash
cd /Users/alqithami/Desktop/2026/July/xmag/xmag_repo
source .venv/bin/activate
mkdir -p scripts
curl -fL https://raw.githubusercontent.com/alqithami/xmag/main/scripts/export_mdpi_hard_holdout_support.py -o scripts/export_mdpi_hard_holdout_support.py
curl -fL https://raw.githubusercontent.com/alqithami/xmag/main/scripts/run_mdpi_hardcase_raw.sh -o scripts/run_mdpi_hardcase_raw.sh
curl -fL https://raw.githubusercontent.com/alqithami/xmag/main/scripts/generate_mdpi_remaining_figures.sh -o scripts/generate_mdpi_remaining_figures.sh
chmod +x scripts/run_mdpi_hardcase_raw.sh scripts/generate_mdpi_remaining_figures.sh
```

## Generate every remaining figure

```bash
bash scripts/generate_mdpi_remaining_figures.sh
```

The script first reuses existing `score_components.npz` files. If one is missing, it reruns only that hard holdout and seed with heavy baselines disabled. It does not repeat the complete five-seed/all-holdout suite.

## Expected figures

```text
results/mdpi_r1/manuscript_figures/main/pareto_bytes_auroc.pdf
results/mdpi_r1/manuscript_figures/main/per_family_open_set.pdf
results/mdpi_r1/manuscript_figures/hard_holdouts/roc_udpflood.pdf
results/mdpi_r1/manuscript_figures/hard_holdouts/roc_slowratedos.pdf
results/mdpi_r1/manuscript_figures/hard_holdouts/score_histogram_udpflood.pdf
results/mdpi_r1/manuscript_figures/hard_holdouts/score_histogram_slowratedos.pdf
results/mdpi_r1/manuscript_figures/hard_holdouts/accepted_unknown_confusion_udpflood.pdf
results/mdpi_r1/manuscript_figures/hard_holdouts/accepted_unknown_confusion_slowratedos.pdf
```

PNG counterparts are generated for visual inspection.

## Final archive to share

```text
results/mdpi_r1/xmag_mdpi_remaining_figures.zip
```

Confirm it with:

```bash
ls -lh results/mdpi_r1/xmag_mdpi_remaining_figures.zip
```

Open the figure directory on macOS with:

```bash
open results/mdpi_r1/manuscript_figures
```

## Failure recovery

If the command stops, run the same command again:

```bash
bash scripts/generate_mdpi_remaining_figures.sh
```

Existing hard-holdout score arrays are reused. Only missing raw support is regenerated.
