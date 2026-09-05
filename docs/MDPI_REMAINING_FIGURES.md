# Reproduce the MDPI Diagnostic Figures

The hard-holdout figure workflow is complete and remains checked in for reproducibility. It generates the per-flow figures for the difficult 5G-NIDD holdouts `UDPFlood` and `SlowrateDoS`, together with the communication frontier and per-family summary.

No shell heredoc is required.

## Required local state

Run from a workspace containing:

```text
data/5G-NIDD/Encoded.csv
runs/mdpi_r1/
configs/holdouts/
scripts/mdpi_revision_trial.py
scripts/mdpi_revision_common.py
```

The workspace does not need to be a Git checkout.

## Generate the complete figure set

```bash
bash scripts/generate_mdpi_remaining_figures.sh
```

The script reuses existing `score_components.npz` files. If one is missing, it reruns only that hard holdout and seed with heavy baselines disabled; it does not repeat the complete five-seed/all-holdout suite.

## Generated outputs

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

PNG counterparts are produced for visual inspection. Generated figures and archives remain local by policy; the supporting CSV/JSON data and plotting code are committed.
