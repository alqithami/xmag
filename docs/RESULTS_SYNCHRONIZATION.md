# Code and result synchronization

This update reconciles the figure-update package with the code-only repository.
It does not retrain models, tune scores, change thresholds, or replace measured results.
The existing README edit and the manuscript-exclusion policy are preserved.

## Changes

The aggregator now uses the intended composite score for X-MAG-COS message-size
comparisons and entropy for the full-feature RF comparison. Other score variants
remain available in `protocol_matched_summary.csv`; they are not extra budget points.
The eight-row `message_pareto.csv` is a comparison table, not a claim that all eight
points are Pareto-optimal. Separate AUROC and operational-recall figures are produced.

Per-family summaries use the compact 16Q composite variant, preserve native
5G-NIDD/CICIoT2023 labels, and record the actual seed count. Standard deviations
are blank for the single-seed CICIoT2023 experiment, not falsely reported as zero.
The official-folder sample counts supplied by the author are retained separately.

Duplicate trial keys, unmatched method coverage, mixed scenarios, and invalid
probability-valued detection metrics now stop aggregation instead of silently
changing the reported protocol. `--expected-trials` optionally enforces the exact
number of seed-holdout pairs. CI installs the revision dependencies and exercises
the aggregation and plotting path as well as the existing regression tests.

## Validation and provenance

The supplied primary archive contains 640 metric rows: 16 method/score/budget groups
with 40 matched trials each. The secondary archive contains 112 rows: 16 groups
with seven category holdouts and seed 42. Recomputed principal summaries agree
with both supplied summaries to absolute/relative tolerance 1e-12.
No previously reported experimental value was revised by this synchronization.

`results/RESULT_SYNC_MANIFEST.json` records input archive checksums, output
checksums, group counts, selection rules, and the source commit. Raw datasets,
per-flow arrays, manuscripts, and reviewer replies are not uploaded.

## Re-aggregate existing results without retraining

```bash
python scripts/mdpi_revision_aggregate.py --metrics-csv results/mdpi_r1/all_metrics.csv --expected-trials 40 --outdir results/mdpi_r1
python scripts/mdpi_revision_aggregate.py --metrics-csv results/mdpi_r1_ciciot/all_metrics.csv --expected-trials 7 --outdir results/mdpi_r1_ciciot
```

These commands require the `all_metrics.csv` files from the retained experiment
archives or local runs. Those full logs remain local under the existing repository
policy. The original `--runs-root` interface still works. Re-running aggregation
into a separate output directory is recommended when auditing committed snapshots.
