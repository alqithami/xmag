# Reviewer extension: exact FPR, component replacements, and quantization

Use **`start.py`**, not the lower-level `run.py`, for the supplied experiment workspace. This launcher binds the extension to the exact Round-2 source snapshot and automatically reuses its retained scores before fitting replacement models. No scientific results are fabricated or supplied with this code release.

## Existing Mac workspace

```text
/Users/alqithami/Desktop/2026/July/xmag/xmag_repo
```

Keep the existing `.venv`. No package upgrade is required for the supplied Python 3.12.9 / NumPy 2.5.0 / pandas 3.0.3 / scikit-learn 1.9.0 environment. The requirements file is for new installations and CI, not an instruction to upgrade an active experiment environment.

Install this directory as `reviewer_r3` in that workspace, then run:

```bash
caffeinate -i python -u reviewer_r3/start.py --root "$PWD"
```

The workspace need not be a Git checkout. No editable installation, Git initialization, heredoc, or new dataset download is required. Do not update code or packages after starting the resumable run.

## What the snapshot established

The supplied source inventory contains 40 `round2_scores.npz` files but no fitted model archives under `runs/mdpi_r2`. The score caches contain the three composite layouts (12B, 16Q, 24B), their calibration scores, and the ordered known labels. They do not retain all alternative score components or centralized-model scores.

The launcher therefore performs two distinct stages:

1. Read and validate all 40 retained score archives. Compute retrospective fixed-FPR results for the three original composite layouts **without training**. Check their known-label order against the original split. Save these measurements separately with `retained_round2_` prefixes.
2. Fit shared reference models once per task for the algorithm replacements, missing controls, and fixed-model precision diagnostic. Cache the new models and scores. Do not relabel these deterministic refits as identical historical fits; report an explicit reconciliation with old scalar results.

The real dataset must remain at `data/5G-NIDD/Encoded.csv`, SHA-256 `7c238e2d5dabbc1afcd01c50db91372d6f6808f7572bb92baeb2df03a18af90c`. The launcher validates the actual source files, source provenance, and existing holdout configurations before expensive work. It does not overwrite or repair those files silently.

## Exact integration points

The real-data preparer is imported directly from the verified local `scripts/xmag_round2_run.py`. Its original train/test row identities are preserved. Validation is halved using **`default_rng(seed+9017).permutation(n_validation)`**, not a newly stratified split. The adapter also preserves the executed class `argmax`, the attribution proxy's intermediate float32 storage, float64 prototype accumulation, and zero-probability entropy convention.

One disclosed reproducibility change is an explicit logistic seed `seed+500` in place of the former `random_state=None`. Source prediction uses one worker to stabilize accumulation; fitting defaults to four workers. The new controls are internally matched and their differences from the retained results are reported, not forced to zero.

`snapshot_protocol.py` selects these audited integration functions before the lower-level run loop executes. It never edits the original Round-2 source.

## Prespecified experiment

Eight held-out families times seeds **7, 21, 42, 84, 123**. All alternatives use identical training/normalization/calibration/test observations and a 16-byte full-content message, except explicitly named 12B/24B/full-feature controls. Unknown test labels never select algorithms, mixture weights, or hyperparameters.

| Replaced block | Reference | Alternative |
| --- | --- | --- |
| Local classifier and model-dependent feature importance | Random Forest | Extra Trees, 30 trees |
| Local anomaly detector | Isolation Forest | Eight-component diagonal Gaussian mixture |
| Coordinator classifier | One-versus-rest logistic | Extra Trees, 50 trees, depth 12, minimum leaf 5 |
| Prototype distance | Standardized RMS/L2 | Standardized mean absolute/L1 |
| Fusion rule | Composite | Arithmetic mean of the same normalized components |
| Fusion rule | Composite | Maximum of the same normalized components |

GMM uses at most 16,384 training-only rows per source; the cap is fixed before evaluation. It does not use test data or pretend to be a full-data fit. Convergence failure is reported as failure, not silently replaced by another algorithm or recorded as success.

Controls are the reference 16Q, 12B, 24B, uncertainty-only 16Q, and centralized full-feature RF. The extension has **40 tasks, 11 method/score conditions per task, 1,320 nominal rows, and 2,640 fixed-FPR rows**, plus a separate 720-row reanalysis of the exact retained layouts. These are repeated matched evaluations, not 440 independent datasets. The peer, source-occupancy, SHAP and previous full baseline battery are not repeated.

## Same actual FPR

The primary retrospective comparison permits exactly `floor(target*n_negative)` false positives for each method in a seed/holdout/denominator. Score is primary; ties use a predeclared pseudorandom row-position ordering, shared across methods. The tie seed is fixed at 20260930.

The achieved fraction is `floor(target*n_negative)/n_negative` and is checked to be identical across methods. Finite samples cannot represent every requested percentage exactly; actual false-positive counts and denominators are exported. Paired tests use recall at these matched **realized** rates.

Companion columns retain ordinary strict-score and inclusive-score endpoints and exact-target **expected** FPR/recall under boundary randomization. Expected values are not described as observed deterministic rates. Both all-known and Benign-only denominators are reported.

These are retrospective test-ROC comparisons. Negative test scores determine their boundaries; unknown scores do not. They are not deployment thresholds fitted without test data. Independent known-calibration operating results remain separate. Benign open-set rejection is only one part of the total IDS false-alert rate.

## Quantization and theory support

The diagnostic freezes the float32-trained logistic head, prototypes, and normalizers, preserves integer identifiers, and quantizes the calibration and test scalar fields. It propagates input-rounding intervals through sigmoid outputs, normalized OVR probabilities, possible predicted prototype classes, and monotone score fusion. It checks analytic score bounds for **both calibration and test**, the calibration order-statistic shift, and a score-margin upper bound on changed rejection decisions.

The fixed diagnostic operator is evaluated in float64 with a reported numerical tolerance. This is not a directed-rounding machine proof, an assertion of mathematical novelty, or a pure-quantization interpretation of different-content/separately refitted methods. The manuscript's theoretical argument still needs to be written and checked against the returned measurements.

## Progress and resume

In another terminal, from the same workspace and environment:

```bash
python reviewer_r3/start.py --root "$PWD" --status
```

After interruption, run the original `caffeinate` command again. Completed fingerprint-matched tasks are skipped; partial tasks reuse cached models and completed conditions. Do not delete output directories or reinstall the extension to resume. Changed code, data, configurations, or package versions are refused rather than mixed.

All new outputs are isolated in `runs/mdpi_r3` and `results/mdpi_r3`. At least 10 GiB free disk is required. Keep the Mac connected to power, lid open, and terminal open. `caffeinate` prevents idle sleep, not shutdown or every possible interruption. Fixture timing is not a full-data runtime prediction.

## Completion and files to return

A successful full run creates:

```text
results/mdpi_r3_review_text.txt
results/mdpi_r3_reviewer_results.zip
```

Upload both. The text exposes the tables without ZIP extraction; the ZIP contains the analysis, code, source snapshots, checksums, and provenance. Raw traffic, fitted models, and per-flow arrays remain local. They are not included in the sharing archive. The launcher and worker preserve failure tracebacks under `results/mdpi_r3`.

Software tests cover same-realized-FPR counts, scalar ties, expected-FPR accounting, conformal order statistics, exact snapshot splitting/rounding, packet validity, all replacements, analytic quantization checks, packaging, and resume. Integration uses small software fixtures, never invented 5G-NIDD measurements.

Official APIs:
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
- https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
