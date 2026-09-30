# Targeted reviewer extension: replacements, identical FPR, and quantization

This isolated extension implements the remaining experimental requests: core-algorithm replaceability, recall at identical realized empirical false-positive rates, and numerical support for a message-quantization/decision-stability argument. No scientific results are fabricated or included in this code release.

## Run in the existing Mac workspace

The workspace need not be a Git checkout. Keep the full dataset at `data/5G-NIDD/Encoded.csv`. The required SHA-256 is `7c238e2d5dabbc1afcd01c50db91372d6f6808f7572bb92baeb2df03a18af90c`.

With the existing virtual environment active and this directory installed as `reviewer_r3`:

```bash
python -m pip install -r reviewer_r3/requirements.txt
caffeinate -i python -u reviewer_r3/run.py --root "$PWD"
```

A full repository clone may instead use `experiments/reviewer_r3/run.py`. No editable installation, Git initialization, heredoc, or new dataset download is required. Do not change packages or scripts after starting the resumable experiment. Already installed requirements are not explicitly upgraded.

## Prespecified experiment

Eight leave-one-attack-family-out tasks times seeds 7, 21, 42, 84, and 123. Every method uses the same explicit 56/7/7/30 known-data split, training-only preprocessing, metadata ownership, and receiver-decoded evidence. Calibration and test rows never fit a detector or normalizer. Shared source models and evidence are reused within each task.

Six replacements change one named component block in the 16Q pipeline:

| Block | Reference | Replacement |
| --- | --- | --- |
| Local classifier and its model-dependent importance | Random Forest | Extra Trees |
| Local anomaly detector | Isolation Forest | Eight-component diagonal Gaussian mixture |
| Coordinator classifier | One-versus-rest logistic | Bounded 50-tree Extra Trees |
| Prototype distance | Standardized RMS/L2 | Standardized mean absolute/L1 |
| Fusion | Composite rule | Arithmetic mean of the same normalized components |
| Fusion | Composite rule | Maximum of the same normalized components |

GMM uses at most 16,384 training-only rows per source, prespecified before test evaluation. The coordinator replacement uses depth 12 and minimum leaf five. Failed convergence or invalid numerical output is an error, not a successful measurement.

The additional controls are 12B, 24B, uncertainty-only on 16Q, and centralized full-feature RF. The full scope is **40 tasks, 11 method/score conditions each, 1,320 nominal operating rows, and 2,640 fixed-FPR rows**. This does not mean 440 independent datasets or 440 separate source-model fits.

Reference models are fitted once per task and cached in this new extension. Exact historical model caches were not available for verification, so the extension does not assume their compatibility. It automatically compares regenerated controls with available Round-2 scalar results and reports discrepancies; old observations are never silently overwritten or relabeled. The validation-halving rule and full environment are recorded. Use internally matched extension comparisons rather than mixing protocols. The old peer, ownership, SHAP, and full baseline battery are not rerun.

## Same actual FPR, not merely the same nominal setting

The primary retrospective comparison allows exactly `floor(target * n_negative)` false positives for every method in each seed/holdout/denominator. Score is the primary ordering; ties are ordered by a fixed pseudorandom row-position key, shared across methods. The tie seed is fixed at 20260930 and is not selected from results.

The actual empirical rate is `floor(target*n_negative)/n_negative`; it is identical across methods. A finite sample cannot always represent exactly 0.1%, 1%, or 5%, so the precise count and achieved fraction are exported. The suite checks that the between-method actual-FPR spread is zero. Paired tests use recall at these matched **realized** rates.

Companion columns retain strict-score and inclusive-score threshold endpoints, plus exact-target **expected** FPR/recall under boundary randomization. Those interpolated expectations are not mislabeled as observed deterministic rates. The complete tie information is available for sensitivity analysis.

All-known and Benign-only negative populations are reported separately. These ROC boundaries use the designated negative test population and are retrospective comparisons, not independently selected deployment thresholds. Unknown scores do not select boundaries or weights. Independent known-calibration operating results remain in separate tables. Benign open-set rejection is not the entire IDS false-alert rate.

## Quantization diagnostic

The pure precision experiment freezes the float32-trained logistic head, prototypes, and normalizers. It preserves integer identifiers and quantizes calibration/test scalar fields to binary16. Sigmoid and normalized-OVR intervals, possible predicted prototype classes, and monotone fusion bound score perturbations. The experiment checks calibration order-statistic movement and score-margin coverage of rejection changes.

The diagnostic numerical operator is evaluated in float64 with an explicit numerical tolerance, not a directed-rounding machine proof. It does not establish a theorem's novelty or supply a pure-quantization interpretation of different-content messages or separately refitted heads. The mathematical statement and proof still require manuscript integration after results are inspected.

## Progress, interruption, and completion

In another terminal:

```bash
python reviewer_r3/run.py --root "$PWD" --status
```

If interrupted, run the same original command again. Completed fingerprint-matched tasks are skipped; partial tasks reuse their cached models and completed conditions. Do not delete the output directories to resume. Code, data, configuration, and package-version changes are refused rather than mixed.

Outputs are isolated in `runs/mdpi_r3` and `results/mdpi_r3`. Fitted models and full score arrays remain locally cached. At least 10 GiB free disk is required. Keep the machine connected to power, the lid open, and the terminal open. The command prevents idle sleep, not every possible interruption. No full-data runtime estimate is claimed from fixture tests.

At completion upload these two files:

```text
results/mdpi_r3_review_text.txt
results/mdpi_r3_reviewer_results.zip
```

The text includes all new tables and metadata without requiring ZIP extraction. The ZIP contains code, scalar results, checksums, provenance, and local Round-2 script snapshots when available. Neither output contains the original dataset, fitted models, or per-flow arrays. A failure traceback is saved as `results/mdpi_r3/failure_traceback.txt`.

## Validation

Six actual-FPR tests check equal realized false-positive counts, finite-sample targets, score ties, label-independent boundary choice, and deterministic repeatability. Ten further tests cover expected-FPR accounting, conformal cutoffs, packet integrity, overflow rejection, score monotonicity, training-only screening, all substitutions, quantization diagnostics, full packaging, and resumability. Integration uses a labeled software fixture; it is never reported as 5G-NIDD evidence.

Official API references:
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
- https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
