# Targeted reviewer extension: replacements, fixed FPR, and quantization

This isolated extension addresses the three further requests: algorithmic component replaceability, recall at a common empirical false-positive operating point, and numerical evidence for a message-quantization/decision-stability argument. No scientific measurements are included in this code release.

Run from the existing Mac experiment workspace. It need not be a Git checkout. The full real CSV must remain at `data/5G-NIDD/Encoded.csv`; the expected SHA-256 is `7c238e2d5dabbc1afcd01c50db91372d6f6808f7572bb92baeb2df03a18af90c`.

## Installation and run

With the existing virtual environment active:

```bash
python -m pip install -r experiments/reviewer_r3/requirements.txt
caffeinate -i python -u experiments/reviewer_r3/run.py --root "$PWD"
```

The script path can be absolute when this directory has been downloaded separately. No editable package installation, Git initialization, heredoc, or dataset download is required. Dependencies already satisfying the requirements are not explicitly upgraded. Do not change the environment after beginning a resumable run.

## Scope

Eight leave-one-family-out tasks times seeds 7, 21, 42, 84, and 123. All methods in this extension use the same explicitly constructed 56/7/7/30 known-data split, training-only preprocessing, metadata ownership, and receiver-decoded packets. Models and score normalization exclude calibration and test rows. Shared local models and fields are reused within each task.

The 16Q reference has six one-component-block replacements:

1. Local RF replaced by Extra Trees, including the importance proxy that belongs to the replacement model.
2. Local Isolation Forest replaced by an eight-component diagonal Gaussian mixture fitted on at most 16,384 training-only rows per source.
3. Logistic coordinator replaced by a bounded 50-tree Extra Trees coordinator (depth 12, minimum leaf five).
4. Standardized RMS prototype residual replaced by standardized mean absolute residual, retaining prototype centers.
5. Composite evidence fusion replaced by the mean of the same three normalized components.
6. Composite fusion replaced by their maximum.

Additional controls are 12B, 24B, uncertainty-only on 16Q, and centralized RF. The full scope is 40 tasks, 11 method/score conditions each, 1,320 calibration-based operating rows, and 2,640 fixed-FPR rows. This is not 440 independent datasets or 440 separate local-model fits.

The extension refits shared reference components because exact historical model caches have not been verified. It preserves and reconciles historical controls rather than assuming numerical equivalence or mixing old and new measurements. It does not rerun the previous peer, ownership, SHAP, or complete baseline battery. Any discrepancy with Round-2 is exported in `round2_control_reconciliation.csv` and must be inspected before manuscript integration. The full fitted-model set and exact score arrays remain locally cached for later analysis.

## What 'same FPR' means here

At targets 0.1%, 1%, and 5%, the code identifies an empirical negative-score boundary. It reports the deterministic strict and inclusive FPR/recall endpoints. Where ties prevent an exact deterministic target, it reports boundary randomization and the resulting **expected** FPR and recall. The expected empirical FPR equals the requested target to numerical tolerance. This is not mislabeled as an observed deterministic rate. Test labels are used only for this retrospective ROC comparison, not to fit deployed thresholds, select models, or tune weights.

Both all-known and Benign-only negative populations are evaluated. Independent-calibration nominal operating results remain a separate table. Benign open-set rejection is not the entire IDS false-alert rate.

## Quantization analysis

The pure precision diagnostic freezes a float32-trained logistic head, prototypes, and normalization. The same integer fields are retained; calibration and test scalar fields are rounded through binary16 serialization. Sigmoid and normalized-OVR interval bounds, possible prototype classes, and monotone fusion provide a score perturbation bound. Calibration-order-statistic movement and decision-margin coverage are measured. The diagnostic operator is evaluated in float64 and checked with a stated numerical tolerance, not claimed as a directed-rounding machine proof. This does not prove the novelty of a theorem or compare different-content records by a pure quantization argument.

## Progress, restart, and package

```bash
python experiments/reviewer_r3/run.py --root "$PWD" --status
```

Restart the original run command after interruption; completed fingerprint-matched tasks are skipped and partially completed tasks reuse cached fitted components. Do not delete `runs/mdpi_r3` to resume. Four training workers are the default; prediction accumulation is single-worker for reproducibility.

After completion, share:

```text
results/mdpi_r3_review_text.txt
results/mdpi_r3_reviewer_results.zip
```

The text contains every new summary and table and is readable without archive extraction. The ZIP contains results, code, provenance, checksums, and the local Round-2 script snapshots when present. It excludes the raw dataset, local model caches, and per-flow score arrays.

Outputs are isolated in `runs/mdpi_r3` and `results/mdpi_r3`. At least 10 GiB free disk is required. Keep the computer awake and connected to power; closing its lid or terminating the terminal may interrupt a foreground run. No credible total-duration estimate is assumed before full-data timings exist.

## Software validation

`test_extension.py` exercises tied-FPR boundaries, exact expected-FPR accounting, deterministic brackets, conservative conformal cutoffs, packet serialization and overflow checks, monotone scores, training-only screening, all substitutions, quantization diagnostics, completeness, packaging, and resume behavior. Integration uses a labeled software fixture, never reported as 5G-NIDD evidence.

Implementation references:
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
- https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
