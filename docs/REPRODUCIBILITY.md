# Reproducing X-MAG-COS

This is the single protocol guide for the code and derived evidence accompanying the accepted *Mathematics* article. The default branch also retains underlying study traces and supporting results; use this guide rather than reassembling a workflow from earlier technical notes.

## 1. Requirements

Use **Python 3.12** and the pinned release dependency list. The original matched extension ran using Python 3.12.9 on macOS/Apple Silicon; fixed seeds and versions improve reproducibility but cannot guarantee bitwise-identical model fits on different hardware.

```bash
git clone https://github.com/alqithami/xmag.git
cd xmag
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduction.txt
```

Using a tagged release or fixed commit is preferable to using moving `main` for published experiments.

## 2. Validate the frozen evidence (no raw data)

```bash
python scripts/reproduce_paper.py verify
python -m pytest -q
```

The verification uses Python's standard library to confirm the 38-file supplemental archive, its 37 source/result checksums, completed task counts, reference metrics, holdout/source fingerprints, and consistency between the archive and browsable tables. Regression tests cover packet size, FPR ties, source-split handling, and numerical diagnostics.

Expected headline output: `X-MAG-COS publication evidence: PASS`. This is evidence-integrity verification, **not a rerun of model training**.

## 3. Smoke-test the pipeline (synthetic)

```bash
python scripts/reproduce_paper.py smoke
```

The smoke test creates a small local synthetic dataset and executes the original development pipeline. It does not reproduce any paper statistics or replace the real dataset.

## 4. Obtain the original real data

Download the official encoded 5G-NIDD export independently; instructions and rights are in [DATA.md](DATA.md). Put it at:

```text
data/5G-NIDD/Encoded.csv
```

The file must hash to **`7c238e2d5dabbc1afcd01c50db91372d6f6808f7572bb92baeb2df03a18af90c`**. No conversion or replacement by another 5G-NIDD file is automatic.

## 5. Full study execution

```bash
python scripts/reproduce_paper.py full
```

This workflow performs two dependent stages:

1. Evaluate each of eight 5G-NIDD attack-family holdouts across seeds **7, 21, 42, 84, 123** using the exact executed core source under `scripts/`. Per-flow/calibration score caches and models are stored locally in `runs/mdpi_r2/` (not committed). A complete matched run includes 40 tasks.
2. Reuse those score caches and run the final component/quantization experiment under `experiments/paper/`. It generates 1,320 nominal and 2,640 exact-matched-FPR rows (plus the retained-score analysis), with output under `results/publication_reproduction/`. This directory is ignored by Git to prevent accidental modification of frozen evidence.

The second stage preserves the original source-code fingerprints for the published study. **Do not modify the original `scripts/mdpi_revision_common.py`, `scripts/xmag_round2_core.py`, or `scripts/xmag_round2_run.py`**. The launcher verifies those digests and fails if they change.

The experiment can take substantial compute, memory, and disk space. Trial caches permit resumption on the same environment with the same code/data fingerprint. Do not mix output folders from different environment versions or config hashes. A fresh run is not guaranteed to produce byte-identical model artifacts on different platforms.

## 6. What the public archive can and cannot prove

| Procedure | Requires third-party raw data | What it establishes |
| --- | --- | --- |
| `verify` | No | Frozen derived-result integrity, checksums, table consistency |
| `pytest` | No | Software invariants and fixture tests |
| `smoke` | No | Basic pipeline plumbing on synthetic input |
| `full` | **Yes, exact data** | Fresh protocol-matched run; compare aggregate outcomes and disclose environment variance |

The retrospective fixed-FPR curves use test-set negatives to select cutoffs; those thresholds are **not** independently calibrated deployment operating points. The recorded single-corpus matched tasks do not constitute 40 independent deployments. The attribution field is a proxy, not a full SHAP explanation. Cross-dataset CICIoT2023 evidence remains an explicitly separate transfer-stress exercise in the stored supplementary evidence.

## 7. Find results and cite

- [Publication results, operating definitions, and limitations](RESULTS.md)
- [Data sourcing and dataset fingerprint](DATA.md)
- [Evidence provenance and archive](PROVENANCE.md)
- [Publication software citation](../CITATION.cff)

When reporting reproduced values, include the Git commit or tag, dataset checksum, Python/package versions, platform, and whether reported thresholds are retrospective or independently calibrated.
