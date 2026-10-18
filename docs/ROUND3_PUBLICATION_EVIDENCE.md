# Round-3 publication evidence for X-MAG-COS
Publication-stage evidence synchronized 18 October 2026; experiment completion recorded 1 October 2026. This record accompanies the accepted *Mathematics* manuscript, “X-MAG-COS: Compact Evidence Messaging for Source-Partitioned Open-Set Intrusion Detection in 5G-IoT Networks” (manuscript mathematics-4541032).

## Scope and provenance

- Prespecified fresh matched extension: **8 leave-one-family-out tasks × 5 seeds = 40 task instances**, with **11 evaluation conditions**, **3 targeted FPR levels (0.1%, 1%, 5%)**, and **two negative-population definitions (all-known and Benign-only)**.
- **1,320 nominal-operating rows**, **2,640 fixed-FPR rows**, **240 quantization-diagnostic rows**, and a **separate 720-row** matched fixed-FPR readout from retained Round-2 scores. Historical rows are not pooled as independent fresh runs.
- The primary retrospective comparisons match **identical integer false-positive counts per paired task**; boundary randomization is separately represented as an expected-rate companion, not an observed operating point.
- The full 38-file research-evidence archive is at [`artifacts/xmag_r3_complete_evidence.zip`](../artifacts/xmag_r3_complete_evidence.zip), **SHA-256 `0f94e7f057c9d5914b8251add19a98228c0368bda2bb621e888b7be8aa912c92`**. It preserves the 37 original evidence files plus the hash manifest; the files and hashes were checked against that manifest.
- Archived methods/code (also available in browsable form) are under [`experiments/reviewer_r3/`](../experiments/reviewer_r3/). The local source snapshots inside the archive preserve the exact executed Round-2 dependencies.
- Raw third-party traffic, saved model checkpoints and per-flow score arrays are **not** distributed. Re-running training requires separately obtained authorized datasets and the state/data assumptions described in the experiment README.

## Primary fixed-FPR comparison (all-known denominator)

Mean unknown recall across 40 paired task instances:

| Method | Target 0.1% FPR | Target 1% FPR | Target 5% FPR |
| --- | ---: | ---: | ---: |
| X-MAG-COS 16Q reference | 0.0934 | 0.5484 | 0.7242 |
| Centralized full-feature Random Forest | 0.6781 | 0.7557 | 0.7942 |
| Local Extra Trees substitution | 0.0336 | 0.5238 | 0.8062 |
| Arithmetic-mean fusion substitution | 0.2482 | 0.4492 | 0.5724 |

Source: [`results/mdpi_r3/fixed_fpr_summary.csv`](../results/mdpi_r3/fixed_fpr_summary.csv). Extra Trees is strongest among these rows at 5% but performs worse than the baseline at more stringent levels; none of these rows implies universal superiority. The test-ROC cutoff is retrospective rather than a deployment-calibrated threshold.

## Other key observations

- Earlier corrected 5G-NIDD 16Q UDPFlood mean AUROC: **0.481810**, with mean uncertainty about **0.0190** across five seeds, reported in the retained Round-2 audit. Older Round-1 diagnostic values (0.4533) are a distinct historical protocol and must not be cited as the corrected accepted-paper result.
- Quantization checks report zero observed numerical bound violations under the stated **1e-8 tolerance**. This is **not** an independent machine-checked mathematical proof.
- The 16-byte record includes an optional attribution proxy. Evidence does not establish analyst-facing explainability; the 12-byte detection-only alternative remains relevant.
- CICIoT2023 is evaluated separately as a cross-dataset stress test in the earlier work; it is not a generalization success claim.
- Source ownership is based on two populated metadata partitions; it does **not** represent eight functioning physical sensor agents.

## Files to inspect

- [`results/mdpi_r3/RUN_COMPLETENESS.json`](../results/mdpi_r3/RUN_COMPLETENESS.json) — completion counts and exact-FPR checks.
- [`results/mdpi_r3/PLAN.json`](../results/mdpi_r3/PLAN.json) — prespecified protocol, data fingerprint, methods and environment.
- [`results/mdpi_r3/fixed_fpr_summary.csv`](../results/mdpi_r3/fixed_fpr_summary.csv) — primary comparisons.
- [`results/mdpi_r3/fixed_fpr_by_family.csv`](../results/mdpi_r3/fixed_fpr_by_family.csv) — task-specific comparisons.
- [`results/mdpi_r3/paired_fixed_fpr_statistics.csv`](../results/mdpi_r3/paired_fixed_fpr_statistics.csv) — paired uncertainty and multiplicity adjustments.
- [`results/mdpi_r3/quantization_diagnostics.csv`](../results/mdpi_r3/quantization_diagnostics.csv) — numerical checks.
- [`results/mdpi_r3/round2_control_reconciliation.csv`](../results/mdpi_r3/round2_control_reconciliation.csv) — differences between retained Round-2 controls and fresh refits.
- [`results/mdpi_r3/MANIFEST.json`](../results/mdpi_r3/MANIFEST.json) — SHA-256 inventory for all research files in the complete archive.

## Verification

The source archive passed manifest SHA-256 validation for all 37 entries; the code passed **23 targeted tests on the original Mac environment (18 October 2026)** with a single NumPy overflow warning in an overflow-handling regression test. For the repository's browsable source:

```bash
python -m pip install -r experiments/reviewer_r3/requirements.txt
python -m pytest -q experiments/reviewer_r3/test_actual_fpr.py experiments/reviewer_r3/test_extension.py experiments/reviewer_r3/test_snapshot_protocol.py
```

These are software-integrity checks, not independent replicated scientific trials. For data-dependent reproduction and caveats consult the archive's `results/mdpi_r3/code/README.md`, `PLAN.json`, and `INTERPRETATION.md`.

## Historical record

Older summary metrics in `docs/RESULTS_STATUS.md`, `docs/MDPI_ROUND1_RESULTS.md`, and `results/mdpi_r1/` reflect earlier protocols. They remain available for provenance, but the latest manuscript should be compared against the Round-3 extension plus the explicitly separated corrected Round-2 evidence described here.
