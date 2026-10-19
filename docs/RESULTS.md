# Reported measurements and interpretation

This document identifies the frozen derived evidence accompanying the accepted *Mathematics* article, **X-MAG-COS: Compact Evidence Messaging for Source-Partitioned Open-Set Intrusion Detection in 5G-IoT Networks** (Alliheedi and Alqithami, 2026).

## Study scope

- Primary corpus: 5G-NIDD.
- Leave-one-attack-family-out: **8 families × 5 seeds = 40 matched tasks**; seeds `7, 21, 42, 84, 123`.
- **11 method/score conditions** per task; **1,320 nominal rows**, **2,640 fixed-FPR rows** (three FPR levels × two negative populations), and **240 numerical quantization-diagnostic rows** in the fresh matched extension.
- A separate **720-row fixed-FPR analysis of original saved scores** is documented and must not be combined with the new model refits as if they were one identical experiment.
- Protocol, hashes and environment: [`results/mdpi_r3/PLAN.json`](../results/mdpi_r3/PLAN.json), [`RUN_COMPLETENESS.json`](../results/mdpi_r3/RUN_COMPLETENESS.json), and [`MANIFEST.json`](../results/mdpi_r3/MANIFEST.json).

## Unknown-attack recall at matched observed false-positive counts

Means across the 40 task instances, **all-known denominator**. The exact false-positive count per paired task equals `floor(target × n_negative)`, with prespecified tie handling.

| Condition | 0.1% target FPR | 1% target FPR | 5% target FPR |
| --- | ---: | ---: | ---: |
| X-MAG-COS 16Q reference | 0.0934 | 0.5484 | 0.7242 |
| Centralized full-feature Random Forest | 0.6781 | 0.7557 | 0.7942 |
| Local Extra Trees replacement | 0.0336 | 0.5238 | 0.8062 |
| Arithmetic-mean score fusion | 0.2482 | 0.4492 | 0.5724 |

**Source:** [`fixed_fpr_summary.csv`](../results/mdpi_r3/fixed_fpr_summary.csv) and [`fixed_fpr_by_family.csv`](../results/mdpi_r3/fixed_fpr_by_family.csv). The reported levels represent *targets*; realized finite-sample FPRs match between paired methods but can differ infinitesimally from the exact percentages. These are **retrospective test-ROC operating points**, not thresholds calibrated for deployment. For independent known-calibration operating points, use [`nominal_summary.csv`](../results/mdpi_r3/nominal_summary.csv).

The component replacements are **not uniformly better** than the reference. Some differences are nonsignificant after multiplicity correction; the appropriate paired statistics and family-block uncertainty are in [`paired_fixed_fpr_statistics.csv`](../results/mdpi_r3/paired_fixed_fpr_statistics.csv).

## Notable limitations

- **UDPFlood:** corrected historical 16Q AUROC is 0.481810 (below chance on average), with four of five seeds below 0.5. The 2026-09 corrected analysis supersedes earlier differently defined pilot diagnostics. This is not a success case.
- **Source ownership:** two occupied metadata-derived partitions, rather than eight independently communicating real sensor devices.
- **Communication:** the 16-byte full-evidence 16Q encoding is a concrete operating point, not evidence of a universal optimal message budget. The 12-byte detection-only alternative is also documented.
- **Attribution:** a lightweight feature-importance proxy; no claim of full SHAP fidelity or analyst-complete explanations.
- **Generalization:** CICIoT2023 is a transfer-stress experiment with limitations; no claim of general cross-dataset robustness.
- **Numerical checks:** the quantization diagnostic reports no observed violations at the documented `1e-8` tolerance; it is not a formally machine-checked proof.
- **Deployment:** no fielded physical 5G network, guaranteed operational FPR, or production cybersecurity protection is claimed.

## Supporting records

- [Method/component interpretations](../results/mdpi_r3/INTERPRETATION.md)
- [Quantization diagnostics](../results/mdpi_r3/quantization_diagnostics.csv)
- [Round-2/control reconciliation](../results/mdpi_r3/round2_control_reconciliation.csv)
- [Complete research evidence archive](../artifacts/xmag_r3_complete_evidence.zip), including full per-task derived measurements, exact executed-source snapshots, split manifests, and code; file checksums in [`MANIFEST.json`](../results/mdpi_r3/MANIFEST.json).

Earlier study summaries in `results/mdpi_r1/` and `results/mdpi_r1_ciciot/` retain their original experimental identities. See [provenance](PROVENANCE.md) rather than silently substituting newer values into historical tables.
