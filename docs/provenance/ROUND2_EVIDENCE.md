# Completed Round-2 evidence integration

The author supplied `xmag_round2_review_text.txt`, exporting the completed `mdpi_r2` audit. It reports 40/40 requested seed-holdout tasks. This directory preserves **selected, traceable projections** and new reanalysis code. It is not a replacement for the author's complete audit archive, the exact executed training overlay, or per-flow arrays.

## Verified conclusions

- The nominal eight-slot 5G-NIDD map has two occupied metadata partitions: slot 0 has 114,571 Benign records; slot 1 has 1,101,319 records including every attack. No eight-physical-device interpretation is supported.
- Label-alignment permutation checks have maximum numerical error about 1.11e-16. With a Benign-only constant source, an equal two-source average makes Benign an argmax for every input. This—not empty placeholder votes—explains the observed constant-Benign macro-F1. Active and legacy averaging use two participating models in the executed audit.
- Corrected 16Q mean AUROC/recall at nominal 5% are 0.905429/0.724128, versus 0.899877/0.704646 for 12B. Achieved rejection rates differ and the formats also differ in precision/framing, so this is not an isolated attribution effect or a fixed-achieved-FPR comparison.
- Four-metric Holm-adjusted signed-rank p-values are 1.0 for AUROC and 0.130198 for recall. Positive unadjusted mean intervals and nonsignificant adjusted ranked tests concern different estimands. No equivalence claim follows.
- Controlled peer verification does not reproduce the earlier ideal gain or 1%-loss inversion. The maximum seed-level absolute AUROC change from 1% loss alone is 0.000144 (SlowrateDoS) and 0.000137 (UDPFlood).
- At nominal 0.1%, corrected 16Q mean recall is 0.088810. It does not dominate uncertainty-only (0.442998) or centralized RF (0.680993). Nominal alpha is not a hard realized error bound. The complete audit contains outlying RF rejection rates that must remain visible.
- Corrected UDPFlood 16Q mean AUROC is 0.481810: four of five seeds are below 0.5, not all five. All 2,285,058 accepted nominal-5% seed-trial decisions are Benign predictions; this is not a unique-flow count.

## Reanalysis

`python scripts/analyze_round2_review_evidence.py` validates all 40 paired keys and summary means to 1e-12, reports trial- and family-block uncertainty, and summarizes the matched peer rows. Four metrics form the Holm family, separately per testing procedure. Trials reuse one corpus and are not independent deployments.

`python scripts/render_round2_histogram_tex.py` validates all four seed-42 histogram populations against their exact sample counts and emits plotting-only LaTeX. This is not manuscript prose or training code.

A manual document-build utility can compile author-supplied source in an ephemeral Actions workspace. Source URLs are passed as inputs; publication source is not committed. Build artifacts and build-check limitations are explicit. The manuscript remains outside version control.

The main Round-1 directories are retained as historical snapshots. Do not replace their protocol labels or claim all corrected Round-2 training code has been synchronized merely because these selected result projections and analyses are available.
