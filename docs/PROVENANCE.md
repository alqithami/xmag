# Evidence provenance and research integrity

**Authoritative publication snapshot:** the paper's accepted-stage study evidence is identified by `results/mdpi_r3/PLAN.json`, `RUN_COMPLETENESS.json`, `MANIFEST.json`, and the complete supporting ZIP in `artifacts/`.

The names `mdpi_r1`, `mdpi_r2`, and `mdpi_r3` are retained internally to preserve exact original paths, source digests, and audit boundaries. They denote separately executed research protocols and are not interchangeable revisions of a single observation.

The public default branch provides one recommended entry point: `scripts/reproduce_paper.py`. The prior experimental and editorial record remains recoverable from Git history and the supporting evidence archive. Former procedural notes are sequestered under `docs/provenance/`, not presented as the primary usage instructions.

## Integrity and limitations

- The exact old 40-task training source is restored under `scripts/`, checked against the fixed source SHA-256 values in `results/mdpi_r3/PLAN.json`.
- The complete evidence ZIP contains **38 files**, including its own manifest. The manifest lists SHA-256 for **37** evidence files and is checked by `scripts/verify_release.py`.
- The complete ZIP's SHA-256 is **`0f94e7f057c9d5914b8251add19a98228c0368bda2bb621e888b7be8aa912c92`**.
- Retained score archives and fresh algorithm replacements have separate identifiers and reconciliation tables. **Do not pool them as independent datasets or represent deterministic refits as exact historical model states.**
- The accepted paper's code and result tables are available without its proprietary/third-party data; full retraining requires exact source data and substantial compute.
- A unit-test pass and manifest verification establish software/evidence integrity, **not** independent scientific replication.
- The data-availability dates are historical research-access metadata; file modification times are not evidence that a website was accessed on the same day.

## Publication release

Cite the fixed Git commit or its publication tag rather than a moving `main` branch. If you make changes after publication, develop them on a separate branch and use a new release; never retroactively replace frozen measurements.
