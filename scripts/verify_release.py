#!/usr/bin/env python3
"""Validate the immutable X-MAG-COS publication evidence and executable sources.

Standard-library only: no third-party dataset or trained model is required.
This validates retained artifacts, not scientific generalization or bitwise
replay on another machine.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = "7c238e2d5dabbc1afcd01c50db91372d6f6808f7572bb92baeb2df03a18af90c"
ARCHIVE_SHA = "0f94e7f057c9d5914b8251add19a98228c0368bda2bb621e888b7be8aa912c92"
ARCHIVE = ROOT / "artifacts/xmag_r3_complete_evidence.zip"
EVIDENCE = ROOT / "results/mdpi_r3"
SEEDS = {7, 21, 42, 84, 123}
HOLDOUTS = {
    "HTTPFlood", "ICMPFlood", "SlowrateDoS", "SYNFlood",
    "SYNScan", "TCPConnectScan", "UDPFlood", "UDPScan",
}
CHECKS = {
    ("baseline_16q", "all_known"): (0.0934, 0.5484, 0.7242),
    ("central_rf", "all_known"): (0.6781, 0.7557, 0.7942),
    ("local_extratrees_16q", "all_known"): (0.0336, 0.5238, 0.8062),
    ("fusion_mean_16q", "all_known"): (0.2482, 0.4492, 0.5724),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rows_from_bytes(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def require_sources(plan: dict) -> None:
    cfg = plan["configuration"]
    for name, digest in cfg["executed_round2_source_sha256"].items():
        p = ROOT / "scripts" / name
        require(p.is_file(), f"Missing executed source: {p}")
        require(sha256(p) == digest, f"Executed source was modified: {p}")
    for holdout, digest in cfg["local_holdout_configuration_sha256"].items():
        p = ROOT / "configs/holdouts" / f"real_5g_nidd_{holdout.lower()}.yaml"
        require(p.is_file() and sha256(p) == digest, f"Missing/changed holdout: {p}")
    require((ROOT / "experiments/paper/start.py").is_file(),
            "Missing publication experiment entry point")
    require((ROOT / "experiments/paper/test_extension.py").is_file(),
            "Missing publication experiment regression tests")
    require(plan["dataset_sha256"] == DATA_SHA, "Primary data fingerprint changed")


def require_summary() -> None:
    path = EVIDENCE / "fixed_fpr_summary.csv"
    actual = rows_from_bytes(path.read_bytes())
    require(len(actual) == 66, "Expected 11 methods x 2 denominators x 3 FPR levels")
    require(len({(r["method"], r["denominator"], r["target_fpr"]) for r in actual}) == 66,
            "Duplicate primary summary group")
    for (method, denominator), target in CHECKS.items():
        subset = [r for r in actual if r["method"] == method and r["denominator"] == denominator]
        subset.sort(key=lambda r: float(r["target_fpr"]))
        require(len(subset) == 3, f"Missing summary row for {method}")
        require([float(r["target_fpr"]) for r in subset] == [0.001, 0.01, 0.05],
                f"Incorrect FPR grid for {method}")
        for row, reference in zip(subset, target):
            val = float(row["empirical_recall_mean"])
            require(abs(val - reference) < 5.1e-5, f"Reported metric mismatch: {method} {val} vs {reference}")
            require(0 <= val <= 1, f"Out-of-range recall: {method} {val}")


def require_archive() -> tuple[int, int]:
    require(ARCHIVE.is_file(), "Missing frozen evidence archive")
    require(sha256(ARCHIVE) == ARCHIVE_SHA, "Evidence ZIP does not match frozen SHA-256")
    manifest = load(EVIDENCE / "MANIFEST.json")
    entries = manifest["files"]
    require(len(entries) == 37, "Wrong number of archived evidence files")
    with zipfile.ZipFile(ARCHIVE) as z:
        actual = set(z.namelist())
        expected = {f"results/mdpi_r3/{k}" for k in entries}
        expected.add("results/mdpi_r3/MANIFEST.json")
        require(actual == expected, "Evidence ZIP file inventory mismatch")
        for name, digest in entries.items():
            archived = z.read(f"results/mdpi_r3/{name}")
            require(hashlib.sha256(archived).hexdigest() == digest,
                    f"Archive checksum mismatch: {name}")
            committed = EVIDENCE / name
            if committed.is_file():
                require(committed.read_bytes() == archived,
                        f"Public extracted table differs from frozen archive: {name}")
        for name, count in (("all_fixed_fpr_metrics.csv", 2640),
                            ("all_nominal_metrics.csv", 1320),
                            ("quantization_diagnostics.csv", 240),
                            ("retained_round2_fixed_fpr.csv", 720)):
            require(len(rows_from_bytes(z.read("results/mdpi_r3/" + name))) == count,
                    f"Incorrect {name} row count")
        fixed = rows_from_bytes(z.read("results/mdpi_r3/all_fixed_fpr_metrics.csv"))
        require({int(r["seed"]) for r in fixed} == SEEDS, "Wrong seed set in fixed-FPR data")
        require({r["held_out_attack"] for r in fixed} == HOLDOUTS,
                "Wrong attack holdout set in fixed-FPR data")
    return len(entries) + 1, len(fixed)


def main() -> None:
    try:
        for rel in ("README.md", "CITATION.cff", "LICENSE", "requirements-reproduction.txt",
                    "docs/DATA.md", "docs/REPRODUCIBILITY.md", "docs/RESULTS.md",
                    "docs/PROVENANCE.md", "results/mdpi_r3/PLAN.json",
                    "results/mdpi_r3/RUN_COMPLETENESS.json"):
            require((ROOT / rel).is_file(), f"Missing publication file: {rel}")
        plan = load(EVIDENCE / "PLAN.json")
        complete = load(EVIDENCE / "RUN_COMPLETENESS.json")
        require_sources(plan)
        require(complete["status"] == "complete_for_prespecified_extension",
                "The frozen extension is not marked complete")
        require(complete["completed_tasks"] == complete["expected_tasks"] == 40,
                "Incomplete experimental task grid")
        require(complete["conditions_per_task"] == 11, "Wrong number of conditions")
        require(complete["nominal_rows"] == 1320 and complete["fixed_fpr_rows"] == 2640,
                "Unexpected experimental result counts")
        require(complete["quantization_numerical_violations"] == 0,
                "Recorded quantization check failed")
        require(abs(float(complete["max_between_method_empirical_fpr_spread"])) < 1e-12,
                "Realized paired FPRs differ")
        require_summary()
        archived_files, fixed_rows = require_archive()
        print("X-MAG-COS publication evidence: PASS")
        print("40/40 task instances; 11 conditions; 1,320 nominal rows")
        print(f"{fixed_rows:,} fixed-FPR measurements; {archived_files} verified archive files")
        print("Primary comparison, executed-source and data/config fingerprints: PASS")
        print("Raw datasets and trained models are not needed for this audit.")
    except (AssertionError, KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"X-MAG-COS publication evidence: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
