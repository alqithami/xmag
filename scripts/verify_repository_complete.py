#!/usr/bin/env python3
"""Fail fast when a required X-MAG-COS code or result artifact is missing.

This check uses only the Python standard library so that it can run before the
optional scientific stack is installed. It intentionally verifies committed
software and derived summaries, not raw datasets or local per-flow arrays.
"""
from __future__ import annotations

import csv
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ".github/workflows/ci.yml",
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "requirements.txt",
    "requirements-mdpi-r1.txt",
    "xmag_pipeline.py",
    "scripts/mdpi_revision_common.py",
    "scripts/mdpi_revision_trial.py",
    "scripts/mdpi_revision_aggregate.py",
    "scripts/mdpi_revision_sensitivity.py",
    "scripts/mdpi_revision_statistics.py",
    "scripts/mdpi_revision_shap_fidelity.py",
    "scripts/mdpi_revision_network_attack.py",
    "scripts/export_mdpi_hard_holdout_support.py",
    "scripts/plot_mdpi_manuscript_figures.py",
    "scripts/run_mdpi_round1_all.sh",
    "scripts/run_mdpi_hardcase_raw.sh",
    "scripts/generate_mdpi_remaining_figures.sh",
    "results/mdpi_r1/protocol_matched_summary.csv",
    "results/mdpi_r1/message_pareto.csv",
    "results/mdpi_r1/paired_statistics.csv",
    "results/mdpi_r1/hyperparameter_sensitivity_summary.csv",
    "results/mdpi_r1/network_attack_udpflood.csv",
    "results/mdpi_r1/network_attack_slowratedos.csv",
    "results/mdpi_r1/shap_fidelity_udpflood.csv",
    "results/mdpi_r1/shap_fidelity_slowratedos.csv",
    "results/mdpi_r1/hard_holdout_metrics.csv",
    "results/mdpi_r1/hard_holdout_summary.csv",
    "results/mdpi_r1/accepted_unknown_confusion_aggregate.csv",
    "results/mdpi_r1/accepted_unknown_confusion_by_seed.csv",
    "results/mdpi_r1/environment_manifest.json",
    "results/mdpi_r1/per_family_16q_figure_data.csv",
    "results/mdpi_r1_ciciot/protocol_matched_summary.csv",
    "results/mdpi_r1_ciciot/message_pareto.csv",
    "docs/EXPERIMENTAL_PROTOCOL.md",
    "docs/MDPI_ROUND1_RESULTS.md",
    "docs/MDPI_ROUND1_FIGURE_RESULTS.md",
    "docs/MDPI_ROUND1_RUNBOOK.md",
    "docs/MDPI_ROUND1_STATUS.md",
]

HOLDOUT_CONFIGS = {
    "real_5g_nidd_httpflood.yaml",
    "real_5g_nidd_icmpflood.yaml",
    "real_5g_nidd_slowratedos.yaml",
    "real_5g_nidd_synflood.yaml",
    "real_5g_nidd_synscan.yaml",
    "real_5g_nidd_tcpconnectscan.yaml",
    "real_5g_nidd_udpflood.yaml",
    "real_5g_nidd_udpscan.yaml",
}

FORBIDDEN_PUBLICATION_DIRS = {
    "manuscript",
    "paper",
    "papers",
    "submission",
    "submissions",
    "responses",
    "response_to_reviewers",
}

CSV_REQUIREMENTS = {
    "results/mdpi_r1/protocol_matched_summary.csv": {
        "method", "score", "message_bytes_per_flow", "known_f1_mean",
        "auroc_mean", "recall_mean", "fpr_mean",
    },
    "results/mdpi_r1/message_pareto.csv": {
        "method", "message_bytes_per_flow", "auroc_mean", "recall_mean",
    },
    "results/mdpi_r1/hard_holdout_summary.csv": {
        "held_out_attack", "n_seeds", "auroc_mean", "recall_mean", "fpr_mean",
    },
    "results/mdpi_r1_ciciot/protocol_matched_summary.csv": {
        "method", "score", "message_bytes_per_flow", "auroc_mean", "recall_mean",
    },
}


def fail(messages: list[str]) -> None:
    print("Repository completeness check failed:", file=sys.stderr)
    for message in messages:
        print(f"  - {message}", file=sys.stderr)
    raise SystemExit(1)


def csv_columns(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return set(next(reader))
        except StopIteration:
            return set()


def main() -> None:
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
        elif path.stat().st_size == 0:
            errors.append(f"empty required file: {relative}")

    holdout_dir = ROOT / "configs" / "holdouts"
    present_holdouts = {p.name for p in holdout_dir.glob("*.yaml")} if holdout_dir.is_dir() else set()
    missing_holdouts = HOLDOUT_CONFIGS - present_holdouts
    if missing_holdouts:
        errors.append(f"missing holdout configurations: {sorted(missing_holdouts)}")

    for dirname in sorted(FORBIDDEN_PUBLICATION_DIRS):
        if (ROOT / dirname).exists():
            errors.append(f"publication directory must not be committed: {dirname}/")

    citation_path = ROOT / "CITATION.cff"
    if citation_path.is_file():
        citation = citation_path.read_text(encoding="utf-8")
        for forbidden in ("Placeholder", "CRPG-00-0000"):
            if forbidden in citation:
                errors.append(f"CITATION.cff contains forbidden placeholder: {forbidden}")

    for relative, required_columns in CSV_REQUIREMENTS.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        columns = csv_columns(path)
        missing = required_columns - columns
        if missing:
            errors.append(f"{relative} is missing columns: {sorted(missing)}")

    python_files = [ROOT / "xmag_pipeline.py", *(ROOT / "scripts").glob("*.py")]
    for path in sorted(python_files):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"Python syntax error in {path.relative_to(ROOT)}: {exc.msg}")

    oversized = []
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and path.stat().st_size > 25 * 1024 * 1024:
            oversized.append(str(path.relative_to(ROOT)))
    if oversized:
        errors.append(f"unexpected committed files larger than 25 MiB: {sorted(oversized)}")

    if errors:
        fail(errors)

    print("Repository completeness check passed.")
    print(f"Required files: {len(REQUIRED_FILES)}")
    print(f"Holdout configurations: {len(HOLDOUT_CONFIGS)}")
    print(f"Python files compiled: {len(python_files)}")
    print("Publication source and raw datasets are excluded by policy.")


if __name__ == "__main__":
    main()
