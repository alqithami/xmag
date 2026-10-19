"""Non-data publication package regression checks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_release  # noqa: E402


def test_study_archive_and_summaries():
    verify_release.main()


def test_primary_paper_links_resolve():
    must_exist = (
        "README.md", "CITATION.cff", "LICENSE",
        "docs/RESULTS.md", "docs/DATA.md", "docs/PROVENANCE.md",
        "docs/REPRODUCIBILITY.md", "artifacts/xmag_r3_complete_evidence.zip",
        "experiments/paper/start.py", "scripts/xmag_round2_run.py",
    )
    assert all((ROOT / name).is_file() for name in must_exist)


def test_sensitive_inputs_and_transient_outputs_are_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in ("data/*", "runs/", "*.npz", "*.joblib", "results/publication_reproduction/"):
        assert entry in ignored
