#!/usr/bin/env python3
"""Single entry point for X-MAG-COS publication evidence and real-data replay."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/5G-NIDD/Encoded.csv"
EXPECTED_DATA = "7c238e2d5dabbc1afcd01c50db91372d6f6808f7572bb92baeb2df03a18af90c"


def call(*args: str) -> None:
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def verify_data() -> None:
    if not DATA.is_file():
        raise SystemExit(
            f"Missing official 5G-NIDD Encoded.csv: {DATA}\n"
            "Obtain it from the dataset provider and see docs/DATA.md."
        )
    digest = hashlib.sha256()
    with DATA.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_DATA:
        raise SystemExit(
            "The supplied Encoded.csv differs from the published experiment.\n"
            f"Expected SHA-256: {EXPECTED_DATA}\nActual SHA-256:   {digest.hexdigest()}"
        )
    print("5G-NIDD input SHA-256: verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "smoke", "full"],
                        help="Audit archived evidence, run a synthetic smoke test, or rerun real-data experiments.")
    parser.add_argument("--output-name", default="publication_reproduction",
                        help="Fresh output scope for the final matched extension (default: publication_reproduction).")
    args = parser.parse_args()

    if args.command == "verify":
        call("scripts/verify_release.py")
        return

    if args.command == "smoke":
        call("xmag_pipeline.py", "synth", "--out", "data/synthetic_5g_nidd_like.csv", "--rows", "1200")
        call("xmag_pipeline.py", "run", "--config", "configs/synthetic_smoke.yaml")
        print("Synthetic software smoke test passed. This is not a scientific measurement.")
        return

    if args.output_name == "mdpi_r3" or not args.output_name.replace("_", "").isalnum():
        raise SystemExit("Choose a new alphanumeric output scope; frozen study files cannot be overwritten.")
    call("scripts/verify_release.py")
    verify_data()
    call("scripts/xmag_round2_run.py", "--csv", "data/5G-NIDD/Encoded.csv",
         "--configs", "configs/holdouts", "--outdir", "results/mdpi_r2_reproduced",
         "--runs-root", "runs/mdpi_r2")
    call("experiments/paper/start.py", "--root", str(ROOT), "--csv",
         "data/5G-NIDD/Encoded.csv", "--output-name", args.output_name)
    print(f"Completed replay outputs: results/{args.output_name}/", flush=True)
    print("Compare protocols and aggregate estimates, not binary-identical fitted model files.")


if __name__ == "__main__":
    main()
