#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
export MPLBACKEND=Agg
"$PYTHON" -u scripts/xmag_round2_run.py "$@"
