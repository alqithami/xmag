#!/usr/bin/env bash
set -euo pipefail

SCRIPT=${XMAG_COMPOSITE_SCRIPT:-scripts/xmag_composite_open_score_diagnostic.py}
mkdir -p runs/tmp_configs

for SEED in 7 42 123
do
  for CFG in configs/holdouts/real_5g_nidd_*.yaml
  do
    NAME=$(basename "$CFG" .yaml)
    TMP="runs/tmp_configs/${NAME}_seed${SEED}.yaml"
    python - <<PY
from pathlib import Path
p = Path("$CFG")
out = Path("$TMP")
s = p.read_text()
s = s.replace("random_state: 42", "random_state: $SEED")
s = s.replace("output_dir: runs/real_holdout_", "output_dir: runs/seed${SEED}_real_holdout_")
out.write_text(s)
PY
    echo "Running COS-24B seed=${SEED} holdout=${NAME}"
    python "$SCRIPT" \
      --config "$TMP" \
      --out "runs/seed${SEED}_cos24_${NAME}" \
      --top-m 1 \
      --k-values 1 \
      --gamma 0.25
  done
done
