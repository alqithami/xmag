#!/usr/bin/env bash
set -euo pipefail

for CFG in configs/holdouts/real_5g_nidd_*.yaml
do
  NAME=$(basename "$CFG" .yaml)
  echo "Running $NAME"
  python scripts/xmag_hybrid_message_diagnostic.py \
    --config "$CFG" \
    --out "runs/hybrid_${NAME}" \
    --alpha 1.0 \
    --top-m 1
done
