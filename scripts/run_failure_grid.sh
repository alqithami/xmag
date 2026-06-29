#!/usr/bin/env bash
set -euo pipefail

for ATTACK in udpflood slowratedos
do
  for A in 0.0 0.25 0.5 0.75 1.0
  do
    SAFE=${A/./p}
    echo "Running failure grid: attack=${ATTACK}, alpha=${A}"
    python scripts/xmag_hybrid_message_diagnostic.py \
      --config "configs/holdouts/real_5g_nidd_${ATTACK}.yaml" \
      --out "runs/failure_${ATTACK}_alpha_${SAFE}" \
      --alpha "$A" \
      --top-m 1 2 3 4 5 6 8
  done
done
