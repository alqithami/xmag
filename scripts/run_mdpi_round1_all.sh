#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
CONFIG_DIR="${CONFIG_DIR:-configs/holdouts}"
RUN_ROOT="${RUN_ROOT:-runs/mdpi_r1}"
RESULT_ROOT="${RESULT_ROOT:-results/mdpi_r1}"
SEEDS=(${SEEDS:-7 21 42 84 123})

mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

$PYTHON -m pip install -r requirements-mdpi-r1.txt

echo "=== Protocol-matched five-seed/all-holdout suite ==="
for SEED in "${SEEDS[@]}"; do
  for CFG in "$CONFIG_DIR"/real_5g_nidd_*.yaml; do
    NAME="$(basename "$CFG" .yaml)"
    OUT="$RUN_ROOT/seed${SEED}/${NAME}"
    if [[ -f "$OUT/metrics.csv" ]]; then
      echo "skip existing seed=$SEED config=$NAME"
      continue
    fi
    echo "seed=$SEED config=$NAME"
    $PYTHON scripts/mdpi_revision_trial.py --config "$CFG" --seed "$SEED" --out "$OUT"
  done
done

echo "=== Hyperparameter sensitivity ==="
$PYTHON scripts/mdpi_revision_sensitivity.py --runs-root "$RUN_ROOT" --out "$RESULT_ROOT/hyperparameter_sensitivity.csv"

echo "=== Paired statistical tests ==="
$PYTHON scripts/mdpi_revision_statistics.py --runs-root "$RUN_ROOT" --out "$RESULT_ROOT/paired_statistics.csv"

echo "=== Network artifacts and active attacks on hard holdouts ==="
for ATTACK in udpflood slowratedos; do
  OUTFILE="$RESULT_ROOT/network_attack_${ATTACK}.csv"
  if [[ -f "$OUTFILE" ]]; then
    echo "skip existing $OUTFILE"
    continue
  fi
  $PYTHON scripts/mdpi_revision_network_attack.py --config "$CONFIG_DIR/real_5g_nidd_${ATTACK}.yaml" --seed 42 --out "$OUTFILE"
done

echo "=== TreeSHAP fidelity on two hard holdouts ==="
for ATTACK in udpflood slowratedos; do
  OUTFILE="$RESULT_ROOT/shap_fidelity_${ATTACK}.csv"
  if [[ -f "$OUTFILE" ]]; then
    echo "skip existing $OUTFILE"
    continue
  fi
  $PYTHON scripts/mdpi_revision_shap_fidelity.py --config "$CONFIG_DIR/real_5g_nidd_${ATTACK}.yaml" --seed 42 --sample 2000 --out "$OUTFILE"
done

echo "=== Aggregate and create figures ==="
$PYTHON scripts/mdpi_revision_aggregate.py --runs-root "$RUN_ROOT" --outdir "$RESULT_ROOT"

echo "=== Archive outputs ==="
rm -f "$RESULT_ROOT/xmag_mdpi_round1_outputs.zip"
zip -qr "$RESULT_ROOT/xmag_mdpi_round1_outputs.zip" "$RESULT_ROOT"
echo "Finished: $RESULT_ROOT/xmag_mdpi_round1_outputs.zip"
