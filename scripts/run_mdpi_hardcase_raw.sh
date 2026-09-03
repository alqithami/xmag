#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
PRIMARY_ROOT="runs/mdpi_r1"
FALLBACK_ROOT="runs/mdpi_raw_support"
CONFIG_ROOT="configs/holdouts"
SEEDS=(7 21 42 84 123)
HOLDOUTS=(udpflood slowratedos)

if [[ ! -f scripts/mdpi_revision_trial.py ]]; then
  echo "Missing scripts/mdpi_revision_trial.py." >&2
  exit 1
fi

if [[ ! -f scripts/export_mdpi_hard_holdout_support.py ]]; then
  echo "Missing scripts/export_mdpi_hard_holdout_support.py." >&2
  exit 1
fi

mkdir -p "$FALLBACK_ROOT"

find_existing_npz() {
  local seed="$1"
  local attack="$2"
  find "$PRIMARY_ROOT" "$FALLBACK_ROOT" -type f -name score_components.npz 2>/dev/null \
    | grep -Ei "seed[_-]?${seed}.*${attack}|${attack}.*seed[_-]?${seed}" \
    | head -n 1 || true
}

for seed in "${SEEDS[@]}"; do
  for attack in "${HOLDOUTS[@]}"; do
    existing="$(find_existing_npz "$seed" "$attack")"
    if [[ -n "$existing" ]]; then
      echo "Found raw artifact: seed=${seed}, holdout=${attack}"
      continue
    fi

    config="$CONFIG_ROOT/real_5g_nidd_${attack}.yaml"
    out="$FALLBACK_ROOT/seed${seed}/real_5g_nidd_${attack}"
    if [[ ! -f "$config" ]]; then
      echo "Missing configuration: $config" >&2
      exit 1
    fi

    echo "Regenerating missing raw support: seed=${seed}, holdout=${attack}"
    rm -rf "$out"
    "$PYTHON" scripts/mdpi_revision_trial.py \
      --config "$config" \
      --seed "$seed" \
      --out "$out" \
      --skip-heavy-baselines
  done
done

# Export from the canonical run root first. When regenerated files were needed,
# copy them into a temporary combined root so that all five seeds are included.
COMBINED_ROOT="runs/mdpi_hardcase_combined"
rm -rf "$COMBINED_ROOT"
mkdir -p "$COMBINED_ROOT"

for root in "$PRIMARY_ROOT" "$FALLBACK_ROOT"; do
  if [[ -d "$root" ]]; then
    while IFS= read -r -d '' trial; do
      rel="${trial#${root}/}"
      target="$COMBINED_ROOT/$rel"
      mkdir -p "$(dirname "$target")"
      cp -R "$trial" "$target"
    done < <(find "$root" -type f -name score_components.npz -print0 | xargs -0 -n1 dirname -z 2>/dev/null || true)
  fi
done

"$PYTHON" scripts/export_mdpi_hard_holdout_support.py \
  --runs-root "$COMBINED_ROOT"

echo
echo "Completed. Share: results/mdpi_r1/mdpi_hard_holdout_support.zip"
