#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
RUN_ROOT="runs/mdpi_r1"
RESULT_ROOT="results/mdpi_r1"
HARD_ROOT="$RESULT_ROOT/hard_holdout_support"
FINAL_ROOT="$RESULT_ROOT/manuscript_figures"
FINAL_ZIP="$RESULT_ROOT/xmag_mdpi_remaining_figures.zip"

required=(
  scripts/mdpi_revision_aggregate.py
  scripts/mdpi_revision_trial.py
  scripts/mdpi_revision_common.py
  scripts/run_mdpi_hardcase_raw.sh
  scripts/export_mdpi_hard_holdout_support.py
)

for file in "${required[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file" >&2
    exit 1
  fi
done

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "Missing matched-run directory: $RUN_ROOT" >&2
  exit 1
fi

if ! find "$RUN_ROOT" -type f -name metrics.csv -print -quit | grep -q .; then
  echo "No metrics.csv files were found below $RUN_ROOT." >&2
  echo "The five-seed matched experiment must exist before figures can be generated." >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT"

printf '\n[1/4] Checking Python scripts...\n'
"$PYTHON" -m py_compile \
  scripts/mdpi_revision_common.py \
  scripts/mdpi_revision_trial.py \
  scripts/mdpi_revision_aggregate.py \
  scripts/export_mdpi_hard_holdout_support.py

printf '\n[2/4] Regenerating the communication frontier and per-family figure...\n'
"$PYTHON" scripts/mdpi_revision_aggregate.py \
  --runs-root "$RUN_ROOT" \
  --outdir "$RESULT_ROOT"

printf '\n[3/4] Generating hard-holdout ROC, score-distribution, and confusion figures...\n'
bash scripts/run_mdpi_hardcase_raw.sh

printf '\n[4/4] Collecting manuscript-ready figures...\n'
rm -rf "$FINAL_ROOT"
mkdir -p "$FINAL_ROOT/main" "$FINAL_ROOT/hard_holdouts" "$FINAL_ROOT/support"

copy_required() {
  local source="$1"
  local destination="$2"
  if [[ ! -f "$source" ]]; then
    echo "Expected output was not generated: $source" >&2
    exit 1
  fi
  cp -f "$source" "$destination"
}

copy_required "$RESULT_ROOT/figures/pareto_bytes_auroc.pdf" "$FINAL_ROOT/main/"
copy_required "$RESULT_ROOT/figures/pareto_bytes_auroc.png" "$FINAL_ROOT/main/"
copy_required "$RESULT_ROOT/figures/per_family_open_set.pdf" "$FINAL_ROOT/main/"
copy_required "$RESULT_ROOT/figures/per_family_open_set.png" "$FINAL_ROOT/main/"

for holdout in udpflood slowratedos; do
  copy_required "$HARD_ROOT/figures/roc_${holdout}.pdf" "$FINAL_ROOT/hard_holdouts/"
  copy_required "$HARD_ROOT/figures/roc_${holdout}.png" "$FINAL_ROOT/hard_holdouts/"
  copy_required "$HARD_ROOT/figures/score_histogram_${holdout}.pdf" "$FINAL_ROOT/hard_holdouts/"
  copy_required "$HARD_ROOT/figures/score_histogram_${holdout}.png" "$FINAL_ROOT/hard_holdouts/"
  copy_required "$HARD_ROOT/figures/accepted_unknown_confusion_${holdout}.pdf" "$FINAL_ROOT/hard_holdouts/"
  copy_required "$HARD_ROOT/figures/accepted_unknown_confusion_${holdout}.png" "$FINAL_ROOT/hard_holdouts/"
done

copy_required "$RESULT_ROOT/message_pareto.csv" "$FINAL_ROOT/support/"
copy_required "$RESULT_ROOT/protocol_matched_summary.csv" "$FINAL_ROOT/support/"
copy_required "$HARD_ROOT/hard_holdout_metrics.csv" "$FINAL_ROOT/support/"
copy_required "$HARD_ROOT/hard_holdout_summary.csv" "$FINAL_ROOT/support/"
copy_required "$HARD_ROOT/accepted_unknown_confusion_aggregate.csv" "$FINAL_ROOT/support/"
copy_required "$HARD_ROOT/accepted_unknown_confusion_by_seed.csv" "$FINAL_ROOT/support/"
copy_required "$HARD_ROOT/environment_manifest.json" "$FINAL_ROOT/support/"

cat > "$FINAL_ROOT/FIGURE_MANIFEST.txt" <<'EOF'
Main manuscript figures
-----------------------
main/pareto_bytes_auroc.pdf
  Communication payload versus mean unknown-family AUROC.

main/per_family_open_set.pdf
  Mean AUROC and split-conformal 5% recall for each held-out 5G-NIDD family.

Reviewer-requested hard-holdout figures
---------------------------------------
hard_holdouts/roc_udpflood.pdf
hard_holdouts/roc_slowratedos.pdf
  Mean ROC curves across available seeds, with one-standard-deviation bands.

hard_holdouts/score_histogram_udpflood.pdf
hard_holdouts/score_histogram_slowratedos.pdf
  Known-versus-held-out composite-score distributions.

hard_holdouts/accepted_unknown_confusion_udpflood.pdf
hard_holdouts/accepted_unknown_confusion_slowratedos.pdf
  Known classes assigned to held-out flows that were not rejected.
EOF

rm -f "$FINAL_ZIP"
(
  cd "$RESULT_ROOT"
  zip -qr "$(basename "$FINAL_ZIP")" "$(basename "$FINAL_ROOT")"
)

printf '\nGenerated figures:\n'
find "$FINAL_ROOT" -type f \( -name '*.pdf' -o -name '*.png' \) -print | sort

printf '\nCompleted. Share this archive:\n%s\n' "$FINAL_ZIP"
