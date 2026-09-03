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
  scripts/mdpi_revision_common.py
  scripts/mdpi_revision_trial.py
  scripts/mdpi_revision_aggregate.py
  scripts/export_mdpi_hard_holdout_support.py
  scripts/run_mdpi_hardcase_raw.sh
  scripts/plot_mdpi_manuscript_figures.py
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
  exit 1
fi

mkdir -p "$RESULT_ROOT"

printf '\n[1/4] Checking revision and plotting scripts...\n'
"$PYTHON" -m py_compile \
  scripts/mdpi_revision_common.py \
  scripts/mdpi_revision_trial.py \
  scripts/mdpi_revision_aggregate.py \
  scripts/export_mdpi_hard_holdout_support.py \
  scripts/plot_mdpi_manuscript_figures.py

printf '\n[2/4] Refreshing protocol-matched summary tables...\n'
"$PYTHON" scripts/mdpi_revision_aggregate.py \
  --runs-root "$RUN_ROOT" \
  --outdir "$RESULT_ROOT"

printf '\n[3/4] Generating hard-holdout ROC, score-distribution, and assignment support...\n'
bash scripts/run_mdpi_hardcase_raw.sh

printf '\n[4/4] Generating publication-ready figures and archive...\n'
rm -rf "$FINAL_ROOT"
"$PYTHON" scripts/plot_mdpi_manuscript_figures.py \
  --results-root "$RESULT_ROOT" \
  --hard-support-dir "$HARD_ROOT" \
  --outdir "$FINAL_ROOT"

cp -f "$RESULT_ROOT/message_pareto.csv" "$FINAL_ROOT/"
cp -f "$RESULT_ROOT/protocol_matched_summary.csv" "$FINAL_ROOT/"
cp -f "$HARD_ROOT/hard_holdout_metrics.csv" "$FINAL_ROOT/"
cp -f "$HARD_ROOT/hard_holdout_summary.csv" "$FINAL_ROOT/"
cp -f "$HARD_ROOT/accepted_unknown_confusion_aggregate.csv" "$FINAL_ROOT/"
cp -f "$HARD_ROOT/accepted_unknown_confusion_by_seed.csv" "$FINAL_ROOT/"
cp -f "$HARD_ROOT/environment_manifest.json" "$FINAL_ROOT/"

cat > "$FINAL_ROOT/FIGURE_MANIFEST.txt" <<'EOF'
Publication-ready main figures
------------------------------
pareto_auroc.pdf
  Communication payload versus mean unknown-family AUROC.

per_family_5g.pdf
  Mean AUROC and split-conformal 5% recall by held-out family, with one-standard-deviation error bars.

Reviewer-requested hard-family diagnostics
------------------------------------------
roc_udpflood.pdf
roc_slowratedos.pdf
  Mean ROC curves across five seeds, with one-standard-deviation bands.

score_histogram_udpflood.pdf
score_histogram_slowratedos.pdf
  Known-versus-held-out composite-score distributions.

accepted_unknown_assignments.pdf
  Aggregate known-class assignments of held-out decisions that were not rejected.

The separate accepted_unknown_confusion_*.pdf files are retained as supplementary diagnostics.
EOF

rm -f "$FINAL_ZIP"
(
  cd "$RESULT_ROOT"
  zip -qr "$(basename "$FINAL_ZIP")" "$(basename "$FINAL_ROOT")"
)

printf '\nGenerated manuscript-ready figures:\n'
find "$FINAL_ROOT" -maxdepth 1 -type f \( -name '*.pdf' -o -name '*.png' \) -print | sort
printf '\nCompleted. Share this archive:\n%s\n' "$FINAL_ZIP"
