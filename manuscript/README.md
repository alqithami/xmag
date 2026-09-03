# Revised MDPI Mathematics Manuscript

The revised paper is split into `main.tex` and eight section files under `sections/` to keep the repository source reviewable. The source uses the official MDPI class path:

```text
Definitions/mdpi.cls
```

Obtain the current MDPI LaTeX template from the journal and place its `Definitions/` directory inside `manuscript/` before compiling.

## Generate figures

From the repository root, after reproducing or restoring the matched experiment outputs:

```bash
bash scripts/generate_mdpi_remaining_figures.sh
mkdir -p manuscript/figures
cp results/mdpi_r1/manuscript_figures/*.pdf manuscript/figures/
```

The required manuscript figures are:

```text
per_family_5g.pdf
pareto_auroc.pdf
roc_udpflood.pdf
roc_slowratedos.pdf
score_histogram_udpflood.pdf
score_histogram_slowratedos.pdf
accepted_unknown_assignments.pdf
sensitivity_auroc.pdf
ciciot_per_category.pdf
```

The last two figures are already reproducible from the committed sensitivity and CICIoT2023 aggregate results. The complete resubmission package supplied to the authors contains all final PDFs.

## Compile

From `manuscript/`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Author actions before resubmission

1. Replace the anonymous author, ORCID, affiliation, and correspondence placeholders with the exact submitted metadata.
2. Add the verified National Cybersecurity Authority award number if the award has a formal number.
3. Keep the funding statement without an invented placeholder if no formal award number exists.
4. Verify that the source and response letter use the same manuscript title and author metadata.
