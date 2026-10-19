# Reviewer guide

## Minimal runnable check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python xmag_pipeline.py synth --out data/synthetic_5g_nidd_like.csv --rows 1200
python xmag_pipeline.py run --config configs/synthetic_smoke.yaml
python xmag_pipeline.py summarize --run-dir runs/synthetic_smoke
```

The synthetic check verifies that the full pipeline runs, but it is not a scientific result.

## Reproducing the pilot experiment

1. Download 5G-NIDD independently.
2. Copy `configs/5g_nidd_syn_flood.yaml` to a local config.
3. Set `dataset.path` to the local `Encoded.csv` path.
4. Run the audit and experiment commands.

## What to inspect

- `runs/*/split_summary.json`: leakage-controlled split details.
- `runs/*/transformed_features.csv`: post-preprocessing feature names.
- `runs/*/results.csv`: performance, open-set, and communication metrics.

## Current limitations

The first explanation-token implementation uses a lightweight feature-importance contribution proxy. SHAP/LIME hooks can be added after the first leakage-controlled result table is stable. Transparent edge-monitor agents are an experimental abstraction, not a real deployment claim.
