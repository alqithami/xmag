# X-MAG-IDS

**X-MAG-IDS** is a reproducible research repository for **explanation-as-communication multi-agent intrusion detection** in 5G-enabled IoT networks.

The key idea is simple: instead of sending raw traffic or full feature vectors, each lightweight security agent sends compact top-k explanation tokens. A coalition-level coordinator aggregates these tokens to detect known attacks and flag unknown/zero-day attacks.

## Research question

Can lightweight edge security agents communicate compact explanation tokens while preserving intrusion-detection and zero-day detection performance under communication and computation budgets?

## Repository contents

```text
xmag_pipeline.py              Complete runnable first pipeline
configs/                      YAML experiment configurations
docs/                         Reviewer guide and experimental protocol
manuscript/                   LaTeX manuscript scaffold
tests/                        Smoke tests
.github/workflows/ci.yml      Continuous-integration test
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

Generate a synthetic 5G-NIDD-like smoke-test dataset and run the pipeline:

```bash
python xmag_pipeline.py synth --out data/synthetic_5g_nidd_like.csv --rows 1200
python xmag_pipeline.py audit --csv data/synthetic_5g_nidd_like.csv --outdir runs/synthetic_audit
python xmag_pipeline.py run --config configs/synthetic_smoke.yaml
python xmag_pipeline.py summarize --run-dir runs/synthetic_smoke
```

## Running on 5G-NIDD

Download 5G-NIDD separately according to its terms, then edit `dataset.path` in `configs/5g_nidd_syn_flood.yaml`:

```bash
python xmag_pipeline.py audit --csv /path/to/Encoded.csv --outdir runs/5g_nidd_audit
python xmag_pipeline.py run --config configs/5g_nidd_syn_flood.yaml
```

Default pilot choices:

- primary dataset: 5G-NIDD `Encoded.csv`
- first unknown holdout: `SYN Flood`
- fallback holdout: `UDP Scan`
- transparent edge-monitor agents: `N = 8`
- token budgets: `k in {1, 3, 5}`

## Implemented baselines

1. Centralized full-feature classifier.
2. Local owner-agent-only classifier.
3. Majority vote across agents.
4. Logit-average communication.
5. Raw top-k feature coordinator.
6. Explanation-token coordinator.

## Reported metrics

- known accuracy
- known balanced accuracy
- known macro-F1
- unknown AUROC
- unknown recall at calibrated threshold
- known false-alarm rate at calibrated threshold
- message bytes per flow
- training time
- prediction time

## Scientific caution

The synthetic data path is only a smoke test. Paper claims should be made only after running the pipeline on the selected public IDS datasets and completing leakage-control inspection.
