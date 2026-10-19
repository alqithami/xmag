# Paper experiment: matched controls and fixed-FPR evaluation

This is the original, fingerprinted implementation used for the accepted-paper matched controls, plus tests. **Do not edit the scientific source files to produce a different set of numbers under the original paper's result labels.**

The primary command is run through the repository root:

```bash
python scripts/reproduce_paper.py full
```

This invokes `scripts/xmag_round2_run.py` first to create the required 40 source-model and score caches. It then executes `experiments/paper/start.py` for the component substitutions, matched realized false-positive counts, and quantization checks. To inspect pre-existing saved outputs, use `python scripts/reproduce_paper.py verify`.

The companion unit/fixture tests are:

```bash
python -m pytest -q experiments/paper/test_actual_fpr.py experiments/paper/test_extension.py experiments/paper/test_snapshot_protocol.py
```

### What is preserved

- The implemented local/coordination model choices and their source SHA-256 hashes.
- The exact data split and seed policy, with a deliberately separate normalization/calibration split.
- Retrospective FPR matching that uses the **same integer false-positive budget per paired task** and prespecified tie ordering.
- Independent-calibration operating points as separate measures; expected boundary randomization is **not** relabelled as observed rejection.
- Quantization-bound diagnostics for a fixed float32-trained model; numeric tolerance rather than a machine-checked proof.
- Protocol fingerprints and a reconciliation separating fresh refits from retained score archives.

See [protocol](../../docs/REPRODUCIBILITY.md), [results](../../docs/RESULTS.md), and [provenance](../../docs/PROVENANCE.md). Raw datasets, fitted models, and flow-level score files are not committed. The supplemental archive contains the exact recorded experiment source and independent file hashes.
