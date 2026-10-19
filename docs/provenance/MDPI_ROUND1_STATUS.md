# MDPI Round-1 Code and Results Status

The repository-side revision is complete.

## Committed and verified

- protocol-matched five-seed/eight-holdout experiment implementation;
- centralized, distributed, FedAvg, EVT-tail, DH, score, and message-content baselines;
- composite-score sensitivity and paired statistical tests;
- network-artifact and targeted source-manipulation experiments;
- TreeSHAP attribution-proxy fidelity analysis;
- 12/16/18/20/24/30-byte communication frontier;
- CICIoT2023 transfer-stress summaries;
- UDPFlood and SlowrateDoS ROC, score-distribution, and accepted-unknown analysis code;
- derived summary CSV and JSON artifacts needed to reproduce the reported findings;
- regression tests for message serialization, composite-score properties, explicit one-vs-rest coordination, and the FedAvg dtype failure;
- CI checks on Python 3.11 and 3.12.

The requested hard-holdout figures have been generated successfully. Binary publication figures are not tracked because they are reproducible from the committed scripts and derived result data.

## Intentionally excluded

- journal manuscript and response letters;
- publisher templates and publication-ready PDFs;
- raw 5G-NIDD and CICIoT2023 datasets;
- complete local run directories and per-flow arrays;
- generated ZIP archives.

Run the repository audit with:

```bash
python scripts/verify_repository_complete.py
pytest -q
```
