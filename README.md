# X-MAG-COS

**Compact Evidence Messaging for Source-Partitioned Open-Set Intrusion Detection in 5G-IoT Networks**

Reproducibility companion to the article by **Mohammed Alliheedi and Saad Alqithami**, accepted in *Mathematics* (2026; manuscript mathematics-4541032). Bibliographic DOI and final issue details will be added when the publisher assigns them.

X-MAG-COS exchanges compact class, anomaly, and optional attribution evidence between source-specific monitors and a coordinator. A known-class classifier and a separately calibrated open-set detector evaluate previously unseen attacks. This repository contains the implementation, evaluation protocol, aggregated measurements, tests, and provenance needed to inspect the reported evidence.

## Start here

**Verify the published evidence without downloading traffic datasets:**

```bash
git clone https://github.com/alqithami/xmag.git
cd xmag
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python scripts/reproduce_paper.py verify
python -m pytest -q
```

**Run a synthetic software smoke test** (not scientific evidence):

```bash
python scripts/reproduce_paper.py smoke
```

**Re-run the real-data study** after obtaining the official 5G-NIDD dataset and placing the expected encoded file at `data/5G-NIDD/Encoded.csv`:

```bash
python scripts/reproduce_paper.py full
```

The complete replay is compute-intensive; it is resumable and stores models and flow-level scores locally under the ignored `runs/` directory. The original input file must match SHA-256 `7c238e2d5dabbc1afcd01c50db91372d6f6808f7572bb92baeb2df03a18af90c`. A different data export will be rejected rather than silently substituted. See **[Reproducibility](docs/REPRODUCIBILITY.md)** for requirements and the distinction between a frozen-evidence audit and experimental reruns.

## Reported results

For the 40 matched seed/attack-holdout instances, the final fixed-false-positive-rate comparison gives the following **mean unknown-attack recall**. The negative population here is all known-traffic observations, and the cutoffs are retrospective, not deployed calibration thresholds.

| Model | 0.1% target FPR | 1% target FPR | 5% target FPR |
| --- | ---: | ---: | ---: |
| X-MAG-COS 16Q (16-byte evidence) | 0.0934 | 0.5484 | 0.7242 |
| Centralized full-feature Random Forest | 0.6781 | 0.7557 | 0.7942 |
| Local Extra Trees replacement | 0.0336 | 0.5238 | 0.8062 |
| Mean-fusion replacement | 0.2482 | 0.4492 | 0.5724 |

These results do **not** establish superiority at every FPR: centralized RF is strong, and UDPFlood remains a serious open-set failure case. Source metadata identify **two populated partitions**, not eight independent physical devices. The attribution field is a feature-importance proxy, not a complete SHAP explanation. See **[Results and limitations](docs/RESULTS.md)**.

## Project contents

| Location | Purpose |
| --- | --- |
| `scripts/reproduce_paper.py` | One entry point for evidence audit, smoke test, and full real-data replay |
| `scripts/verify_release.py` | Standard-library verification of study scope, input/code fingerprints, tables, and archive |
| `scripts/xmag_round2_*.py`, `scripts/mdpi_revision_common.py` | Preserved, hash-verified core for the 40-task evaluation |
| `experiments/paper/` | Matched fixed-FPR, ablation, and quantization code and tests |
| `configs/holdouts/` | Eight held-out attack-family configurations |
| `results/mdpi_r3/` | Publication-stage aggregate tables, diagnostics, and protocol manifests |
| `results/mdpi_r1/`, `results/mdpi_r1_ciciot/`, `results/mdpi_r2/` | Supporting analyses retained with their original protocols |
| `artifacts/xmag_r3_complete_evidence.zip` | Hash-checked 38-file supporting evidence, including execution snapshots |
| `docs/` | [Data](docs/DATA.md) · [Protocol](docs/REPRODUCIBILITY.md) · [Results](docs/RESULTS.md) · [Provenance](docs/PROVENANCE.md) |

**Data policy:** Third-party raw datasets are not redistributed. Full per-flow arrays and trained models remain local. The public archive allows independent examination of recorded evidence and software tests, but the initial full evaluation requires downloading the specified input and substantial computation. Dataset and licensing instructions appear in [Data](docs/DATA.md).

## Reference, license, and stability

Please cite the *Mathematics* article for the scientific findings and this repository's fixed publication release for the software and evidence. See [`CITATION.cff`](CITATION.cff). Software is distributed under the [MIT License](LICENSE); external datasets retain their original terms.

This publication version freezes the validated study files and derived evidence. Any subsequent developments belong on separate branches/releases and must not silently revise the results cited by the accepted paper.
