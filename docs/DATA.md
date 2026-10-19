# Data and access

The X-MAG-COS evaluation uses third-party network intrusion datasets. **Neither the dataset files nor per-flow records are redistributed by this repository.** You must obtain them independently and comply with the respective licenses/terms.

## 5G-NIDD — primary matched evaluation

- Publisher's dataset DOI: https://doi.org/10.21227/XTEP-HV36
- Official Fairdata record: https://etsin.fairdata.fi/dataset/9d13ef28-2ca7-44b0-9950-225359afac65
- Place the encoded export at `data/5G-NIDD/Encoded.csv` from the repository root.
- The exact experiment input had **SHA-256 `7c238e2d5dabbc1afcd01c50db91372d6f6808f7572bb92baeb2df03a18af90c`**.
- The original study's data preprocessing excludes labels, attack tools, endpoint identifiers, and other potentially leaking fields. The `sVid` metadata is used only for grouping/attribution; it is excluded from predictors.
- Nominal eight-slot metadata yields **two occupied source partitions** in this recorded dataset, so these experiments are not evidence of coordination between eight physical monitors.

Confirm the input before attempting a full reproduction:

```bash
python - <<'PY'
from pathlib import Path
from hashlib import sha256
p = Path("data/5G-NIDD/Encoded.csv")
with p.open("rb") as f:
    h = sha256()
    for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
```

If the checksum differs, do not interpret the resulting metrics as a direct reproduction of the reported experiment.

## CICIoT2023 — transfer-stress evaluation

- Official dataset page: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Previously measured transfer-stress summaries and category counts are retained under `results/mdpi_r1_ciciot/`.
- The available public snapshots document the evaluated sample/category construction, but the full third-party raw-data transformations and saved predictions are not redistributed here. The cross-dataset results must **not** be interpreted as evidence of broad transfer success.

## Reproduction boundary

Three activities have different evidentiary status:

1. **Audit frozen evidence:** no raw data required; verifies manifests, published tables, and executable tests.
2. **Synthetic smoke check:** verifies software plumbing only, and must not be reported as a scientific result.
3. **Full real-data replay:** uses the exact dataset above to re-execute the corrected 40-task multi-agent experiment and fixed-FPR extension, including additional model refits. Values may vary across library versions/platforms, even when seeds and training protocols match.

The archive does not contain proprietary/sensitive traffic, downloadable third-party dataset copies, trained estimator archives, or per-flow score arrays.
