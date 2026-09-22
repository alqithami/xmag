"""Aggregation regressions; fixtures are test inputs, not scientific results."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mdpi_revision_aggregate import (aggregate, per_family_summary,
                                     select_pareto_rows, validate_metrics)


def fixture_rows():
    rows = []
    for seed in [7, 42]:
        for family in ["DDoS", "Mirai"]:
            for method, score, size in [
                ("X-MAG-COS-16Q", "composite", 16),
                ("X-MAG-COS-24B", "composite", 24),
                ("X-MAG-COS-24B", "local_anomaly", 24),
                ("Central-RF-full", "entropy", 352),
                ("Central-RF-full", "EVT-tail", 352),
            ]:
                rows.append(dict(seed=seed, held_out_attack=family, method=method,
                    score=score, scenario="standard", message_bytes_per_flow=size,
                    known_macro_f1=0.9, unknown_auroc=0.8 + seed / 1000,
                    unknown_recall_conformal05=0.7, known_fpr_conformal05=0.05,
                    per_flow_predict_us=1.0, model_size_bytes=np.nan))
    return pd.DataFrame(rows)


def test_aggregate_retains_measurements_and_counts():
    data = fixture_rows()
    before = data.copy(deep=True)
    validate_metrics(data, expected_trials=4)
    result = aggregate(data)
    assert set(result.n) == {4}
    assert np.allclose(result.auroc_mean, 0.8245)
    pd.testing.assert_frame_equal(data, before)


def test_frontier_does_not_mix_score_ablations():
    rows = select_pareto_rows(aggregate(fixture_rows()))
    assert len(rows) == 3
    assert rows.method.is_unique
    assert set(rows.score) == {"composite", "entropy"}


def test_ciciot_category_names_are_preserved():
    family = per_family_summary(fixture_rows())
    assert set(family.held_out_attack) == {"DDoS", "Mirai"}
    assert set(family.n_seeds) == {2}
    assert set(family.method) == {"X-MAG-COS-16Q"}


def test_duplicate_trials_are_rejected():
    data = fixture_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(pd.concat([data, data.iloc[:1]], ignore_index=True))


def test_missing_baseline_trial_is_rejected():
    with pytest.raises(ValueError, match="Unmatched"):
        aggregate(fixture_rows().iloc[1:])


def test_wrong_expected_trial_count_is_rejected():
    with pytest.raises(ValueError, match="Expected 40"):
        validate_metrics(fixture_rows(), expected_trials=40)


def test_mixed_scenarios_are_rejected():
    data = fixture_rows()
    data.loc[0, "scenario"] = "another_scenario"
    with pytest.raises(ValueError, match="one scenario"):
        aggregate(data)


def test_cli_from_saved_csv_generates_tables_and_plots(tmp_path):
    src = tmp_path / "metrics.csv"
    fixture_rows().to_csv(src, index=False)
    out = tmp_path / "output"
    script = Path(__file__).resolve().parents[1] / "scripts/mdpi_revision_aggregate.py"
    subprocess.run([sys.executable, str(script), "--metrics-csv", str(src),
                    "--expected-trials", "4", "--outdir", str(out)], check=True)
    assert (out / "per_family_summary.csv").is_file()
    assert len(pd.read_csv(out / "message_pareto.csv")) == 3
    for name in ["pareto_bytes_auroc", "pareto_bytes_recall", "per_family_open_set"]:
        for ext in ["pdf", "png"]:
            assert (out / "figures" / f"{name}.{ext}").stat().st_size > 0


def test_committed_output_checksums_match_manifest():
    import hashlib
    import json
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "results/RESULT_SYNC_MANIFEST.json").read_text())
    for relative, expected in manifest["outputs"].items():
        path = root / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
        assert len(pd.read_csv(path)) == expected["rows"]


@pytest.mark.parametrize("dataset,count", [("mdpi_r1", 40), ("mdpi_r1_ciciot", 7)])
def test_per_family_values_reconcile_with_principal_result(dataset, count):
    root = Path(__file__).resolve().parents[1] / "results" / dataset
    family = pd.read_csv(root / "per_family_summary.csv")
    summary = pd.read_csv(root / "protocol_matched_summary.csv")
    main = summary[(summary.method == "X-MAG-COS-16Q") & (summary.score == "composite")].iloc[0]
    assert int(main.n) == count == int(family.n_seeds.sum())
    for metric in ["known_f1_mean", "auroc_mean", "recall_mean", "fpr_mean"]:
        assert np.isclose(np.average(family[metric], weights=family.n_seeds), main[metric], atol=1e-12)
