from pathlib import Path
import subprocess
import sys
import yaml


def test_pipeline_smoke(tmp_path):
    data_path = tmp_path / "synthetic.csv"
    run_dir = tmp_path / "run"
    cfg_path = tmp_path / "config.yaml"
    subprocess.check_call([sys.executable, "xmag_pipeline.py", "synth", "--out", str(data_path), "--rows", "300"])
    cfg = yaml.safe_load(Path("configs/synthetic_smoke.yaml").read_text())
    cfg["dataset"]["path"] = str(data_path)
    cfg["experiment"]["output_dir"] = str(run_dir)
    cfg["model"]["n_estimators"] = 10
    cfg["explanation"]["k_values"] = [1, 3]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.check_call([sys.executable, "xmag_pipeline.py", "run", "--config", str(cfg_path)])
    assert (run_dir / "results.csv").exists()
