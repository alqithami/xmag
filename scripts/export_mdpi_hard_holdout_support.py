#!/usr/bin/env python3
"""Export Reviewer-2 hard-holdout support without shell heredocs or retraining.

Reads existing ``score_components.npz`` files from the matched revision runs,
creates ROC curves, score histograms, accepted-unknown confusion summaries,
copies run metadata, and writes one ZIP archive for upload.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def seed_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    return -1


def run_dirs(root: Path, holdout: str) -> list[Path]:
    return sorted(
        {
            p.parent
            for p in root.rglob("score_components.npz")
            if p.parent.name.lower() == f"real_5g_nidd_{holdout.lower()}"
        },
        key=lambda p: (seed_from_path(p), str(p)),
    )


def threshold(values: np.ndarray, alpha: float) -> float:
    return float(np.quantile(values, 1.0 - alpha, method="higher"))


def sample(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= limit:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), size=limit, replace=False)]


def environment() -> dict[str, object]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib", "shap"):
        try:
            mod = __import__(name)
            versions[name] = str(mod.__version__)
        except Exception as exc:
            versions[name] = f"unavailable ({type(exc).__name__})"
    out: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "packages": versions,
    }
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"):
            try:
                out[key] = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs/mdpi_r1")
    ap.add_argument("--out-dir", default="results/mdpi_r1/hard_holdout_support")
    ap.add_argument("--holdouts", nargs="+", default=["udpflood", "slowratedos"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--hist-sample-per-seed", type=int, default=50000)
    ap.add_argument("--include-raw", action="store_true")
    args = ap.parse_args()

    root = Path(args.runs_root)
    out = Path(args.out_dir)
    figures = out / "figures"
    metadata = out / "metadata"
    shutil.rmtree(out, ignore_errors=True)
    figures.mkdir(parents=True)
    metadata.mkdir(parents=True)

    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    found = False

    for holdout in [h.lower() for h in args.holdouts]:
        dirs = run_dirs(root, holdout)
        if not dirs:
            continue
        found = True
        grid = np.linspace(0.0, 1.0, 1001)
        curves: list[np.ndarray] = []
        known_hist: list[np.ndarray] = []
        unknown_hist: list[np.ndarray] = []

        for d in dirs:
            seed = seed_from_path(d)
            npz = d / "score_components.npz"
            with np.load(npz, allow_pickle=True) as z:
                val = np.asarray(z["val_composite"], dtype=float)
                known = np.asarray(z["known_composite"], dtype=float)
                unknown = np.asarray(z["unknown_composite"], dtype=float)
                pred_unknown = np.asarray(z["pred_unknown"]).astype(str)

            tau = threshold(val, args.alpha)
            y = np.r_[np.zeros(len(known), dtype=int), np.ones(len(unknown), dtype=int)]
            s = np.r_[known, unknown]
            fpr, tpr, _ = roc_curve(y, s)
            interp = np.interp(grid, fpr, tpr)
            interp[0], interp[-1] = 0.0, 1.0
            curves.append(interp)
            known_hist.append(sample(known, args.hist_sample_per_seed, seed + 1701))
            unknown_hist.append(sample(unknown, args.hist_sample_per_seed, seed + 2909))

            accepted = unknown < tau
            accepted_total = int(accepted.sum())
            for assigned, count in pd.Series(pred_unknown[accepted]).value_counts().items():
                confusion_rows.append({
                    "held_out_attack": holdout,
                    "seed": seed,
                    "assigned_known_class": str(assigned),
                    "count": int(count),
                    "accepted_unknown_total": accepted_total,
                    "accepted_unknown_fraction": float(count / accepted_total) if accepted_total else 0.0,
                })

            metric_rows.append({
                "held_out_attack": holdout,
                "seed": seed,
                "unknown_auroc": float(roc_auc_score(y, s)),
                "unknown_recall_at_q95": float(np.mean(unknown >= tau)),
                "known_fpr_at_q95": float(np.mean(known >= tau)),
                "threshold": tau,
                "n_known": len(known),
                "n_unknown": len(unknown),
                "n_unknown_accepted": accepted_total,
            })

            for name in ("hardware.json", "profile.json", "protocol.json", "message_serialization.json"):
                src = d / name
                if src.exists():
                    shutil.copy2(src, metadata / f"seed{seed}_{holdout}_{name}")
            if args.include_raw:
                shutil.copy2(npz, metadata / f"seed{seed}_{holdout}_score_components.npz")

        matrix = np.stack(curves)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
        fig, ax = plt.subplots(figsize=(6.7, 4.8))
        ax.plot(grid, mean, label=f"Mean ROC ({len(curves)} seeds)")
        ax.fill_between(grid, np.clip(mean-std, 0, 1), np.clip(mean+std, 0, 1), alpha=0.2, label="±1 SD")
        ax.plot([0,1], [0,1], "--", linewidth=1, label="Random ranking")
        ax.axvline(args.alpha, linestyle=":", linewidth=1, label=f"FPR={args.alpha:.2f}")
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"Open-set ROC: {holdout}", xlim=(0,1), ylim=(0,1.01))
        ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
        fig.savefig(figures / f"roc_{holdout}.png", dpi=300); fig.savefig(figures / f"roc_{holdout}.pdf"); plt.close(fig)

        k = np.concatenate(known_hist); u = np.concatenate(unknown_hist)
        bins = np.linspace(min(k.min(), u.min()), max(k.max(), u.max()), 61)
        fig, ax = plt.subplots(figsize=(6.7, 4.8))
        ax.hist(k, bins=bins, density=True, alpha=0.55, label="Known test flows")
        ax.hist(u, bins=bins, density=True, alpha=0.55, label=f"Held-out {holdout}")
        ax.set(xlabel="Composite open-set score", ylabel="Density", title=f"Score distributions: {holdout}")
        ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
        fig.savefig(figures / f"score_histogram_{holdout}.png", dpi=300); fig.savefig(figures / f"score_histogram_{holdout}.pdf"); plt.close(fig)

    if not found:
        shutil.rmtree(out, ignore_errors=True)
        print(f"No score_components.npz files found below {root}.")
        print("Run these two commands, then rerun this exporter:\n")
        print("python scripts/mdpi_revision_trial.py --config configs/holdouts/real_5g_nidd_udpflood.yaml --seed 42 --skip-heavy-baselines --out runs/mdpi_r1_raw/seed42/real_5g_nidd_udpflood")
        print("python scripts/mdpi_revision_trial.py --config configs/holdouts/real_5g_nidd_slowratedos.yaml --seed 42 --skip-heavy-baselines --out runs/mdpi_r1_raw/seed42/real_5g_nidd_slowratedos")
        print("python scripts/export_mdpi_hard_holdout_support.py --runs-root runs/mdpi_r1_raw")
        raise SystemExit(2)

    metrics = pd.DataFrame(metric_rows).sort_values(["held_out_attack", "seed"])
    metrics.to_csv(out / "hard_holdout_metrics.csv", index=False)
    summary = metrics.groupby("held_out_attack").agg(
        n_seeds=("seed", "nunique"),
        auroc_mean=("unknown_auroc", "mean"), auroc_std=("unknown_auroc", "std"),
        recall_mean=("unknown_recall_at_q95", "mean"), recall_std=("unknown_recall_at_q95", "std"),
        fpr_mean=("known_fpr_at_q95", "mean"), fpr_std=("known_fpr_at_q95", "std"),
    ).reset_index()
    summary.to_csv(out / "hard_holdout_summary.csv", index=False)

    conf = pd.DataFrame(confusion_rows)
    if not conf.empty:
        conf.to_csv(out / "accepted_unknown_confusion_by_seed.csv", index=False)
        agg = conf.groupby(["held_out_attack", "assigned_known_class"], as_index=False)["count"].sum()
        agg["accepted_unknown_fraction"] = agg["count"] / agg.groupby("held_out_attack")["count"].transform("sum")
        agg.to_csv(out / "accepted_unknown_confusion_aggregate.csv", index=False)
        for holdout in agg["held_out_attack"].unique():
            d = agg[agg["held_out_attack"] == holdout].sort_values("accepted_unknown_fraction", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            ax.bar(d["assigned_known_class"], d["accepted_unknown_fraction"])
            ax.set(xlabel="Known class assigned", ylabel="Fraction of missed held-out flows", title=f"Accepted {holdout} flows")
            ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
            fig.savefig(figures / f"accepted_unknown_confusion_{holdout}.png", dpi=300)
            fig.savefig(figures / f"accepted_unknown_confusion_{holdout}.pdf"); plt.close(fig)

    (out / "environment_manifest.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    archive = out.parent / "mdpi_hard_holdout_support.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
    print("\nHard-holdout support generated successfully.")
    print(summary.to_string(index=False))
    print(f"\nShare this file:\n{archive}")


if __name__ == "__main__":
    main()
