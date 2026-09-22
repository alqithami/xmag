#!/usr/bin/env python3
"""Aggregate saved matched trials without retraining or changing measurements."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP = ["method", "score", "message_bytes_per_flow"]
KEY = ["seed", "held_out_attack", *GROUP]
MEASURES = ["known_macro_f1", "unknown_auroc", "unknown_recall_conformal05",
            "known_fpr_conformal05"]
REQUIRED = [*KEY, *MEASURES, "per_flow_predict_us", "model_size_bytes"]
FRONTIER_SCORES = {
    "Class+anomaly-12B": "composite", "Class+proxy-18B": "composite",
    "X-MAG-COS-16Q": "composite", "X-MAG-COS-20B": "composite",
    "X-MAG-COS-24B": "composite", "X-MAG-COS-30B": "composite",
    "Distributed-logit-average": "entropy+anomaly", "Central-RF-full": "entropy",
}


def validate_metrics(df: pd.DataFrame, expected_trials: int | None = None) -> None:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    if df.empty or df[KEY].isna().any().any():
        raise ValueError("Empty metrics or missing trial identifiers.")
    if "scenario" in df and (df.scenario.isna().any() or df.scenario.nunique() != 1):
        raise ValueError("Aggregate exactly one scenario at a time.")
    # Normalize only comparison keys, not the supplied measurements or labels.
    keys = df[KEY].copy()
    keys["held_out_attack"] = keys["held_out_attack"].astype(str).str.strip().str.casefold()
    if keys.duplicated().any():
        raise ValueError("Duplicate seed/holdout/method/score/budget rows; do not double-count reruns.")
    values = df[MEASURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Detection metrics must be finite numbers in [0, 1].")
    pairs = set(map(tuple, keys[["seed", "held_out_attack"]].to_numpy()))
    if expected_trials is not None and len(pairs) != expected_trials:
        raise ValueError(f"Expected {expected_trials} seed-holdout trials, found {len(pairs)}.")
    for name, group in keys.groupby(GROUP, dropna=False):
        observed = set(map(tuple, group[["seed", "held_out_attack"]].to_numpy()))
        if observed != pairs:
            raise ValueError(f"Unmatched trial coverage for {name}: {len(observed)} of {len(pairs)} pairs.")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    validate_metrics(df)
    return df.groupby(GROUP, dropna=False).agg(
        n=("unknown_auroc", "size"),
        known_f1_mean=("known_macro_f1", "mean"), known_f1_std=("known_macro_f1", "std"),
        auroc_mean=("unknown_auroc", "mean"), auroc_std=("unknown_auroc", "std"),
        auroc_min=("unknown_auroc", "min"),
        recall_mean=("unknown_recall_conformal05", "mean"),
        recall_std=("unknown_recall_conformal05", "std"),
        recall_min=("unknown_recall_conformal05", "min"),
        fpr_mean=("known_fpr_conformal05", "mean"), fpr_max=("known_fpr_conformal05", "max"),
        latency_us_mean=("per_flow_predict_us", "mean"),
        model_size_bytes_mean=("model_size_bytes", "mean"),
    ).reset_index()


def select_pareto_rows(summary: pd.DataFrame) -> pd.DataFrame:
    expected = summary["method"].map(FRONTIER_SCORES)
    selected = summary[expected.notna() & summary["score"].eq(expected)].copy()
    return selected.sort_values(["message_bytes_per_flow", "method"]).reset_index(drop=True)


def per_family_summary(df: pd.DataFrame, method: str = "X-MAG-COS-16Q") -> pd.DataFrame:
    selected = df[(df.method == method) & (df.score == "composite")]
    if selected.empty:
        raise ValueError(f"No composite-score rows for {method}.")
    result = selected.groupby("held_out_attack", sort=True).agg(
        n_seeds=("seed", "nunique"), known_f1_mean=("known_macro_f1", "mean"),
        known_f1_std=("known_macro_f1", "std"), auroc_mean=("unknown_auroc", "mean"),
        auroc_std=("unknown_auroc", "std"),
        recall_mean=("unknown_recall_conformal05", "mean"),
        recall_std=("unknown_recall_conformal05", "std"),
        fpr_mean=("known_fpr_conformal05", "mean"), fpr_std=("known_fpr_conformal05", "std"),
    ).reset_index()
    result.insert(0, "method", method)
    return result


def save_figure(fig, directory: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(data: pd.DataFrame, directory: Path) -> None:
    # Separate figures keep labels readable and preserve the existing AUROC filename.
    for metric, error, ylabel, name in [
        ("auroc_mean", "auroc_std", "Mean unknown-family AUROC", "pareto_bytes_auroc"),
        ("recall_mean", "recall_std", "Mean recall at nominal conformal 5%", "pareto_bytes_recall"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for _, row in data.iterrows():
            yerr = float(row[error]) if pd.notna(row[error]) else None
            ax.errorbar(row.message_bytes_per_flow, row[metric], yerr=yerr,
                        fmt="o", capsize=3, label=row.method)
        ax.set_xscale("log")
        ax.set_xlabel("Per-flow evidence / feature payload (bytes; log scale)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.12)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        save_figure(fig, directory, name)


def plot_per_family(data: pd.DataFrame, directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    pos = np.arange(len(data)); width = 0.38
    ax.bar(pos - width / 2, data.auroc_mean, width,
           yerr=data.auroc_std.fillna(0), capsize=3, label="AUROC")
    ax.bar(pos + width / 2, data.recall_mean, width,
           yerr=data.recall_std.fillna(0), capsize=3, label="Recall at conformal 5%")
    # Keep native family/category labels; CICIoT2023 is not a 5G-NIDD label set.
    ax.set_xticks(pos, data.held_out_attack.astype(str), rotation=35, ha="right")
    ax.set_ylabel("Mean across the recorded seeds (bars: sample SD)")
    ax.set_title(str(data.method.iloc[0]))
    ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=0.25); ax.legend()
    save_figure(fig, directory, "per_family_open_set")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/mdpi_r1")
    parser.add_argument("--metrics-csv", help="Aggregate an existing all_metrics.csv instead of local run folders.")
    parser.add_argument("--outdir", default="results/mdpi_r1")
    parser.add_argument("--expected-trials", type=int)
    parser.add_argument("--method", default="X-MAG-COS-16Q")
    args = parser.parse_args()
    try:
        if args.metrics_csv:
            data = pd.read_csv(args.metrics_csv)
        else:
            sources = sorted(Path(args.runs_root).rglob("metrics.csv"))
            if not sources:
                raise ValueError("No metrics.csv files found. Use --metrics-csv for an existing aggregate archive.")
            data = pd.concat([pd.read_csv(path) for path in sources], ignore_index=True)
        validate_metrics(data, args.expected_trials)
        summary = aggregate(data)
        pareto = select_pareto_rows(summary)
        family = per_family_summary(data, args.method)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Aggregation stopped: {exc}\n")
    out = Path(args.outdir); figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "all_metrics.csv", index=False)
    summary.to_csv(out / "protocol_matched_summary.csv", index=False)
    pareto.to_csv(out / "message_pareto.csv", index=False)
    family.to_csv(out / "per_family_summary.csv", index=False)
    plot_pareto(pareto, figures)
    plot_per_family(family, figures)
    print(summary.to_string(index=False))
    print(f"\nSaved verified aggregates and figures under {out}")


if __name__ == "__main__":
    main()
