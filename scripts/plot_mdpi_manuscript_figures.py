#!/usr/bin/env python3
"""Generate publication-ready figures for the MDPI Mathematics revision.

The script reads the protocol-matched result tables and the retained hard-
holdout support produced by ``export_mdpi_hard_holdout_support.py``. It does
not retrain any model and does not require raw third-party datasets.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_ORDER = [
    "httpflood",
    "icmpflood",
    "slowratedos",
    "synflood",
    "synscan",
    "tcpconnectscan",
    "udpflood",
    "udpscan",
]
FAMILY_LABELS = {
    "httpflood": "HTTPFlood",
    "icmpflood": "ICMPFlood",
    "slowratedos": "SlowrateDoS",
    "synflood": "SYNFlood",
    "synscan": "SYNScan",
    "tcpconnectscan": "TCPConnectScan",
    "udpflood": "UDPFlood",
    "udpscan": "UDPScan",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    return path


def select_main(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["method"] == "X-MAG-COS-16Q") & (df["score"] == "composite")].copy()
    if out.empty:
        raise ValueError("No X-MAG-COS-16Q/composite rows were found in all_metrics.csv.")
    return out


def per_family_figure(all_metrics: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    main = select_main(all_metrics)
    stat = (
        main.groupby("held_out_attack", as_index=False)
        .agg(
            auroc_mean=("unknown_auroc", "mean"),
            auroc_std=("unknown_auroc", "std"),
            recall_mean=("unknown_recall_conformal05", "mean"),
            recall_std=("unknown_recall_conformal05", "std"),
        )
    )
    stat["order"] = stat["held_out_attack"].map({x: i for i, x in enumerate(FAMILY_ORDER)})
    stat = stat.sort_values("order")
    labels = [FAMILY_LABELS.get(x, x) for x in stat["held_out_attack"]]

    pos = np.arange(len(stat), dtype=float)
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.25, 4.75))
    ax.bar(
        pos - width / 2,
        stat["auroc_mean"],
        width,
        yerr=stat["auroc_std"].fillna(0),
        capsize=3,
        label="AUROC",
    )
    ax.bar(
        pos + width / 2,
        stat["recall_mean"],
        width,
        yerr=stat["recall_std"].fillna(0),
        capsize=3,
        label="Recall at conformal 5%",
    )
    ax.set_xticks(pos, labels, rotation=34, ha="right")
    ax.set_ylabel("Mean performance over five seeds")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "per_family_5g.pdf", bbox_inches="tight")
    fig.savefig(outdir / "per_family_5g.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    stat.drop(columns="order").to_csv(outdir / "per_family_16q_figure_data.csv", index=False)
    return stat


def communication_frontier(summary: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    wanted = [
        "Class+anomaly-12B",
        "X-MAG-COS-16Q",
        "Class+proxy-18B",
        "X-MAG-COS-20B",
        "X-MAG-COS-24B",
        "X-MAG-COS-30B",
        "Distributed-logit-average",
        "Central-RF-full",
    ]
    pieces = []
    for method in wanted:
        part = summary[summary["method"] == method].copy()
        if method == "Central-RF-full":
            part = part[part["score"] == "entropy"]
        elif method == "Distributed-logit-average":
            part = part[part["score"] == "entropy+anomaly"]
        else:
            part = part[part["score"] == "composite"]
        if not part.empty:
            pieces.append(part.iloc[[0]])
    pareto = pd.concat(pieces, ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.45, 4.75))
    full_content = pareto[pareto["method"].isin([
        "X-MAG-COS-16Q", "X-MAG-COS-20B", "X-MAG-COS-24B", "X-MAG-COS-30B"
    ])].sort_values("message_bytes_per_flow")
    ax.plot(
        full_content["message_bytes_per_flow"],
        full_content["auroc_mean"],
        marker="o",
        linewidth=1.6,
        label="Full-content X-MAG-COS",
    )

    markers = {
        "Class+anomaly-12B": ("s", "Class + anomaly"),
        "Class+proxy-18B": ("D", "Class + attribution proxy"),
        "Distributed-logit-average": ("^", "All-source logit average"),
        "Central-RF-full": ("X", "Central RF, full features"),
    }
    for method, (marker, label) in markers.items():
        row = pareto[pareto["method"] == method]
        if row.empty:
            continue
        ax.scatter(
            row["message_bytes_per_flow"], row["auroc_mean"],
            marker=marker, s=68, label=label, zorder=4,
        )

    annotations = {
        "X-MAG-COS-16Q": (0, 10, "16Q"),
        "X-MAG-COS-20B": (0, 10, "20B"),
        "X-MAG-COS-24B": (0, 10, "24B"),
        "X-MAG-COS-30B": (0, 10, "30B"),
    }
    for method, (dx, dy, text) in annotations.items():
        row = pareto[pareto["method"] == method]
        if row.empty:
            continue
        ax.annotate(
            text,
            (float(row.iloc[0]["message_bytes_per_flow"]), float(row.iloc[0]["auroc_mean"])),
            xytext=(dx, dy), textcoords="offset points", ha="center", fontsize=8,
        )

    ax.set_xscale("log")
    ticks = sorted(pareto["message_bytes_per_flow"].dropna().astype(int).unique())
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Application-layer evidence or feature payload (bytes per flow; log scale)")
    ax.set_ylabel("Mean unknown-family AUROC (40 matched trials)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "pareto_auroc.pdf", bbox_inches="tight")
    fig.savefig(outdir / "pareto_auroc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pareto.to_csv(outdir / "message_pareto_figure_data.csv", index=False)
    return pareto


def accepted_assignment_figure(confusion: pd.DataFrame, outdir: Path) -> None:
    families = ["udpflood", "slowratedos"]
    classes = ["HTTPFlood", "Benign"]
    matrix = np.zeros((len(families), len(classes)), dtype=float)
    for i, family in enumerate(families):
        part = confusion[confusion["held_out_attack"].str.lower() == family]
        for j, class_name in enumerate(classes):
            row = part[part["assigned_known_class"] == class_name]
            if not row.empty:
                matrix[i, j] = float(row.iloc[0]["accepted_unknown_fraction"])

    fig, ax = plt.subplots(figsize=(7.25, 3.55))
    left = np.zeros(len(families), dtype=float)
    y = np.arange(len(families))
    for j, class_name in enumerate(classes):
        vals = matrix[:, j]
        bars = ax.barh(y, vals, left=left, label=class_name)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val <= 0:
                continue
            x = left[i] + val / 2
            if val < 0.03:
                ax.text(min(0.995, left[i] + val + 0.005), bar.get_y() + bar.get_height()/2,
                        f"{100*val:.2f}%", va="center", ha="left", fontsize=8)
            else:
                ax.text(x, bar.get_y() + bar.get_height()/2,
                        f"{100*val:.2f}%", va="center", ha="center", fontsize=8)
        left += vals

    ax.set_yticks(y, ["UDPFlood", "SlowrateDoS"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(x*100)}%" for x in np.linspace(0, 1, 6)])
    ax.set_xlabel("Fraction of missed held-out decisions")
    ax.set_ylabel("Held-out family")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "accepted_unknown_assignments.pdf", bbox_inches="tight")
    fig.savefig(outdir / "accepted_unknown_assignments.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_hard_figures(hard_dir: Path, outdir: Path) -> None:
    source = hard_dir / "figures"
    require(source)
    for stem in (
        "roc_udpflood", "roc_slowratedos",
        "score_histogram_udpflood", "score_histogram_slowratedos",
        "accepted_unknown_confusion_udpflood", "accepted_unknown_confusion_slowratedos",
    ):
        for ext in ("pdf", "png"):
            src = require(source / f"{stem}.{ext}")
            shutil.copy2(src, outdir / src.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/mdpi_r1")
    parser.add_argument("--hard-support-dir", default="results/mdpi_r1/hard_holdout_support")
    parser.add_argument("--outdir", default="results/mdpi_r1/manuscript_figures_revised")
    args = parser.parse_args()

    results = Path(args.results_root)
    hard = Path(args.hard_support_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    all_metrics = pd.read_csv(require(results / "all_metrics.csv"))
    summary = pd.read_csv(require(results / "protocol_matched_summary.csv"))
    confusion = pd.read_csv(require(hard / "accepted_unknown_confusion_aggregate.csv"))

    per_family_figure(all_metrics, out)
    communication_frontier(summary, out)
    accepted_assignment_figure(confusion, out)
    copy_hard_figures(hard, out)

    print(f"Generated manuscript-ready figures in: {out}")
    for path in sorted(out.glob("*")):
        if path.suffix.lower() in {".pdf", ".png", ".csv"}:
            print(path)


if __name__ == "__main__":
    main()
