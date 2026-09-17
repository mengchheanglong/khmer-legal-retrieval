"""
Publication-quality visualization generation for Khmer legal retrieval study.

Generates:
1. results/figures/primary_metrics_bar.png:
   Grouped bar chart comparing Recall@1, Recall@5, MRR@10, Hit@5 with 95% bootstrap error bars.
2. results/figures/learning_curves.png:
   Multi-panel training dynamics (InfoNCE loss & validation MRR@10) across epochs for A1, A2, A3.
3. results/figures/recall_at_k.png:
   Recall@k curves (k in {1, 5, 10}) contrasting sparse vs dense architectures.
4. results/figures/code_breakdown.png:
   Comparative performance breakdown between Civil Code 2007 and Criminal Code 2009.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "font.family": "sans-serif",
})

PALETTE = {
    "BM25": "#7f7f7f",           # Gray (lexical baseline)
    "E5 Zero-Shot": "#1f77b4",   # Blue
    "A1 BiLSTM": "#ff7f0e",      # Orange
    "A2 Linear Probe": "#2ca02c",# Green
    "A3 XLM-R Finetuned": "#d62728", # Red (Winner)
}


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_primary_metrics_bar(results_dir: Path, figures_dir: Path) -> Path:
    """Generate grouped bar chart for Primary Benchmark (T-Q) with 95% bootstrap CIs."""
    out_path = figures_dir / "primary_metrics_bar.png"

    models = [
        ("BM25", results_dir / "metrics" / "bm25_baseline.json"),
        ("E5 Zero-Shot", results_dir / "metrics" / "e5_zero_shot.json"),
        ("A1 BiLSTM", results_dir / "metrics" / "a1_bilstm.json"),
        ("A2 Linear Probe", results_dir / "metrics" / "a2_linear_probe.json"),
        ("A3 XLM-R Finetuned", results_dir / "metrics" / "a3_xlmr.json"),
    ]

    metrics = ["Recall@1", "Recall@5", "MRR@10", "Hit@5"]
    n_metrics = len(metrics)
    n_models = len(models)

    x = np.arange(n_metrics)
    width = 0.16

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (m_label, m_file) in enumerate(models):
        data = load_json(m_file)["tq_benchmark"]["overall"]
        means = []
        errs_lower = []
        errs_upper = []

        for m in metrics:
            val = data[m]
            mean = val["mean"]
            means.append(mean)
            errs_lower.append(mean - val["ci_lower"])
            errs_upper.append(val["ci_upper"] - mean)

        yerr = [errs_lower, errs_upper]
        pos = x + (i - (n_models - 1) / 2) * width
        rects = ax.bar(
            pos,
            means,
            width,
            yerr=yerr,
            capsize=3,
            label=m_label,
            color=PALETTE[m_label],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )

        # Label values on top of bars
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_ylabel("Metric Score")
    ax.set_title("Primary Benchmark Performance (T-Q: 200 Questions) with 95% Bootstrap CIs")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0, 0.70)
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved primary metrics bar chart -> {out_path}")
    return out_path


def plot_learning_curves(results_dir: Path, figures_dir: Path) -> Path:
    """Generate training loss and validation MRR@10 curves for A1, A2, and A3."""
    out_path = figures_dir / "learning_curves.png"

    # Best run logs
    a1_log = results_dir / "logs" / "a1_lr3e-3_drop0.3.csv"
    a2_log = results_dir / "logs" / "a2_lr1e-3_wd0.01.csv"
    a3_log = results_dir / "logs" / "a3_lr2e-5_wd0.01.csv"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    runs = [
        ("A1 BiLSTM (from scratch)", a1_log, PALETTE["A1 BiLSTM"], "o"),
        ("A2 Linear Probe (frozen)", a2_log, PALETTE["A2 Linear Probe"], "s"),
        ("A3 XLM-R (full fine-tuning)", a3_log, PALETTE["A3 XLM-R Finetuned"], "^"),
    ]

    for label, log_path, color, marker in runs:
        if not log_path.exists():
            continue
        df = pd.read_csv(log_path)
        ax1.plot(
            df["epoch"],
            df["train_loss"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
        )
        ax2.plot(
            df["epoch"],
            df["val_mrr10"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
        )

    ax1.set_title("Training Loss Convergence (InfoNCE Loss)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.set_title("Validation MRR@10 Generalization")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation MRR@10")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved learning curves figure -> {out_path}")
    return out_path


def plot_recall_at_k(results_dir: Path, figures_dir: Path) -> Path:
    """Generate Recall@k curve comparing sparse vs dense architectures."""
    out_path = figures_dir / "recall_at_k.png"

    models = [
        ("BM25", results_dir / "metrics" / "bm25_baseline.json"),
        ("E5 Zero-Shot", results_dir / "metrics" / "e5_zero_shot.json"),
        ("A1 BiLSTM", results_dir / "metrics" / "a1_bilstm.json"),
        ("A2 Linear Probe", results_dir / "metrics" / "a2_linear_probe.json"),
        ("A3 XLM-R Finetuned", results_dir / "metrics" / "a3_xlmr.json"),
    ]

    ks = [1, 5, 10]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for m_label, m_file in models:
        data = load_json(m_file)
        tq_recalls = [data["tq_benchmark"]["overall"][f"Recall@{k}"]["mean"] for k in ks]
        tt_recalls = [data["tt_benchmark"]["overall"][f"Recall@{k}"]["mean"] for k in ks]

        ax1.plot(
            ks,
            tq_recalls,
            marker="o",
            label=m_label,
            color=PALETTE[m_label],
            linewidth=2,
            markersize=6,
        )
        ax2.plot(
            ks,
            tt_recalls,
            marker="s",
            label=m_label,
            color=PALETTE[m_label],
            linewidth=2,
            markersize=6,
        )

    ax1.set_title("Primary Benchmark (T-Q: 200 Questions)")
    ax1.set_xlabel("Rank (k)")
    ax1.set_ylabel("Recall@k")
    ax1.set_xticks(ks)
    ax1.set_ylim(0, 0.65)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.set_title("Secondary Benchmark (T-T: 148 Titles)")
    ax2.set_xlabel("Rank (k)")
    ax2.set_ylabel("Recall@k")
    ax2.set_xticks(ks)
    ax2.set_ylim(0.4, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.suptitle("Recall@k Progression Across Architectures", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved recall at k figure -> {out_path}")
    return out_path


def plot_code_breakdown(results_dir: Path, figures_dir: Path) -> Path:
    """Generate Civil Code vs Criminal Code comparison."""
    out_path = figures_dir / "code_breakdown.png"

    models = [
        ("BM25", results_dir / "metrics" / "bm25_baseline.json"),
        ("E5 Zero-Shot", results_dir / "metrics" / "e5_zero_shot.json"),
        ("A1 BiLSTM", results_dir / "metrics" / "a1_bilstm.json"),
        ("A2 Linear Probe", results_dir / "metrics" / "a2_linear_probe.json"),
        ("A3 XLM-R Finetuned", results_dir / "metrics" / "a3_xlmr.json"),
    ]

    labels = [m[0] for m in models]
    civ_mrr = []
    crm_mrr = []

    for _, m_file in models:
        data = load_json(m_file)["tq_benchmark"]
        civ_mrr.append(data["civil_code"]["metrics"]["MRR@10"]["mean"])
        crm_mrr.append(data["criminal_code"]["metrics"]["MRR@10"]["mean"])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width / 2, civ_mrr, width, label="Civil Code 2007 (100 Qs)", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x + width / 2, crm_mrr, width, label="Criminal Code 2009 (100 Qs)", color="#c44e52", alpha=0.9)

    ax.set_ylabel("MRR@10")
    ax.set_title("Retrieval Performance by Legal Domain (Civil vs Criminal Code)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved code breakdown figure -> {out_path}")
    return out_path


def generate_all_figures(results_dir: Path = Path("results")) -> list[Path]:
    """Generate all figures and save to results/figures/."""
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_primary_metrics_bar(results_dir, figures_dir),
        plot_learning_curves(results_dir, figures_dir),
        plot_recall_at_k(results_dir, figures_dir),
        plot_code_breakdown(results_dir, figures_dir),
    ]
    return paths


if __name__ == "__main__":
    generate_all_figures()
