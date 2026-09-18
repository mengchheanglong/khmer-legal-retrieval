"""
Consolidated metrics reporting and analysis for Khmer legal retrieval benchmarks.

Loads evaluation metrics from results/metrics/*.json:
- BM25 Baseline (sparse lexical)
- Multilingual-E5 (zero-shot dense)
- Approach A1 (BiLSTM dual encoder from scratch)
- Approach A2 (XLM-R frozen dual encoder + linear probe)
- Approach A3 (XLM-R full fine-tuning dual encoder)

Outputs:
- results/metrics/summary.csv (full machine-readable metrics with 95% CIs)
- Formatted Markdown comparison tables for README.md Section 5.1
"""

import json
from pathlib import Path
import sys
from typing import Any, Optional

import pandas as pd

from src.config.logging import get_logger, setup_logging

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

setup_logging()
logger = get_logger(__name__)

MODELS_CONFIG = [
    {
        "id": "bm25",
        "name": "BM25 Baseline",
        "type": "Sparse Lexical",
        "metrics_file": "bm25_baseline.json",
        "train_file": None,
        "total_params": 0,
        "trainable_params": 0,
        "train_time_sec": 0.0,
        "hardware": "CPU",
    },
    {
        "id": "e5_zero_shot",
        "name": "Multilingual-E5 (Zero-Shot)",
        "type": "Pretrained Dense",
        "metrics_file": "e5_zero_shot.json",
        "train_file": None,
        "total_params": 278_043_648,
        "trainable_params": 0,
        "train_time_sec": 0.0,
        "hardware": "CPU",
    },
    {
        "id": "a1_bilstm",
        "name": "A1: BiLSTM Dual Encoder",
        "type": "From Scratch Dense",
        "metrics_file": "a1_bilstm.json",
        "train_file": "a1_lr3e-3_drop0.3_train.json",
        "total_params": 1_970_784,
        "trainable_params": 1_970_784,
        "train_time_sec": 2035.28,
        "hardware": "CPU",
    },
    {
        "id": "a2_linear_probe",
        "name": "A2: XLM-R Linear Probe",
        "type": "Frozen + Linear Head",
        "metrics_file": "a2_linear_probe.json",
        "train_file": "a2_lr1e-3_wd0.01_train.json",
        "total_params": 278_634_240,
        "trainable_params": 590_592,
        "train_time_sec": 5.92,
        "hardware": "CPU",
    },
    {
        "id": "a3_xlmr",
        "name": "A3: XLM-R Full Fine-Tuning",
        "type": "Full Fine-Tuning",
        "metrics_file": "a3_xlmr.json",
        "train_file": "a3_lr2e-5_wd0.01_train.json",
        "total_params": 278_043_648,
        "trainable_params": 278_043_648,
        "train_time_sec": 319.82,
        "hardware": "CPU (fp32)",
    },
    {
        "id": "a4_prahokbart",
        "name": "A4: PrahokBART Dual Encoder",
        "type": "Khmer Pretrained (Fine-Tuning)",
        "metrics_file": "a4_prahokbart.json",
        "train_file": "a4_lr1e-04_wd0.01_train.json",
        "total_params": 35_827_712,
        "trainable_params": 35_827_712,
        "train_time_sec": 0.0,
        "hardware": "CPU (fp32)",
    },
]


from src.dl.metrics import paired_bootstrap_test


def load_metric_file(path: Path) -> Optional[dict[str, Any]]:
    """Load and parse a JSON metrics file if it exists."""
    if not path.exists():
        logger.warning(f"Metrics file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_summary_table(metrics_dir: Path = Path("results/metrics")) -> pd.DataFrame:
    """Consolidate metrics across all approaches into a structured DataFrame."""
    rows: list[dict[str, Any]] = []

    for cfg in MODELS_CONFIG:
        m_file = metrics_dir / cfg["metrics_file"]
        data = load_metric_file(m_file)
        if not data:
            continue

        train_time = cfg["train_time_sec"]
        if cfg.get("train_file"):
            t_file = metrics_dir / cfg["train_file"]
            if t_file.exists():
                try:
                    with open(t_file, "r", encoding="utf-8") as f:
                        t_data = json.load(f)
                    train_time = t_data.get("total_wall_clock_seconds", train_time)
                except Exception:
                    pass

        tq = data.get("tq_benchmark", {})
        tt = data.get("tt_benchmark", {})

        tq_ov = tq.get("overall", {})
        tq_civ = tq.get("civil_code", {}).get("metrics", {})
        tq_crm = tq.get("criminal_code", {}).get("metrics", {})

        tt_ov = tt.get("overall", {})
        tt_civ = tt.get("civil_code", {}).get("metrics", {})
        tt_crm = tt.get("criminal_code", {}).get("metrics", {})

        row = {
            "model_id": cfg["id"],
            "model_name": cfg["name"],
            "architecture_type": cfg["type"],
            "total_params": cfg["total_params"],
            "trainable_params": cfg["trainable_params"],
            "train_time_sec": train_time,
            "hardware": cfg["hardware"],
            # T-Q Benchmark (Overall)
            "tq_recall@1": tq_ov.get("Recall@1", {}).get("mean"),
            "tq_recall@1_ci_lower": tq_ov.get("Recall@1", {}).get("ci_lower"),
            "tq_recall@1_ci_upper": tq_ov.get("Recall@1", {}).get("ci_upper"),
            "tq_recall@5": tq_ov.get("Recall@5", {}).get("mean"),
            "tq_recall@5_ci_lower": tq_ov.get("Recall@5", {}).get("ci_lower"),
            "tq_recall@5_ci_upper": tq_ov.get("Recall@5", {}).get("ci_upper"),
            "tq_recall@10": tq_ov.get("Recall@10", {}).get("mean"),
            "tq_recall@10_ci_lower": tq_ov.get("Recall@10", {}).get("ci_lower"),
            "tq_recall@10_ci_upper": tq_ov.get("Recall@10", {}).get("ci_upper"),
            "tq_mrr@10": tq_ov.get("MRR@10", {}).get("mean"),
            "tq_mrr@10_ci_lower": tq_ov.get("MRR@10", {}).get("ci_lower"),
            "tq_mrr@10_ci_upper": tq_ov.get("MRR@10", {}).get("ci_upper"),
            "tq_ndcg@10": tq_ov.get("nDCG@10", {}).get("mean"),
            "tq_ndcg@10_ci_lower": tq_ov.get("nDCG@10", {}).get("ci_lower"),
            "tq_ndcg@10_ci_upper": tq_ov.get("nDCG@10", {}).get("ci_upper"),
            "tq_hit@5": tq_ov.get("Hit@5", {}).get("mean"),
            "tq_hit@5_ci_lower": tq_ov.get("Hit@5", {}).get("ci_lower"),
            "tq_hit@5_ci_upper": tq_ov.get("Hit@5", {}).get("ci_upper"),
            # T-Q Per-Code Breakdown
            "tq_civil_recall@5": tq_civ.get("Recall@5", {}).get("mean"),
            "tq_civil_mrr@10": tq_civ.get("MRR@10", {}).get("mean"),
            "tq_criminal_recall@5": tq_crm.get("Recall@5", {}).get("mean"),
            "tq_criminal_mrr@10": tq_crm.get("MRR@10", {}).get("mean"),
            # T-T Benchmark (Overall)
            "tt_recall@1": tt_ov.get("Recall@1", {}).get("mean"),
            "tt_recall@1_ci_lower": tt_ov.get("Recall@1", {}).get("ci_lower"),
            "tt_recall@1_ci_upper": tt_ov.get("Recall@1", {}).get("ci_upper"),
            "tt_recall@5": tt_ov.get("Recall@5", {}).get("mean"),
            "tt_recall@5_ci_lower": tt_ov.get("Recall@5", {}).get("ci_lower"),
            "tt_recall@5_ci_upper": tt_ov.get("Recall@5", {}).get("ci_upper"),
            "tt_mrr@10": tt_ov.get("MRR@10", {}).get("mean"),
            "tt_mrr@10_ci_lower": tt_ov.get("MRR@10", {}).get("ci_lower"),
            "tt_mrr@10_ci_upper": tt_ov.get("MRR@10", {}).get("ci_upper"),
            "tt_hit@5": tt_ov.get("Hit@5", {}).get("mean"),
            "tt_hit@5_ci_lower": tt_ov.get("Hit@5", {}).get("ci_lower"),
            "tt_hit@5_ci_upper": tt_ov.get("Hit@5", {}).get("ci_upper"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_significance_table(metrics_dir: Path = Path("results/metrics")) -> tuple[pd.DataFrame, str]:
    """Compute paired bootstrap hypothesis tests between models on T-Q benchmark."""
    model_data: dict[str, dict[str, Any]] = {}
    for cfg in MODELS_CONFIG:
        m_file = metrics_dir / cfg["metrics_file"]
        data = load_metric_file(m_file)
        if data:
            model_data[cfg["id"]] = {
                "name": cfg["name"],
                "data": data,
            }

    comparisons = [
        ("a3_xlmr", "bm25", "A3 (Fine-Tuning) vs BM25 Baseline"),
        ("a3_xlmr", "e5_zero_shot", "A3 (Fine-Tuning) vs E5 Zero-Shot"),
        ("a3_xlmr", "a1_bilstm", "A3 (Fine-Tuning) vs A1 BiLSTM"),
        ("a3_xlmr", "a2_linear_probe", "A3 (Fine-Tuning) vs A2 Linear Probe"),
        ("a2_linear_probe", "bm25", "A2 (Linear Probe) vs BM25 Baseline"),
        ("a3_xlmr", "a4_prahokbart", "A3 (XLM-R) vs A4 (PrahokBART)"),
        ("a4_prahokbart", "bm25", "A4 (PrahokBART) vs BM25 Baseline"),
    ]

    metrics_to_test = ["Recall@5", "MRR@10"]
    rows: list[dict[str, Any]] = []

    for mod_a_id, mod_b_id, comp_label in comparisons:
        if mod_a_id not in model_data or mod_b_id not in model_data:
            continue

        data_a = model_data[mod_a_id]["data"].get("tq_benchmark", {})
        data_b = model_data[mod_b_id]["data"].get("tq_benchmark", {})

        scores_a_dict = data_a.get("per_query_scores", {})
        scores_b_dict = data_b.get("per_query_scores", {})

        for m in metrics_to_test:
            if m not in scores_a_dict or m not in scores_b_dict:
                continue

            test_res = paired_bootstrap_test(
                scores_a=scores_a_dict[m],
                scores_b=scores_b_dict[m],
                metric_name=m,
                n_bootstrap=10000,
                seed=42,
            )

            p_val = test_res["p_value"]
            if p_val < 0.001:
                sig_label = "*** (p < 0.001)"
            elif p_val < 0.01:
                sig_label = "** (p < 0.01)"
            elif p_val < 0.05:
                sig_label = "* (p < 0.05)"
            else:
                sig_label = "n.s. (not sig.)"

            rows.append({
                "comparison": comp_label,
                "metric": m,
                "mean_a": test_res["mean_a"],
                "mean_b": test_res["mean_b"],
                "delta": test_res["delta"],
                "ci_lower": test_res["ci_lower"],
                "ci_upper": test_res["ci_upper"],
                "p_value": test_res["p_value"],
                "significance": sig_label,
                "wins": test_res["wins"],
                "losses": test_res["losses"],
                "ties": test_res["ties"],
            })

    df_sig = pd.DataFrame(rows)

    # Format markdown
    lines: list[str] = []
    lines.append("### Paired Statistical Significance (T-Q Primary Benchmark, 10,000 Bootstrap Resamples)\n")
    lines.append(
        "| Comparison | Metric | Mean A | Mean B | Difference (Δ) [95% CI] | p-value | Significance | Wins / Losses / Ties |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for _, r in df_sig.iterrows():
        comp = r["comparison"]
        m = r["metric"]
        ma = f"{r['mean_a']:.4f}"
        mb = f"{r['mean_b']:.4f}"
        diff_str = f"**{r['delta']:+.4f}** [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
        pval = f"{r['p_value']:.4f}"
        sig = r["significance"]
        wlt = f"{r['wins']}W / {r['losses']}L / {r['ties']}T"
        lines.append(f"| {comp} | {m} | {ma} | {mb} | {diff_str} | {pval} | {sig} | {wlt} |")

    md_sig = "\n".join(lines)
    return df_sig, md_sig


def format_markdown_table(df: pd.DataFrame, sig_markdown: str = "") -> str:
    """Format DataFrame into GitHub-flavored markdown tables for README."""
    lines: list[str] = []

    lines.append("### Primary Benchmark (T-Q: 200 Human-Verified Questions)\n")
    lines.append(
        "| Approach / Model | Type | Trainable Params | Recall@1 [95% CI] | Recall@5 [95% CI] | MRR@10 [95% CI] | Hit@5 [95% CI] |"
    )
    lines.append(
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    )

    for _, r in df.iterrows():
        name = r["model_name"]
        atype = r["architecture_type"]
        params = f"{r['trainable_params'] / 1e6:.2f} M" if r["trainable_params"] > 0 else "0"
        r1 = f"**{r['tq_recall@1']:.4f}** [{r['tq_recall@1_ci_lower']:.4f}, {r['tq_recall@1_ci_upper']:.4f}]"
        r5 = f"**{r['tq_recall@5']:.4f}** [{r['tq_recall@5_ci_lower']:.4f}, {r['tq_recall@5_ci_upper']:.4f}]"
        mrr = f"**{r['tq_mrr@10']:.4f}** [{r['tq_mrr@10_ci_lower']:.4f}, {r['tq_mrr@10_ci_upper']:.4f}]"
        hit5 = f"**{r['tq_hit@5']:.4f}** [{r['tq_hit@5_ci_lower']:.4f}, {r['tq_hit@5_ci_upper']:.4f}]"
        lines.append(f"| {name} | {atype} | {params} | {r1} | {r5} | {mrr} | {hit5} |")

    lines.append("\n### Secondary Benchmark (T-T: 148 Held-Out Article Titles)\n")
    lines.append(
        "| Approach / Model | Type | Recall@1 [95% CI] | Recall@5 [95% CI] | MRR@10 [95% CI] | Hit@5 [95% CI] |"
    )
    lines.append(
        "| :--- | :--- | :---: | :---: | :---: | :---: |"
    )

    for _, r in df.iterrows():
        name = r["model_name"]
        atype = r["architecture_type"]
        r1 = f"**{r['tt_recall@1']:.4f}** [{r['tt_recall@1_ci_lower']:.4f}, {r['tt_recall@1_ci_upper']:.4f}]"
        r5 = f"**{r['tt_recall@5']:.4f}** [{r['tt_recall@5_ci_lower']:.4f}, {r['tt_recall@5_ci_upper']:.4f}]"
        mrr = f"**{r['tt_mrr@10']:.4f}** [{r['tt_mrr@10_ci_lower']:.4f}, {r['tt_mrr@10_ci_upper']:.4f}]"
        hit5 = f"**{r['tt_hit@5']:.4f}** [{r['tt_hit@5_ci_lower']:.4f}, {r['tt_hit@5_ci_upper']:.4f}]"
        lines.append(f"| {name} | {atype} | {r1} | {r5} | {mrr} | {hit5} |")

    lines.append("\n### Computational Efficiency & Resource Footprint\n")
    lines.append(
        "| Approach / Model | Total Params | Trainable Params | Training Time | Hardware |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :--- |"
    )

    for _, r in df.iterrows():
        name = r["model_name"]
        tot = f"{r['total_params'] / 1e6:.2f} M" if r["total_params"] > 0 else "0"
        trn = f"{r['trainable_params'] / 1e6:.2f} M" if r["trainable_params"] > 0 else "0"
        t_sec = f"{r['train_time_sec']:.1f} s" if r["train_time_sec"] > 0 else "0 s"
        hw = r["hardware"]
        lines.append(f"| {name} | {tot} | {trn} | {t_sec} | {hw} |")

    if sig_markdown:
        lines.append("\n" + sig_markdown)

    return "\n".join(lines)


def generate_report(results_dir: Path = Path("results")) -> Path:
    """Generate summary CSV and markdown tables."""
    metrics_dir = results_dir / "metrics"
    df = build_summary_table(metrics_dir)

    summary_csv = metrics_dir / "summary.csv"
    df.to_csv(summary_csv, index=False)
    logger.info(f"Saved consolidated summary table -> {summary_csv}")

    df_sig, sig_md = build_significance_table(metrics_dir)
    if not df_sig.empty:
        sig_csv = metrics_dir / "statistical_significance.csv"
        df_sig.to_csv(sig_csv, index=False)
        sig_md_file = metrics_dir / "statistical_significance.md"
        with open(sig_md_file, "w", encoding="utf-8") as f:
            f.write(sig_md)
        logger.info(f"Saved paired statistical significance report -> {sig_md_file}")

    md_content = format_markdown_table(df, sig_markdown=sig_md)
    summary_md = metrics_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved markdown report -> {summary_md}")

    print("\n" + "=" * 80)
    print("KHMER LEGAL RETRIEVAL STUDY: CONSOLIDATED LEADERBOARD & SIGNIFICANCE")
    print("=" * 80)
    print(md_content)
    print("=" * 80 + "\n")

    return summary_csv


if __name__ == "__main__":
    generate_report()

