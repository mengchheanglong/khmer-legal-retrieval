"""
Information Retrieval (IR) metrics and statistical confidence interval evaluation.

Implements standard ranking metrics for Khmer legal retrieval benchmarks:
- Recall@1, Recall@5, Recall@10
- MRR@10 (Mean Reciprocal Rank)
- nDCG@10 (Normalized Discounted Cumulative Gain)
- Hit@5
- 1,000-sample 95% bootstrap confidence intervals for all metrics
"""

import math
from typing import Any, Sequence

import numpy as np


def compute_recall_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """Compute Recall@k for a single query.

    Recall@k = |Retrieved@k ∩ Relevant| / |Relevant|

    Args:
        ranked_doc_ids: Ordered list of retrieved document IDs.
        relevant_doc_ids: Set of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    hits = len(top_k.intersection(relevant_doc_ids))
    return hits / len(relevant_doc_ids)


def compute_hit_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """Compute Hit@k (Success@k) for a single query.

    Hit@k = 1.0 if at least one relevant document is in top-k, else 0.0.

    Args:
        ranked_doc_ids: Ordered list of retrieved document IDs.
        relevant_doc_ids: Set of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        1.0 if hit else 0.0.
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    return 1.0 if len(top_k.intersection(relevant_doc_ids)) > 0 else 0.0


def compute_mrr_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Reciprocal Rank at k (RR@k) for a single query.

    RR@k = 1 / rank of the first relevant document in top-k (1-indexed).

    Args:
        ranked_doc_ids: Ordered list of retrieved document IDs.
        relevant_doc_ids: Set of ground-truth relevant document IDs.
        k: Maximum rank cutoff (default: 10).

    Returns:
        Reciprocal rank score in [0.0, 1.0].
    """
    if not relevant_doc_ids:
        return 0.0
    for rank_idx, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant_doc_ids:
            return 1.0 / (rank_idx + 1)
    return 0.0


def compute_ndcg_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k (nDCG@k) for a single query.

    For binary relevance (1 if relevant, 0 if not):
        DCG@k = sum_{i=1}^k rel_i / log2(i + 1)
        IDCG@k = sum_{i=1}^{min(k, |Relevant|)} 1 / log2(i + 1)
        nDCG@k = DCG@k / IDCG@k

    Args:
        ranked_doc_ids: Ordered list of retrieved document IDs.
        relevant_doc_ids: Set of ground-truth relevant document IDs.
        k: Rank cutoff (default: 10).

    Returns:
        nDCG score in [0.0, 1.0].
    """
    if not relevant_doc_ids:
        return 0.0

    dcg = 0.0
    for rank_idx, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(rank_idx + 2)

    idcg = 0.0
    num_ideal = min(k, len(relevant_doc_ids))
    for rank_idx in range(num_ideal):
        idcg += 1.0 / math.log2(rank_idx + 2)

    return (dcg / idcg) if idcg > 0.0 else 0.0


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Calculate the mean score and non-parametric bootstrap confidence interval.

    Args:
        scores: List of per-query metric scores.
        n_bootstrap: Number of bootstrap resamples (default: 1,000).
        ci: Confidence level in (0, 1) (default: 0.95).
        seed: Random seed for reproducible resampling (default: 42).

    Returns:
        Dictionary with 'mean', 'ci_lower', and 'ci_upper'.
    """
    if not scores:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    arr = np.array(scores, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))

    if n < 2 or np.all(arr == arr[0]):
        return {
            "mean": round(mean_val, 4),
            "ci_lower": round(mean_val, 4),
            "ci_upper": round(mean_val, 4),
        }

    rng = np.random.default_rng(seed)
    # Generate bootstrap indices: (n_bootstrap, n)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(arr[boot_indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    return {
        "mean": round(mean_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


def evaluate_rankings(
    rankings: list[list[str]],
    ground_truth: list[set[str]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute the full suite of retrieval metrics with 95% bootstrap CIs.

    Metrics computed:
    - Recall@1
    - Recall@5
    - Recall@10
    - MRR@10
    - nDCG@10
    - Hit@5

    Args:
        rankings: List of retrieved doc_id lists (one ranked list per query).
        ground_truth: List of sets of expected doc_ids (one set per query).
        n_bootstrap: Number of bootstrap iterations for CI (default: 1,000).
        seed: Random seed for bootstrap reproducibility (default: 42).

    Returns:
        Dictionary mapping metric names to {'mean', 'ci_lower', 'ci_upper'}.
    """
    if len(rankings) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: rankings ({len(rankings)}) vs "
            f"ground_truth ({len(ground_truth)})"
        )

    num_queries = len(rankings)
    if num_queries == 0:
        return {}

    per_query_scores: dict[str, list[float]] = {
        "Recall@1": [],
        "Recall@5": [],
        "Recall@10": [],
        "MRR@10": [],
        "nDCG@10": [],
        "Hit@5": [],
    }

    for rank_list, target_set in zip(rankings, ground_truth):
        per_query_scores["Recall@1"].append(compute_recall_at_k(rank_list, target_set, k=1))
        per_query_scores["Recall@5"].append(compute_recall_at_k(rank_list, target_set, k=5))
        per_query_scores["Recall@10"].append(compute_recall_at_k(rank_list, target_set, k=10))
        per_query_scores["MRR@10"].append(compute_mrr_at_k(rank_list, target_set, k=10))
        per_query_scores["nDCG@10"].append(compute_ndcg_at_k(rank_list, target_set, k=10))
        per_query_scores["Hit@5"].append(compute_hit_at_k(rank_list, target_set, k=5))

    results: dict[str, dict[str, float]] = {}
    for metric_name, score_list in per_query_scores.items():
        results[metric_name] = bootstrap_ci(
            score_list,
            n_bootstrap=n_bootstrap,
            ci=0.95,
            seed=seed,
        )

    return results
