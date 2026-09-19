from __future__ import annotations

from typing import Sequence
import numpy as np


def precision_at_k(
    retrieved_ids: Sequence[str],
    expected_ids: Sequence[str],
    k: int,
) -> float | None:
    """
    Precision@k: proportion of retrieved top-k items that are relevant.
    Returns None if expected_ids is empty (ground truth unavailable).
    """
    if not expected_ids:
        return None
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = len(set(top_k) & set(expected_ids))
    return relevant_retrieved / float(len(top_k))


def recall_at_k(
    retrieved_ids: Sequence[str],
    expected_ids: Sequence[str],
    k: int,
) -> float | None:
    """
    Recall@k: proportion of relevant items that are retrieved in top-k.
    Returns None if expected_ids is empty (ground truth unavailable).
    """
    if not expected_ids:
        return None
    top_k = retrieved_ids[:k]
    relevant_retrieved = len(set(top_k) & set(expected_ids))
    return relevant_retrieved / float(len(set(expected_ids)))


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    expected_ids: Sequence[str],
) -> float | None:
    """
    Reciprocal Rank (RR): reciprocal of the 1-indexed position of the first relevant item.
    Returns None if expected_ids is empty. Returns 0.0 if no relevant item is retrieved.
    """
    if not expected_ids:
        return None
    expected_set = set(expected_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in expected_set:
            return 1.0 / float(rank)
    return 0.0


def f1_at_k(
    retrieved_ids: Sequence[str],
    expected_ids: Sequence[str],
    k: int,
) -> float | None:
    """Harmonic mean of Precision@k and Recall@k."""
    p = precision_at_k(retrieved_ids, expected_ids, k)
    r = recall_at_k(retrieved_ids, expected_ids, k)
    if p is None or r is None:
        return None
    if p + r == 0:
        return 0.0
    return 2.0 * (p * r) / (p + r)


def memory_reduction_percentage(baseline_active: float, lamm_active: float) -> float:
    """
    Calculates measured percentage reduction in active memories:
        ((Baseline - LAMM) / Baseline) * 100
    """
    if baseline_active <= 0:
        return 0.0
    return ((baseline_active - lamm_active) / float(baseline_active)) * 100.0


def token_reduction_percentage(baseline_tokens: float, lamm_tokens: float) -> float:
    """Calculates measured percentage reduction in context tokens."""
    if baseline_tokens <= 0:
        return 0.0
    return ((baseline_tokens - lamm_tokens) / float(baseline_tokens)) * 100.0


def latency_change_percentage(baseline_latency: float, lamm_latency: float) -> float:
    """Calculates measured percentage change in latency: ((LAMM - Baseline) / Baseline) * 100."""
    if baseline_latency <= 0:
        return 0.0
    return ((lamm_latency - baseline_latency) / float(baseline_latency)) * 100.0


def compute_latency_stats(latencies: Sequence[float]) -> dict[str, float]:
    """Summary statistics for a series of latency measurements."""
    if not latencies:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(latencies)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
