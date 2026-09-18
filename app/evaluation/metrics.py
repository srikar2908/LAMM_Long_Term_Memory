from __future__ import annotations


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float | None:
    if not expected_ids:
        return None
    return len(set(retrieved_ids[:k]) & set(expected_ids)) / max(1, k)
