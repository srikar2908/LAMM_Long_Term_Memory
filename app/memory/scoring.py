from __future__ import annotations

import math
from datetime import datetime, timezone

from app.core.config import Settings
from app.memory.schemas import MemoryRecord, ScoreBundle


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def recency_score(created_or_updated: datetime, decay: float) -> float:
    """
    Exponential recency decay function:
        S_recency = exp(-lambda * age_days)
    Age is measured in fractional days from last update/creation to now.
    """
    now = datetime.now(timezone.utc)
    if created_or_updated.tzinfo is None:
        created_or_updated = created_or_updated.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - created_or_updated).total_seconds() / 86400.0)
    return clamp(math.exp(-decay * age_days))


def utility_score(memory: MemoryRecord) -> float:
    """
    Utility score combining cumulative access count and past retrieval feedback:
        S_utility = max(prior_utility, 1 - exp(-0.25 * access_count))
    """
    access_component = 1.0 - math.exp(-0.25 * memory.access_count)
    prior = memory.utility_score or 0.0
    return clamp(max(prior, access_component))


class WeightedScorer:
    """
    Computes a configurable heuristic composite score across memory signals:
    - Relevance (semantic similarity to current query/context)
    - Recency (exponential decay over time)
    - Confidence (extraction certainty)
    - Utility (past access frequency and retrieval usefulness)
    - Redundancy (penalty for overlapping with existing active memories)

    NOTE: This is an engineering heuristic composite score designed for adaptive
    memory management, not a theoretically optimal or scientifically proven closed form.
    Weights are configurable parameters to be calibrated experimentally.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def score(
        self,
        relevance: float,
        memory_confidence: float,
        redundancy: float,
        utility: float,
        recency: float = 1.0,
    ) -> ScoreBundle:
        rel = clamp(relevance)
        rec = clamp(recency)
        conf = clamp(memory_confidence)
        red = clamp(redundancy)
        util = clamp(utility)

        raw = (
            self.settings.relevance_weight * rel
            + self.settings.recency_weight * rec
            + self.settings.confidence_weight * conf
            + self.settings.utility_weight * util
            - self.settings.redundancy_weight * red
        )
        positive_weight_sum = (
            self.settings.relevance_weight
            + self.settings.recency_weight
            + self.settings.confidence_weight
            + self.settings.utility_weight
        )
        overall = clamp(raw / max(positive_weight_sum, 1e-9))

        return ScoreBundle(
            relevance=rel,
            recency=rec,
            confidence=conf,
            redundancy=red,
            utility=util,
            overall=overall,
        )
