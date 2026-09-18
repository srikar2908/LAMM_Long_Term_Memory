from __future__ import annotations

import math
from datetime import datetime, timezone

from app.core.config import Settings
from app.memory.schemas import MemoryRecord, ScoreBundle


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def recency_score(created_or_updated: datetime, decay: float) -> float:
    age_days = max(0.0, (datetime.now(timezone.utc) - created_or_updated).total_seconds() / 86400)
    return clamp(math.exp(-decay * age_days))


def utility_score(memory: MemoryRecord) -> float:
    access_component = 1.0 - math.exp(-0.25 * memory.access_count)
    return clamp(max(memory.utility_score or 0.0, access_component))


class WeightedScorer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def score(self, relevance: float, memory_confidence: float, redundancy: float, utility: float, recency: float = 1.0) -> ScoreBundle:
        raw = (
            self.settings.relevance_weight * clamp(relevance)
            + self.settings.recency_weight * clamp(recency)
            + self.settings.confidence_weight * clamp(memory_confidence)
            + self.settings.utility_weight * clamp(utility)
            - self.settings.redundancy_weight * clamp(redundancy)
        )
        weight_sum = (
            self.settings.relevance_weight
            + self.settings.recency_weight
            + self.settings.confidence_weight
            + self.settings.utility_weight
        )
        return ScoreBundle(
            relevance=clamp(relevance),
            recency=clamp(recency),
            confidence=clamp(memory_confidence),
            redundancy=clamp(redundancy),
            utility=clamp(utility),
            overall=clamp(raw / max(weight_sum, 1e-9)),
        )
