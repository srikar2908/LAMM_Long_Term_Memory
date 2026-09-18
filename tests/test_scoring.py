from datetime import datetime, timezone

from app.core.config import Settings
from app.memory.scoring import WeightedScorer, recency_score


def test_recency_score_is_normalized():
    assert 0 <= recency_score(datetime.now(timezone.utc), 0.03) <= 1


def test_weighted_score_penalizes_redundancy():
    scorer = WeightedScorer(Settings())
    low = scorer.score(0.8, 0.8, 0.9, 0.1)
    high = scorer.score(0.8, 0.8, 0.0, 0.1)
    assert high.overall > low.overall
