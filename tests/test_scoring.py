from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from app.core.config import Settings
from app.memory.schemas import MemoryRecord
from app.memory.scoring import WeightedScorer, recency_score, utility_score


def test_recency_score_exponential_decay():
    now = datetime.now(timezone.utc)
    decay = 0.03
    score_fresh = recency_score(now, decay)
    assert np_isclose(score_fresh, 1.0, atol=0.01)

    ten_days_ago = now - timedelta(days=10)
    score_10d = recency_score(ten_days_ago, decay)
    # exp(-0.03 * 10) = exp(-0.3) ~= 0.7408
    assert 0.70 <= score_10d <= 0.78

    thirty_days_ago = now - timedelta(days=30)
    score_30d = recency_score(thirty_days_ago, decay)
    assert score_30d < score_10d


def test_utility_score_growth_with_access():
    now = datetime.now(timezone.utc)
    mem_zero = MemoryRecord(id="m0", text="test", created_at=now, updated_at=now, access_count=0)
    mem_accessed = MemoryRecord(id="m5", text="test", created_at=now, updated_at=now, access_count=5)

    u0 = utility_score(mem_zero)
    u5 = utility_score(mem_accessed)
    assert u0 == 0.0
    assert u5 > u0


def test_weighted_scorer_composite():
    settings = Settings(
        relevance_weight=0.30,
        recency_weight=0.15,
        confidence_weight=0.20,
        redundancy_weight=0.25,
        utility_weight=0.10,
    )
    scorer = WeightedScorer(settings)

    # Clean non-redundant candidate
    bundle_good = scorer.score(relevance=0.8, memory_confidence=0.9, redundancy=0.0, utility=0.5, recency=1.0)
    # Highly redundant candidate
    bundle_dup = scorer.score(relevance=0.8, memory_confidence=0.9, redundancy=0.95, utility=0.5, recency=1.0)

    assert bundle_good.overall > bundle_dup.overall
    assert 0.0 <= bundle_good.overall <= 1.0
    assert 0.0 <= bundle_dup.overall <= 1.0


def test_config_threshold_validation():
    # Valid ordering should pass
    Settings(forget_threshold=0.2, archive_threshold=0.4, redundancy_threshold=0.8)

    # Invalid ordering (forget > archive) must raise ValueError
    with pytest.raises(ValueError, match="Invalid threshold ordering"):
        Settings(forget_threshold=0.6, archive_threshold=0.3)

    # Negative weight must raise ValueError
    with pytest.raises(ValueError, match="All scoring weights must be non-negative"):
        Settings(relevance_weight=-0.1)


def np_isclose(a, b, atol=1e-3):
    return abs(a - b) <= atol
