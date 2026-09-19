from __future__ import annotations

from datetime import datetime, timezone
import numpy as np

from app.memory.schemas import LifecycleDecision, LifecycleOperation, LifecycleStatus, MemoryRecord, ScoreBundle
from app.storage.database import connect
from app.storage.repositories import LifecycleEventRepository, MemoryRepository


def test_memory_repository_crud(tmp_path):
    conn = connect(tmp_path / "test_storage.db")
    repo = MemoryRepository(conn, dimension=3)
    now = datetime.now(timezone.utc)

    record = MemoryRecord(
        id="mem_1",
        text="User prefers Python for backend development.",
        created_at=now,
        updated_at=now,
        confidence_score=0.9,
    )
    vec = np.array([0.5, 0.5, 0.7071], dtype="float32")
    repo.create(record, vec)

    # Retrieval
    fetched = repo.get("mem_1")
    assert fetched is not None
    assert fetched.text == "User prefers Python for backend development."
    assert fetched.confidence_score == 0.9

    # Vector retrieval
    stored_vec = repo.get_embedding("mem_1")
    assert stored_vec is not None
    assert stored_vec.shape == (3,)
    assert np.allclose(stored_vec, vec)

    # Update
    fetched.text = "User prefers Java for backend development."
    repo.update(fetched)
    updated = repo.get("mem_1")
    assert updated.text == "User prefers Java for backend development."


def test_lru_lookup_and_mark_accessed(tmp_path):
    conn = connect(tmp_path / "test_lru.db")
    repo = MemoryRepository(conn, dimension=2)
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)

    m1 = MemoryRecord(id="old_mem", text="Old fact", created_at=t1, updated_at=t1)
    m2 = MemoryRecord(id="new_mem", text="New fact", created_at=t2, updated_at=t2)
    repo.create(m1, np.array([1, 0], dtype="float32"))
    repo.create(m2, np.array([0, 1], dtype="float32"))

    # Least recently accessed should be m1
    lru = repo.get_least_recently_used()
    assert lru is not None
    assert lru.id == "old_mem"

    # Access m1
    repo.mark_accessed("old_mem", relevance=0.95)
    # Now m2 was accessed less recently
    lru_after = repo.get_least_recently_used()
    assert lru_after is not None
    assert lru_after.id == "new_mem"


def test_lifecycle_event_recording(tmp_path):
    conn = connect(tmp_path / "test_events.db")
    events_repo = LifecycleEventRepository(conn)

    scores = ScoreBundle(relevance=0.9, recency=1.0, confidence=0.85, redundancy=0.2, utility=0.5, overall=0.78)
    decision = LifecycleDecision(
        operation=LifecycleOperation.UPDATE,
        reason="Updated previous backend language fact.",
        scores=scores,
        affected_memory_ids=["prior_mem_id"],
    )
    event_id = events_repo.record(
        memory_id="new_mem_id",
        decision=decision,
        before_state={"text": "User prefers Python."},
        after_state={"text": "User prefers Java."},
        thresholds={"redundancy_threshold": 0.86},
    )
    assert event_id is not None
    history = events_repo.list()
    assert len(history) == 1
    assert history[0]["operation"] == "UPDATE"
    assert "Java" in history[0]["after_state"]
