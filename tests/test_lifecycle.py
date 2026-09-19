from __future__ import annotations

from datetime import datetime, timezone

from app.core.config import Settings
from app.memory.lifecycle import LammLifecycleController
from app.memory.schemas import (
    CandidateMemory,
    LifecycleOperation,
    MemoryRecord,
    RetrievedMemory,
)


def make_controller(**kwargs) -> LammLifecycleController:
    settings = Settings(
        gemini_api_key="",
        embedding_provider="hash",
        embedding_model="deterministic-hash",
        **kwargs,
    )
    return LammLifecycleController(settings)


def test_lifecycle_1_retain():
    """Operation 1: RETAIN for informative, non-redundant, sufficient confidence candidate."""
    controller = make_controller()
    candidate = CandidateMemory(
        text="User prefers Python for backend development.",
        confidence_score=0.92,
    )
    decision = controller.decide(candidate, similar_memories=[], active_memory_count=0)
    assert decision.operation == LifecycleOperation.RETAIN
    assert "useful" in decision.reason.lower() or "retain" in decision.reason.lower()


def test_lifecycle_2_update():
    """Operation 2: UPDATE when candidate exhibits temporal supersession over existing memory."""
    controller = make_controller()
    now = datetime.now(timezone.utc)
    existing = MemoryRecord(
        id="mem_backend_001",
        text="User prefers Python for backend development.",
        created_at=now,
        updated_at=now,
    )
    similar = [RetrievedMemory(memory=existing, score=0.75)]
    candidate = CandidateMemory(
        text="User has started using Java for my current backend project.",
        confidence_score=0.90,
    )
    decision = controller.decide(candidate, similar_memories=similar, active_memory_count=1)
    assert decision.operation == LifecycleOperation.UPDATE
    assert "mem_backend_001" in decision.affected_memory_ids
    assert "supersedes" in decision.reason.lower() or "modifies" in decision.reason.lower()


def test_lifecycle_3_merge():
    """Operation 3: MERGE when candidate is semantically redundant with an existing memory."""
    controller = make_controller()
    now = datetime.now(timezone.utc)
    existing = MemoryRecord(
        id="mem_lang_001",
        text="User prefers Python for programming.",
        created_at=now,
        updated_at=now,
    )
    similar = [RetrievedMemory(memory=existing, score=0.92)]
    candidate = CandidateMemory(
        text="Python is the user's preferred language for programming.",
        confidence_score=0.88,
    )
    decision = controller.decide(candidate, similar_memories=similar, active_memory_count=1)
    assert decision.operation == LifecycleOperation.MERGE
    assert "mem_lang_001" in decision.affected_memory_ids
    assert "redundant" in decision.reason.lower()


def test_lifecycle_4_compress():
    """Operation 4: COMPRESS when candidate length exceeds configured compression threshold."""
    controller = make_controller(compress_length=100)
    verbose_text = (
        "User is currently developing an extensive architecture for lightweight adaptive memory management "
        "incorporating FAISS vector search, SQLite metadata persistence, automated benchmark evaluation, "
        "and interactive Streamlit demonstration dashboards."
    )
    candidate = CandidateMemory(text=verbose_text, confidence_score=0.85)
    decision = controller.decide(candidate, similar_memories=[], active_memory_count=0)
    assert decision.operation == LifecycleOperation.COMPRESS
    assert "exceeds compression threshold" in decision.reason.lower()


def test_lifecycle_5_archive_low_confidence():
    """Operation 5A: ARCHIVE when candidate confidence is below active threshold."""
    controller = make_controller()
    candidate = CandidateMemory(
        text="Possible speculative preference: user might be interested in Rust.",
        confidence_score=0.40,
    )
    decision = controller.decide(candidate, similar_memories=[], active_memory_count=0)
    assert decision.operation == LifecycleOperation.ARCHIVE
    assert "confidence" in decision.reason.lower()


def test_lifecycle_5_archive_capacity_limit():
    """Operation 5B: ARCHIVE when active memory pool exceeds capacity constraint."""
    controller = make_controller(max_active_memories=3)
    candidate = CandidateMemory(
        text="User enjoys reading machine learning papers on weekends.",
        confidence_score=0.75,
    )
    decision = controller.decide(candidate, similar_memories=[], active_memory_count=3)
    assert decision.operation == LifecycleOperation.ARCHIVE
    assert "capacity" in decision.reason.lower()


def test_lifecycle_6_forget_ephemeral():
    """Operation 6: FORGET when candidate contains ephemeral filler or temporary info."""
    controller = make_controller()
    candidate = CandidateMemory(
        text="This is temporary filler information that is not very useful for long term memory.",
        confidence_score=0.30,
    )
    decision = controller.decide(candidate, similar_memories=[], active_memory_count=0)
    assert decision.operation == LifecycleOperation.FORGET
    assert "ephemeral" in decision.reason.lower() or "forget" in decision.reason.lower()


def test_decision_priority_update_overrides_low_score():
    """
    Architectural Invariant: Semantic relationships (UPDATE, MERGE) must be evaluated
    before score-based eviction. An incoming update is never discarded as FORGET
    just because of a low composite score.
    """
    controller = make_controller()
    now = datetime.now(timezone.utc)
    existing = MemoryRecord(
        id="mem_old_001",
        text="User is working on a legacy python backend.",
        created_at=now,
        updated_at=now,
    )
    similar = [RetrievedMemory(memory=existing, score=0.70)]
    # Candidate has low relevance to current general context, but is a clear update to existing memory
    candidate = CandidateMemory(
        text="User has switched to Java for the current backend project now.",
        confidence_score=0.60,
    )
    decision = controller.decide(candidate, similar_memories=similar, active_memory_count=5, context_relevance=0.1)
    assert decision.operation == LifecycleOperation.UPDATE
    assert "mem_old_001" in decision.affected_memory_ids
