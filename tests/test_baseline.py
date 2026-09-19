from __future__ import annotations

from app.core.config import Settings
from app.evaluation.baseline import RecencyOnlyBaseline, UnmanagedMemoryBaseline
from app.memory.manager import MemoryManager


def test_unmanaged_baseline_unbounded_storage(tmp_path):
    settings = Settings(
        gemini_api_key="",
        database_url=f"sqlite:///{tmp_path / 'unm.db'}",
        faiss_index_path=str(tmp_path / "unm_idx"),
        embedding_provider="hash",
        embedding_model="deterministic-hash",
    )
    manager = MemoryManager(settings)
    baseline = UnmanagedMemoryBaseline(manager)

    # Ingest duplicate and temporary facts
    baseline.ingest("I prefer Python for backend development.")
    baseline.ingest("Python is my preferred backend language.")
    baseline.ingest("This is temporary filler.")

    stats = manager.stats()
    # Unmanaged stores all 3 facts as active
    assert stats["total"] == 3
    assert stats["active"] == 3
    assert stats["archived"] == 0
    assert stats["forgotten"] == 0


def test_recency_only_baseline_lru_eviction(tmp_path):
    budget = 2
    settings = Settings(
        gemini_api_key="",
        database_url=f"sqlite:///{tmp_path / 'rec.db'}",
        faiss_index_path=str(tmp_path / "rec_idx"),
        embedding_provider="hash",
        embedding_model="deterministic-hash",
        recency_baseline_budget=budget,
    )
    manager = MemoryManager(settings)
    baseline = RecencyOnlyBaseline(manager, budget=budget)

    # Turn 1: Add fact 1
    res1 = baseline.ingest("I prefer Python for backend development.")
    assert manager.stats()["active"] == 1

    # Turn 2: Add fact 2
    res2 = baseline.ingest("I am working on an AI project.")
    assert manager.stats()["active"] == 2

    # Turn 3: Add fact 3 -> Exceeds budget 2 -> should evict LRU (fact 1)
    res3 = baseline.ingest("My project uses long-term LLM memory.")
    stats = manager.stats()
    assert stats["total"] == 3
    assert stats["active"] == 2
    assert stats["forgotten"] == 1

    # Fact 1 should no longer be active in FAISS
    retrieved = manager.retrieve("backend development")
    retrieved_texts = [r.memory.text for r in retrieved]
    assert "User prefers Python for backend development." not in retrieved_texts
