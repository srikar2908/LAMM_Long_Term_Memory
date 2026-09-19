from __future__ import annotations

from app.agent.agent import LammAgent
from app.core.config import Settings
from app.memory.schemas import CandidateMemory, LifecycleOperation


def test_end_to_end_mock_pipeline_with_update_and_recall(tmp_path):
    settings = Settings(
        gemini_api_key="",
        database_url=f"sqlite:///{tmp_path / 'e2e.db'}",
        faiss_index_path=str(tmp_path / "e2e_idx"),
        embedding_provider="hash",
        embedding_model="deterministic-hash",
    )
    agent = LammAgent(settings)

    # 1. Turn 1: Initial preference (RETAIN)
    t1 = agent.chat("I prefer Python for backend development.", conversation_id="e2e_conv")
    assert any(d["operation"] == "RETAIN" for d in t1.lifecycle_decisions)
    assert agent.manager.stats()["active"] == 1

    # 2. Turn 2: Redundant preference (MERGE or RETAIN)
    t2 = agent.chat("Python is my preferred backend language.", conversation_id="e2e_conv")
    assert agent.manager.stats()["active"] >= 1

    # 3. Turn 3: Fact update (UPDATE)
    t3 = agent.chat("I have started using Java for my current backend project.", conversation_id="e2e_conv")
    assert any(d["operation"] == "UPDATE" for d in t3.lifecycle_decisions)

    # 4. Turn 4: Ephemeral filler (FORGET)
    t4 = agent.chat("This is temporary filler that is not very useful.", conversation_id="e2e_conv")
    assert any(d["operation"] == "FORGET" for d in t4.lifecycle_decisions)

    # 5. Query: Verifying retrieval of updated fact
    query = "What programming language am I currently using for my backend project?"
    t_query = agent.chat(query, conversation_id="e2e_conv")

    # Assert retrieved memories prioritize the updated fact (Java)
    top_memory_text = t_query.retrieved_memories[0]["text"]
    assert "Java" in top_memory_text
    assert "Java" in t_query.response


def test_dry_run_mode(tmp_path):
    settings = Settings(
        gemini_api_key="",
        database_url=f"sqlite:///{tmp_path / 'dry.db'}",
        faiss_index_path=str(tmp_path / "dry_idx"),
        embedding_provider="hash",
        embedding_model="deterministic-hash",
        dry_run=True,
    )
    agent = LammAgent(settings)
    candidate = CandidateMemory(text="User likes Kotlin for Android.", confidence_score=0.9)

    decision, record = agent.manager.process_candidate(candidate, dry_run=True)
    assert decision.operation == LifecycleOperation.RETAIN
    assert record.metadata.get("dry_run") is True

    # Database must remain empty in dry run
    assert agent.manager.stats()["total"] == 0
    assert len(agent.manager.vector_store.ids) == 0
