from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import LammAgent
from app.api import routes_chat, routes_memory
from app.core.config import Settings
from app.main import app
from app.memory.manager import MemoryManager

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_api_env(tmp_path):
    """Ensures each API test has a clean, isolated SQLite database and FAISS index."""
    test_db = tmp_path / "api_test.db"
    test_idx = tmp_path / "api_test_vector_index"
    test_settings = Settings(
        gemini_api_key="",
        database_url=f"sqlite:///{test_db}",
        faiss_index_path=str(test_idx),
        embedding_provider="hash",
        embedding_model="deterministic-hash",
    )
    new_manager = MemoryManager(test_settings)
    routes_memory.manager = new_manager
    routes_chat.manager = new_manager
    routes_chat.agent = LammAgent(test_settings, manager=new_manager)
    yield


def test_api_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


def test_api_chat():
    resp = client.post("/chat", json={"message": "I prefer Python for backend development.", "conversation_id": "test_conv"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "retrieved_memories" in data
    assert "lifecycle_decisions" in data
    assert "stats" in data


def test_api_memory_lifecycle_and_crud():
    # 1. POST /memory (first insertion in clean database -> RETAIN)
    create_resp = client.post(
        "/memory",
        json={"text": "User prefers Linux for production servers.", "confidence_score": 0.90, "conversation_id": "api_test"},
    )
    assert create_resp.status_code == 200
    created_data = create_resp.json()
    mem_id = created_data["memory"]["id"]
    assert created_data["decision"]["operation"] == "RETAIN"

    # 2. GET /memories
    list_resp = client.get("/memories")
    assert list_resp.status_code == 200
    all_mems = list_resp.json()
    assert any(m["id"] == mem_id for m in all_mems)

    # 3. GET /memory/{id}
    get_resp = client.get(f"/memory/{mem_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == mem_id

    # 4. GET /memory/search
    search_resp = client.get("/memory/search?q=production+servers&top_k=3")
    assert search_resp.status_code == 200
    results = search_resp.json()
    assert len(results) >= 1

    # 5. GET /memory/stats
    stats_resp = client.get("/memory/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["active"] >= 1

    # 6. POST /memory/rebuild-index
    rebuild_resp = client.post("/memory/rebuild-index")
    assert rebuild_resp.status_code == 200
    assert rebuild_resp.json()["status"] == "rebuilt"

    # 7. GET /lifecycle/events
    events_resp = client.get("/lifecycle/events")
    assert events_resp.status_code == 200
    assert len(events_resp.json()) >= 1

    # 8. DELETE /memory/{id} (privacy deletion)
    del_resp = client.delete(f"/memory/{mem_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "forgotten"


def test_api_evaluation_endpoints():
    # POST /evaluation/run
    run_resp = client.post("/evaluation/run")
    assert run_resp.status_code == 200
    data = run_resp.json()
    assert "final_memory_counts" in data
    assert "comparative_metrics" in data

    # GET /evaluation/results
    results_resp = client.get("/evaluation/results")
    assert results_resp.status_code == 200
    res_data = results_resp.json()
    assert "final_memory_counts" in res_data
