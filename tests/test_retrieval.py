from __future__ import annotations

import numpy as np
from app.retrieval.faiss_store import FaissVectorStore


def test_vector_store_add_and_search(tmp_path):
    store = FaissVectorStore(str(tmp_path / "test_idx"), dimension=4)
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
    v3 = np.array([0.7071, 0.7071, 0.0, 0.0], dtype="float32")

    store.add("mem_1", v1)
    store.add("mem_2", v2)
    store.add("mem_3", v3)

    # Search with v1
    results = store.search(v1, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "mem_1"
    assert np.isclose(results[0][1], 1.0, atol=1e-3)
    assert results[1][0] == "mem_3"


def test_vector_store_remove(tmp_path):
    store = FaissVectorStore(str(tmp_path / "test_idx_del"), dimension=3)
    v1 = np.array([1, 0, 0], dtype="float32")
    v2 = np.array([0, 1, 0], dtype="float32")

    store.add("a", v1)
    store.add("b", v2)
    assert len(store.ids) == 2

    # Remove 'a'
    removed = store.remove("a")
    assert removed is True
    assert len(store.ids) == 1
    assert "a" not in store.ids

    # Search should no longer return 'a'
    results = store.search(v1, top_k=5)
    assert all(r[0] != "a" for r in results)


def test_vector_store_rebuild(tmp_path):
    store = FaissVectorStore(str(tmp_path / "test_idx_rebuild"), dimension=3)
    active_embeddings = {
        "id_x": np.array([0, 0, 1], dtype="float32"),
        "id_y": np.array([0, 1, 0], dtype="float32"),
    }
    store.rebuild(active_embeddings)
    assert len(store.ids) == 2
    results = store.search(np.array([0, 0, 1], dtype="float32"), top_k=1)
    assert results[0][0] == "id_x"


def test_vector_store_consistency_errors(tmp_path):
    store = FaissVectorStore(str(tmp_path / "test_idx_cons"), dimension=3)
    store.add("valid_id", np.array([1, 0, 0], dtype="float32"))
    store.add("stale_id", np.array([0, 1, 0], dtype="float32"))

    active_ids = {"valid_id"}
    errors = store.consistency_errors(active_ids)
    assert errors == ["stale_id"]
