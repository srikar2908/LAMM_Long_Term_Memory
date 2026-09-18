import numpy as np

from app.retrieval.faiss_store import FaissVectorStore


def test_vector_store_search(tmp_path):
    store = FaissVectorStore(str(tmp_path / "index"), 3)
    store.add("a", np.array([1, 0, 0], dtype="float32"))
    store.add("b", np.array([0, 1, 0], dtype="float32"))
    results = store.search(np.array([1, 0, 0], dtype="float32"), 1)
    assert results[0][0] == "a"
