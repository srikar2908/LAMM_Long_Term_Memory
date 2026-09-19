from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.retrieval.base import VectorStore

logger = logging.getLogger("lamm.retrieval")

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


class FaissVectorStore(VectorStore):
    """
    FAISS-backed vector store for normalized inner-product (cosine) similarity.
    Architectural rule: SQLite is the source of truth. FAISS is a derived retrieval index.
    Only ACTIVE memories are maintained in this index.
    """

    def __init__(self, index_path: str, dimension: int):
        self.base_path = Path(index_path)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.mapping_path = self.base_path.with_suffix(".mapping.json")
        self.numpy_path = self.base_path.with_suffix(".npz")
        self.faiss_path = self.base_path.with_suffix(".faiss")
        self.ids: list[str] = []
        self.vectors = np.zeros((0, dimension), dtype="float32")
        self.index = None
        self._load()

    def _new_index(self):
        if faiss is None:
            return None
        return faiss.IndexFlatIP(self.dimension)

    def _load(self) -> None:
        if self.mapping_path.exists():
            try:
                self.ids = json.loads(self.mapping_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not load mapping file {self.mapping_path}: {e}")
                self.ids = []
        if self.numpy_path.exists():
            try:
                data = np.load(self.numpy_path)
                self.vectors = data["vectors"].astype("float32")
            except Exception as e:
                logger.warning(f"Could not load numpy backup {self.numpy_path}: {e}")
                self.vectors = np.zeros((0, self.dimension), dtype="float32")
        if faiss is not None and self.faiss_path.exists():
            try:
                self.index = faiss.read_index(str(self.faiss_path))
            except Exception as e:
                logger.warning(f"Could not read FAISS index {self.faiss_path}: {e}. Rebuilding in-memory index.")
                self._rebuild_index_only()
        else:
            self._rebuild_index_only()

    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        """Add or update an active memory embedding in the index."""
        vector = embedding.astype("float32").reshape(1, self.dimension)
        if memory_id in self.ids:
            idx = self.ids.index(memory_id)
            self.vectors[idx] = vector[0]
            self._rebuild_index_only()
        else:
            self.ids.append(memory_id)
            if len(self.vectors) == 0:
                self.vectors = vector
            else:
                self.vectors = np.vstack([self.vectors, vector])
            if self.index is not None:
                self.index.add(vector)
        self.save()

    def remove(self, memory_id: str) -> bool:
        """Remove a memory from the active index (e.g. after FORGET or ARCHIVE)."""
        if memory_id not in self.ids:
            return False
        idx = self.ids.index(memory_id)
        self.ids.pop(idx)
        if len(self.ids) == 0:
            self.vectors = np.zeros((0, self.dimension), dtype="float32")
        else:
            self.vectors = np.delete(self.vectors, idx, axis=0)
        self._rebuild_index_only()
        self.save()
        return True

    def search(self, embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Search top-k most similar active memories by cosine similarity (normalized inner product)."""
        if not self.ids:
            return []
        query = embedding.astype("float32").reshape(1, self.dimension)
        k = min(top_k, len(self.ids))
        if k <= 0:
            return []

        # If FAISS is available and synchronized
        if self.index is not None and getattr(self.index, "ntotal", 0) == len(self.ids):
            scores, idxs = self.index.search(query, k)
            return [(self.ids[int(i)], float(s)) for i, s in zip(idxs[0], scores[0]) if 0 <= int(i) < len(self.ids)]

        # Fallback to NumPy matrix multiplication
        scores = self.vectors @ query[0]
        order = np.argsort(scores)[::-1][:k]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in order]

    def rebuild(self, embeddings: dict[str, np.ndarray]) -> None:
        """Completely rebuild FAISS index from the canonical SQLite active memory set."""
        self.ids = list(embeddings.keys())
        if self.ids:
            self.vectors = np.vstack([embeddings[mid] for mid in self.ids]).astype("float32")
        else:
            self.vectors = np.zeros((0, self.dimension), dtype="float32")
        self._rebuild_index_only()
        self.save()

    def _rebuild_index_only(self) -> None:
        self.index = self._new_index()
        if self.index is not None and len(self.vectors) > 0:
            self.index.add(self.vectors)

    def save(self) -> None:
        self.mapping_path.write_text(json.dumps(self.ids, indent=2), encoding="utf-8")
        np.savez_compressed(self.numpy_path, vectors=self.vectors)
        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, str(self.faiss_path))

    def consistency_errors(self, active_memory_ids: set[str]) -> list[str]:
        """Detect any stale mappings in FAISS that do not exist in SQLite active memories."""
        return [mid for mid in self.ids if mid not in active_memory_ids]
