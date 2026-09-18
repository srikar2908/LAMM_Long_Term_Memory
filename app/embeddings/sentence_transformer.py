from __future__ import annotations

import hashlib
import re

import numpy as np

from app.embeddings.base import EmbeddingProvider


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic offline embedding provider for tests and demos."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, idx] += sign
        return _normalize(vectors)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", fallback_dimension: int = 384):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.dimension = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            self.model = None
            self.fallback = HashEmbeddingProvider(fallback_dimension)
            self.dimension = fallback_dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return self.fallback.embed(texts)
        vectors = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        return _normalize(vectors)


def build_embedding_provider(model_name: str, dimension: int = 384) -> EmbeddingProvider:
    if model_name.lower() in {"hash", "mock", "deterministic"}:
        return HashEmbeddingProvider(dimension)
    return SentenceTransformerEmbeddingProvider(model_name, dimension)
