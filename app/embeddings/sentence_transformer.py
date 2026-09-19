from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger("lamm.embeddings")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic offline embedding provider for reproducible tests and fast demos."""

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
    """Local embedding provider powered by sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", fallback_dimension: int = 384):
        self.model_name = model_name
        self.model = None
        self.fallback = HashEmbeddingProvider(fallback_dimension)
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            # Use get_embedding_dimension if available (sentence-transformers >= 3.0), else get_sentence_embedding_dimension
            if hasattr(self.model, "get_embedding_dimension"):
                self.dimension = int(self.model.get_embedding_dimension())
            else:
                self.dimension = int(self.model.get_sentence_embedding_dimension())
            logger.info(f"Loaded SentenceTransformer '{model_name}' (dimension={self.dimension})")
        except Exception as exc:
            logger.warning(f"Failed to load SentenceTransformer '{model_name}': {exc}. Falling back to HashEmbeddingProvider.")
            self.model = None
            self.dimension = fallback_dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return self.fallback.embed(texts)
        try:
            vectors = self.model.encode(texts, convert_to_numpy=True).astype("float32")
            return _normalize(vectors)
        except Exception as exc:
            logger.error(f"Encoding error with SentenceTransformer: {exc}. Using fallback.")
            return self.fallback.embed(texts)


def build_embedding_provider(
    provider: str = "sentence_transformer",
    model_name: str = "all-MiniLM-L6-v2",
    dimension: int = 384,
) -> EmbeddingProvider:
    provider_clean = (provider or "").strip().lower()
    if provider_clean in {"hash", "mock", "deterministic"} or model_name.lower() in {"hash", "mock", "deterministic"}:
        return HashEmbeddingProvider(dimension)
    return SentenceTransformerEmbeddingProvider(model_name=model_name, fallback_dimension=dimension)
