from __future__ import annotations

import numpy as np
from app.embeddings.sentence_transformer import (
    HashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    build_embedding_provider,
)


def test_hash_embedding_provider():
    provider = HashEmbeddingProvider(dimension=384)
    vectors = provider.embed(["User prefers Python.", "User prefers Python."])
    assert vectors.shape == (2, 384)
    # Vectors must be unit normalized
    norm0 = np.linalg.norm(vectors[0])
    assert np.isclose(norm0, 1.0, atol=1e-4)
    # Determinism: identical text yields identical vectors
    assert np.allclose(vectors[0], vectors[1])


def test_build_embedding_provider():
    p_hash = build_embedding_provider("hash", "mock", 128)
    assert isinstance(p_hash, HashEmbeddingProvider)
    assert p_hash.dimension == 128

    p_st = build_embedding_provider("sentence_transformer", "all-MiniLM-L6-v2", 384)
    assert isinstance(p_st, (SentenceTransformerEmbeddingProvider, HashEmbeddingProvider))
    assert p_st.dimension == 384


def test_sentence_transformer_normalization():
    p = build_embedding_provider("sentence_transformer", "all-MiniLM-L6-v2", 384)
    vec = p.embed_one("Long-term memory management for LLM agents.")
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-4)
