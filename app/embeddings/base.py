from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return normalized embeddings for texts."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
