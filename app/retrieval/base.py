from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        pass

    @abstractmethod
    def search(self, embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        pass

    @abstractmethod
    def rebuild(self, embeddings: dict[str, np.ndarray]) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass
