from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "hash")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///storage/lamm.db")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "storage/vector_index")
    top_k: int = _int("TOP_K", 5)
    relevance_weight: float = _float("RELEVANCE_WEIGHT", 0.30)
    recency_weight: float = _float("RECENCY_WEIGHT", 0.15)
    confidence_weight: float = _float("CONFIDENCE_WEIGHT", 0.20)
    redundancy_weight: float = _float("REDUNDANCY_WEIGHT", 0.25)
    utility_weight: float = _float("UTILITY_WEIGHT", 0.10)
    redundancy_threshold: float = _float("REDUNDANCY_THRESHOLD", 0.86)
    archive_threshold: float = _float("ARCHIVE_THRESHOLD", 0.32)
    forget_threshold: float = _float("FORGET_THRESHOLD", 0.18)
    recency_decay: float = _float("RECENCY_DECAY", 0.03)
    compress_length: int = _int("COMPRESS_LENGTH", 180)
    max_active_memories: int = _int("MAX_ACTIVE_MEMORIES", 100)
    embedding_dimension: int = _int("EMBEDDING_DIMENSION", 384)

    @property
    def gemini_available(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def sqlite_path(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", "", 1))
        return Path(self.database_url)


def get_settings() -> Settings:
    return Settings()
