from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    import yaml
except ImportError:
    yaml = None


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence_transformer")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
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
    recency_baseline_budget: int = _int("RECENCY_BASELINE_BUDGET", 5)
    embedding_dimension: int = _int("EMBEDDING_DIMENSION", 384)
    dry_run: bool = _bool("LAMM_DRY_RUN", False)
    random_seed: int = _int("RANDOM_SEED", 42)

    def __post_init__(self):
        # Validate weights
        weights = [
            self.relevance_weight,
            self.recency_weight,
            self.confidence_weight,
            self.redundancy_weight,
            self.utility_weight,
        ]
        if any(w < 0 for w in weights):
            raise ValueError(f"All scoring weights must be non-negative. Got: {weights}")
        if sum(weights) <= 0:
            raise ValueError("Sum of scoring weights must be greater than zero.")

        # Validate threshold relationships
        if not (0.0 <= self.forget_threshold <= self.archive_threshold <= 1.0):
            raise ValueError(
                f"Invalid threshold ordering: must satisfy 0 <= forget_threshold ({self.forget_threshold}) "
                f"<= archive_threshold ({self.archive_threshold}) <= 1.0"
            )
        if not (0.0 <= self.redundancy_threshold <= 1.0):
            raise ValueError(
                f"Redundancy threshold must be between 0.0 and 1.0. Got: {self.redundancy_threshold}"
            )

    @property
    def gemini_available(self) -> bool:
        return bool(self.gemini_api_key.strip())

    @property
    def sqlite_path(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", "", 1))
        return Path(self.database_url)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gemini_model": self.gemini_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "database_url": self.database_url,
            "faiss_index_path": self.faiss_index_path,
            "top_k": self.top_k,
            "relevance_weight": self.relevance_weight,
            "recency_weight": self.recency_weight,
            "confidence_weight": self.confidence_weight,
            "redundancy_weight": self.redundancy_weight,
            "utility_weight": self.utility_weight,
            "redundancy_threshold": self.redundancy_threshold,
            "archive_threshold": self.archive_threshold,
            "forget_threshold": self.forget_threshold,
            "recency_decay": self.recency_decay,
            "compress_length": self.compress_length,
            "max_active_memories": self.max_active_memories,
            "recency_baseline_budget": self.recency_baseline_budget,
            "embedding_dimension": self.embedding_dimension,
            "dry_run": self.dry_run,
            "random_seed": self.random_seed,
            "gemini_available": self.gemini_available,
        }


def load_settings_from_yaml(yaml_path: str | Path) -> Settings:
    path = Path(yaml_path)
    if not path.exists():
        return Settings()
    if yaml is None:
        return Settings()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Extract flat or nested fields
    kwargs: dict[str, Any] = {}
    if "embedding" in data:
        if "provider" in data["embedding"]:
            kwargs["embedding_provider"] = data["embedding"]["provider"]
        if "model" in data["embedding"]:
            kwargs["embedding_model"] = data["embedding"]["model"]
        if "dimension" in data["embedding"]:
            kwargs["embedding_dimension"] = data["embedding"]["dimension"]

    if "retrieval" in data:
        if "top_k" in data["retrieval"]:
            kwargs["top_k"] = data["retrieval"]["top_k"]

    if "lamm" in data:
        for k, v in data["lamm"].items():
            kwargs[k] = v

    if "evaluation" in data:
        if "seed" in data["evaluation"]:
            kwargs["random_seed"] = data["evaluation"]["seed"]
        if "recency_baseline_budget" in data["evaluation"]:
            kwargs["recency_baseline_budget"] = data["evaluation"]["recency_baseline_budget"]

    # Flat overrides
    for k, v in data.items():
        if hasattr(Settings, k):
            kwargs[k] = v

    return Settings(**kwargs)


def get_settings(config_path: str | Path | None = None) -> Settings:
    config_file = config_path or os.getenv("LAMM_CONFIG_PATH")
    if config_file and Path(config_file).exists():
        return load_settings_from_yaml(config_file)
    return Settings()
