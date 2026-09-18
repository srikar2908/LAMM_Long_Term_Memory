class LammError(Exception):
    """Base application error."""


class MemoryNotFoundError(LammError):
    """Raised when a memory id is unknown."""


class IndexConsistencyError(LammError):
    """Raised when vector index mappings are stale or corrupted."""
