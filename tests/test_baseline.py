from app.core.config import Settings
from app.evaluation.baseline import UnmanagedMemoryBaseline
from app.memory.manager import MemoryManager


def test_unmanaged_baseline_stores_all_candidates(tmp_path):
    settings = Settings(gemini_api_key="", database_url=f"sqlite:///{tmp_path / 'b.db'}", faiss_index_path=str(tmp_path / "b_index"), embedding_model="hash")
    manager = MemoryManager(settings)
    baseline = UnmanagedMemoryBaseline(manager)
    baseline.ingest("I prefer Python for backend development.")
    assert manager.stats()["total"] == 1
