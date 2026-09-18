from app.agent.agent import LammAgent
from app.core.config import Settings


def test_end_to_end_mock_pipeline(tmp_path):
    settings = Settings(gemini_api_key="", database_url=f"sqlite:///{tmp_path / 'e2e.db'}", faiss_index_path=str(tmp_path / "e2e_index"), embedding_model="hash")
    agent = LammAgent(settings)
    first = agent.chat("I prefer Python for backend development.")
    assert first.extracted_memories
    second = agent.chat("I have started using Java for my current backend project.")
    assert second.lifecycle_decisions
    final = agent.chat("What programming language am I currently using for my backend project?")
    assert "Java" in final.response
