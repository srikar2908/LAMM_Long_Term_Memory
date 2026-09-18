import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.agent import LammAgent
from app.core.config import Settings


DEMO_TURNS = [
    "I prefer Python for backend development.",
    "I am working on an AI project.",
    "My project uses long-term LLM memory.",
    "I also like machine learning.",
    "Python is my preferred backend language.",
    "I have started using Java for my current backend project.",
    "Maybe remember that I sometimes browse AI news.",
    "I am writing a detailed LAMM project explanation with memory extraction, scoring, redundancy detection, lifecycle decisions, SQLite persistence, vector retrieval, evaluation reports, and a Streamlit dashboard for review demonstration purposes.",
    "This is temporary filler information that is not very useful for long term project personalization.",
]


if __name__ == "__main__":
    settings = Settings(database_url="sqlite:///storage/demo_lamm.db", faiss_index_path="storage/demo_vector_index", embedding_model="hash")
    if settings.sqlite_path.exists():
        settings.sqlite_path.unlink()
    agent = LammAgent(settings)
    print("LAMM deterministic demo")
    for turn in DEMO_TURNS:
        response = agent.chat(turn, "final_demo")
        print(f"\nUSER: {turn}")
        for decision in response.lifecycle_decisions:
            print(f"DECISION: {decision['operation']} - {decision['reason']}")
    final = agent.chat("What programming language am I currently using for my backend project?", "final_demo")
    print("\nQUERY: What programming language am I currently using for my backend project?")
    print(f"ASSISTANT: {final.response}")
    print("RETRIEVED:")
    for item in final.retrieved_memories:
        print(f"- {item['text']} ({item['score']:.2f})")
    print("STATS:", final.stats)
