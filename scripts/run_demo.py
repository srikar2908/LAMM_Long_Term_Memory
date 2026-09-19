from __future__ import annotations

import os
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


def run_realistic_demo():
    use_gemini = os.getenv("USE_GEMINI", "0") == "1"
    gemini_key = os.getenv("GEMINI_API_KEY", "") if use_gemini else ""

    settings = Settings(
        gemini_api_key=gemini_key,
        database_url="sqlite:///storage/demo_lamm.db",
        faiss_index_path="storage/demo_vector_index",
        embedding_provider="hash",
        embedding_model="deterministic-hash",
    )

    # Clean up prior demo database and index
    if settings.sqlite_path.exists():
        settings.sqlite_path.unlink()
    for p in [Path(settings.faiss_index_path).with_suffix(ext) for ext in [".faiss", ".mapping.json", ".npz"]]:
        if p.exists():
            p.unlink()

    agent = LammAgent(settings)
    mode_str = "Gemini LLM (online)" if use_gemini and settings.gemini_available else "Deterministic Mock (offline)"
    print("=" * 70)
    print("LAMM: Lightweight Adaptive Memory Management - Conversational Demo")
    print(f"Operational Mode: {mode_str}")
    print("=" * 70)

    for turn_idx, turn in enumerate(DEMO_TURNS, start=1):
        response = agent.chat(turn, conversation_id="demo_session")
        print(f"\n[Turn {turn_idx}] USER: {turn}")
        print(f"[Turn {turn_idx}] ASSISTANT: {response.response}")
        for decision in response.lifecycle_decisions:
            print(f"  -> LIFECYCLE DECISION: {decision['operation']} | Reason: {decision['reason']}")

    # Final query verifying memory recall of updated information
    query = "What programming language am I currently using for my backend project?"
    print("\n" + "=" * 70)
    print(f"FINAL TEST QUERY: {query}")
    print("=" * 70)
    final_response = agent.chat(query, conversation_id="demo_session")
    print(f"\nASSISTANT: {final_response.response}")
    print("\nRETRIEVED ACTIVE MEMORIES:")
    for item in final_response.retrieved_memories:
        print(f"  * {item['text']} (relevance score: {item['score']:.2f})")

    print("\nFINAL SYSTEM METRICS & STATS:")
    for k, v in final_response.stats.items():
        print(f"  * {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    run_realistic_demo()
