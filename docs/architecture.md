# LAMM Architecture

LAMM separates the LLM from the memory-management controller.

The LLM may generate responses, extract memory candidates, and help compress text when configured. The LAMM controller owns scoring, lifecycle decisions, persistence, retrieval, and event logging.

## Components

- Agent pipeline: receives messages, retrieves memories, builds context, calls LLM/mock LLM, extracts memory candidates, and applies lifecycle decisions.
- Memory extraction: Gemini-based when available, deterministic fallback otherwise.
- Embeddings: sentence-transformers when configured, deterministic hash embeddings for offline mode.
- Retrieval: FAISS normalized inner product when available, NumPy fallback otherwise.
- Storage: SQLite stores memory text and metadata as the source of truth.
- Lifecycle events: every decision is recorded with scores and reasons.

## Data Flow

1. User sends a message.
2. Query embedding is generated.
3. Top-k active memories are retrieved.
4. Context is built from selected memories and current conversation.
5. LLM or mock LLM generates a response.
6. Candidate memories are extracted.
7. LAMM scores each candidate.
8. Lifecycle operation is applied.
9. SQLite, vector index, and lifecycle event log are updated.
