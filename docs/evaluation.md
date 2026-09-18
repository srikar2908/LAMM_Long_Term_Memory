# Evaluation

The evaluation framework compares:

- `LAMM`: adaptive lifecycle management
- `UNMANAGED_MEMORY`: stores every extracted memory

## Metrics

- Memory growth
- Active memory count
- Archived memory count
- Forgotten memory count
- Lifecycle operation distribution
- Retrieval latency
- Approximate context size
- Precision@k when ground-truth expected IDs are provided

## Dataset Format

```json
{
  "conversation_id": "conv_001",
  "turns": [
    {"role": "user", "text": "..."}
  ],
  "memory_queries": [
    {"query": "...", "expected_memory_ids": []}
  ]
}
```

The included synthetic dataset is for development and demonstration only.
