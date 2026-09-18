# LAMM Algorithm

## Inputs

- New candidate memory `M`
- Existing memory set `S`
- Current context `C`
- Resource constraints `R`

## Pseudocode

```text
INPUT:
new memory M
existing memory set S
current context C
resource constraints R

1. Generate embedding for M
2. Retrieve similar memories from active memory set S
3. Calculate:
   relevance
   recency
   confidence
   redundancy
   utility
4. Compute weighted lifecycle score
5. Select one operation:
   RETAIN / UPDATE / MERGE / COMPRESS / ARCHIVE / FORGET
6. Apply operation
7. Update vector index
8. Update SQLite
9. Record lifecycle event
```

## Scoring

```text
score = relevance*w_relevance
      + recency*w_recency
      + confidence*w_confidence
      + utility*w_utility
      - redundancy*w_redundancy
```

Initial weights are configurable assumptions, not validated research findings.

## Explainability

Every decision returns operation, reason, scores, affected memory IDs, confidence, and metadata.
