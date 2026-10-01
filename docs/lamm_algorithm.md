# LAMM Lifecycle Algorithm & Mathematical Specification

## 1. Mathematical Formulation of Memory Signals

The LAMM controller computes an interpretable composite score $S_{\text{composite}} \in [0, 1]$ across five distinct signals:

### 1.1 Relevance ($S_{\text{rel}}$)
Measures semantic similarity between candidate memory $M$ and the current conversational query or context $C$:
$$S_{\text{rel}} = \text{clamp}\left(\frac{\mathbf{v}_M \cdot \mathbf{v}_C}{\|\mathbf{v}_M\|_2 \|\mathbf{v}_C\|_2}\right)$$

### 1.2 Recency ($S_{\text{rec}}$)
Applies exponential temporal decay based on elapsed days $\Delta t_{\text{days}}$ since creation or last access:
$$S_{\text{rec}} = \exp(-\lambda_{\text{decay}} \cdot \Delta t_{\text{days}})$$
where default decay rate $\lambda_{\text{decay}} = 0.03$.

### 1.3 Confidence ($S_{\text{conf}}$)
The extraction certainty score assigned by the rule-based extractor or LLM parser ($S_{\text{conf}} \in [0, 1]$).

### 1.4 Redundancy ($S_{\text{red}}$)
Calculated as the maximum of semantic vector cosine similarity and lexical Jaccard word-overlap against the active memory pool:
$$S_{\text{red}} = \max\left(\max_{m \in \mathcal{S}_{\text{active}}} \cos(\mathbf{v}_M, \mathbf{v}_m), \max_{m \in \mathcal{S}_{\text{active}}} \text{Jaccard}(\text{words}(M), \text{words}(m))\right)$$

### 1.5 Utility ($S_{\text{util}}$)
Derived from past retrieval access frequency $N_{\text{access}}$:
$$S_{\text{util}} = \max\left(\text{prior\_util},\, 1 - \exp(-\beta \cdot N_{\text{access}})\right)$$
where default access saturation coefficient $\beta = 0.25$.

---

## 2. Composite Heuristic Scoring Formula

The overall score is computed as a weighted combination with a redundancy penalty:

$$S_{\text{composite}} = \text{clamp}\left( \frac{w_{\text{rel}} S_{\text{rel}} + w_{\text{rec}} S_{\text{rec}} + w_{\text{conf}} S_{\text{conf}} + w_{\text{util}} S_{\text{util}} - w_{\text{red}} S_{\text{red}}}{w_{\text{rel}} + w_{\text{rec}} + w_{\text{conf}} + w_{\text{util}}} \right)$$

### Default Weights & Thresholds Configuration
| Parameter | Default Value | Description |
| :--- | :---: | :--- |
| $w_{\text{rel}}$ (Relevance Weight) | `0.30` | Importance of context relevance |
| $w_{\text{rec}}$ (Recency Weight) | `0.15` | Weight of memory freshness |
| $w_{\text{conf}}$ (Confidence Weight) | `0.20` | Extraction confidence weighting |
| $w_{\text{red}}$ (Redundancy Weight) | `0.25` | Penalty factor for duplicate content |
| $w_{\text{util}}$ (Utility Weight) | `0.10` | Historical access utility weighting |
| $\tau_{\text{red}}$ (Redundancy Threshold) | `0.86` | Threshold for MERGE detection |
| $\tau_{\text{arc}}$ (Archive Threshold) | `0.32` | Lower bound for active retention |
| $\tau_{\text{fgt}}$ (Forget Threshold) | `0.18` | Threshold for eviction/forgetting |
| $L_{\text{comp}}$ (Compress Length) | `180` chars | Text length triggering compression |

> **Note on Heuristics:** These default weights and thresholds are engineering assumptions designed for demonstration and research calibration; they are fully configurable via `configs/default.yaml` or environment variables.

---

## 3. Lifecycle Decision Priority Engine

To prevent score-based eviction from prematurely discarding critical updates or merges, decisions follow a strict, deterministic priority order:

```text
Incoming Candidate Memory (M)
              │
              ▼
   [1. UPDATE Detection]
   Evidence of modification, supersession, contradiction, or temporal shift?
        ├── YES ──► OPERATION: UPDATE (Modify existing memory in-place)
        └── NO
              │
              ▼
   [2. MERGE Detection]
   Redundancy score >= Redundancy Threshold (0.86) or Lexical Similarity >= 0.40?
        ├── YES ──► OPERATION: MERGE (Combine facts, deduplicate)
        └── NO
              │
              ▼
   [3. COMPRESS Detection]
   Candidate character length > Compression Limit (180 chars)?
        ├── YES ──► OPERATION: COMPRESS (Condense in-place representation)
        └── NO
              │
              ▼
   [4. FORGET Evaluation]
   Explicit ephemeral keywords detected OR S_composite <= Forget Threshold (0.18)?
        ├── YES ──► OPERATION: FORGET (Mark forgotten, remove from FAISS)
        └── NO
              │
              ▼
   [5. ARCHIVE Evaluation]
   Candidate confidence < 0.50 OR Active Count >= Max Capacity OR S_composite <= Archive Threshold (0.32)?
        ├── YES ──► OPERATION: ARCHIVE (Mark archived, remove from FAISS)
        └── NO
              │
              ▼
   [6. RETAIN Evaluation]
   Informative, durable, high-confidence fact
        └───► OPERATION: RETAIN (Insert into SQLite, add vector to FAISS)
```

---

## 4. Algorithmic Invariants

1. **Evidence-Based UPDATE:**  
   `UPDATE` is not triggered by mere cosine similarity. It requires explicit linguistic indicators of state change (e.g., *"started using"*, *"switched to"*, *"no longer"*, *"now prefer"*, *"instead of"*, *"updated to"*) or contradictory values targeting the same subject.
2. **In-Place COMPRESS:**  
   `COMPRESS` modifies the active text representation of the memory record. It updates the embedding vector in FAISS without incrementing total database memory records.
3. **Capacity Enforcement:**  
   If active memory count reaches `max_active_memories`, incoming lower-priority memories are routed to `ARCHIVE` rather than overflowing the retrieval pool.
4. **Auditability & Explainability:**  
   Every decision records:
   - Operation type (`RETAIN`, `UPDATE`, `MERGE`, `COMPRESS`, `ARCHIVE`, `FORGET`)
   - Natural-language reason
   - Full 5-signal score bundle and composite score
   - Affected memory IDs
   - Before-state and after-state snapshots
