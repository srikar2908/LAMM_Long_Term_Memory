from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from app.agent.agent import LammAgent
from app.core.config import Settings
from app.evaluation.baseline import UnmanagedMemoryBaseline
from app.evaluation.metrics import precision_at_k
from app.memory.manager import MemoryManager


class EvaluationRunner:
    def __init__(self, dataset_path: str = "data/evaluation/synthetic_demo.json"):
        self.dataset_path = Path(dataset_path)

    def load_dataset(self) -> dict:
        return json.loads(self.dataset_path.read_text())

    def run(self) -> dict:
        data = self.load_dataset()
        lamm_settings = Settings(database_url="sqlite:///storage/eval_lamm.db", faiss_index_path="storage/eval_lamm_vector_index", embedding_model="hash")
        baseline_settings = Settings(database_url="sqlite:///storage/eval_baseline.db", faiss_index_path="storage/eval_baseline_vector_index", embedding_model="hash")
        for path in [lamm_settings.sqlite_path, baseline_settings.sqlite_path]:
            if path.exists():
                path.unlink()
        lamm = LammAgent(lamm_settings)
        baseline_manager = MemoryManager(baseline_settings)
        baseline = UnmanagedMemoryBaseline(baseline_manager)
        rows = []
        for turn_idx, turn in enumerate(data["turns"], start=1):
            if turn["role"] != "user":
                continue
            lamm_response = lamm.chat(turn["text"], data["conversation_id"])
            baseline.ingest(turn["text"], data["conversation_id"])
            rows.append({"system": "LAMM", "turn": turn_idx, **lamm_response.stats})
            rows.append({"system": "UNMANAGED_MEMORY", "turn": turn_idx, **baseline_manager.stats()})
        retrieval_rows = []
        for query in data.get("memory_queries", []):
            for name, manager in [("LAMM", lamm.manager), ("UNMANAGED_MEMORY", baseline_manager)]:
                start = time.perf_counter()
                retrieved = manager.retrieve(query["query"])
                latency = time.perf_counter() - start
                ids = [item.memory.id for item in retrieved]
                retrieval_rows.append(
                    {
                        "system": name,
                        "query": query["query"],
                        "retrieved_count": len(retrieved),
                        "latency_seconds": latency,
                        "precision_at_k": precision_at_k(ids, query.get("expected_memory_ids", []), manager.settings.top_k),
                    }
                )
        events = Counter(event["operation"] for event in lamm.manager.events.list())
        result = {
            "dataset": str(self.dataset_path),
            "note": "Synthetic demo data; not a benchmark result.",
            "memory_growth": rows,
            "retrieval": retrieval_rows,
            "lifecycle_operation_counts": dict(events),
            "lamm_stats": lamm.manager.stats(),
            "baseline_stats": baseline_manager.stats(),
        }
        self.save(result)
        return result

    def save(self, result: dict) -> None:
        Path("results/tables").mkdir(parents=True, exist_ok=True)
        Path("results/figures").mkdir(parents=True, exist_ok=True)
        Path("results/reports").mkdir(parents=True, exist_ok=True)
        growth = pd.DataFrame(result["memory_growth"])
        retrieval = pd.DataFrame(result["retrieval"])
        growth.to_csv("results/tables/memory_growth.csv", index=False)
        retrieval.to_csv("results/tables/retrieval.csv", index=False)
        Path("results/reports/evaluation_report.json").write_text(json.dumps(result, indent=2))
        if plt is not None and not growth.empty:
            ax = growth.pivot(index="turn", columns="system", values="total").plot(marker="o")
            ax.set_title("Memory Growth Over Synthetic Turns")
            ax.set_xlabel("Turn")
            ax.set_ylabel("Stored memories")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig("results/figures/memory_growth.png")
            plt.close(fig)
            lifecycle = pd.Series(result["lifecycle_operation_counts"])
            if not lifecycle.empty:
                ax = lifecycle.plot(kind="bar", title="LAMM Lifecycle Operation Distribution")
                ax.set_xlabel("Operation")
                ax.set_ylabel("Count")
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig("results/figures/lifecycle_operations.png")
                plt.close(fig)
        elif plt is None:
            Path("results/figures/README.txt").write_text("Matplotlib is not installed; plots were skipped. Install requirements.txt to enable figures.")
