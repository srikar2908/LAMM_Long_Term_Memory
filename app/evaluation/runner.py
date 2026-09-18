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
try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

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
        lamm_settings = Settings(
            gemini_api_key="",
            database_url="sqlite:///storage/eval_lamm.db",
            faiss_index_path="storage/eval_lamm_vector_index",
            embedding_model="hash",
        )
        baseline_settings = Settings(
            gemini_api_key="",
            database_url="sqlite:///storage/eval_baseline.db",
            faiss_index_path="storage/eval_baseline_vector_index",
            embedding_model="hash",
        )
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
            self._write_fallback_figures(result, growth)

    def _write_fallback_figures(self, result: dict, growth: pd.DataFrame) -> None:
        if Image is None or ImageDraw is None:
            Path("results/figures/README.txt").write_text("Matplotlib and Pillow are unavailable; plots were skipped. Install requirements.txt to enable figures.")
            return
        self._simple_png(
            Path("results/figures/memory_growth.png"),
            "Memory Growth Over Synthetic Turns",
            [f"{row.system} turn {row.turn}: total={row.total}" for row in growth.itertuples()],
        )
        counts = result["lifecycle_operation_counts"]
        self._simple_png(
            Path("results/figures/lifecycle_operations.png"),
            "LAMM Lifecycle Operation Distribution",
            [f"{key}: {value}" for key, value in counts.items()],
        )

    def _simple_png(self, path: Path, title: str, lines: list[str]) -> None:
        width = 1100
        height = max(360, 80 + 26 * len(lines))
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 20), title, fill=(20, 20, 20))
        y = 64
        for line in lines[:40]:
            draw.text((32, y), line, fill=(45, 45, 45))
            y += 26
        image.save(path)
