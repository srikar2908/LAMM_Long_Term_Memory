from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from app.agent.agent import LammAgent
from app.core.config import Settings
from app.evaluation.baseline import RecencyOnlyBaseline, UnmanagedMemoryBaseline
from app.evaluation.metrics import (
    compute_latency_stats,
    f1_at_k,
    latency_change_percentage,
    mean_reciprocal_rank,
    memory_reduction_percentage,
    precision_at_k,
    recall_at_k,
    token_reduction_percentage,
)
from app.memory.manager import MemoryManager


class EvaluationRunner:
    """
    Experimental Evaluation Harness.
    Conducts fair 3-way comparative evaluation:
      - LAMM: Lightweight Adaptive Memory Management (adaptive lifecycle)
      - UNMANAGED_MEMORY: Unconditional persistence without eviction (unbounded growth)
      - RECENCY_ONLY: Fixed LRU budget memory eviction without semantic signals

    Architectural Invariant:
    All systems receive strictly identical conversation sequences, candidate memories,
    evaluation queries, embedding models, and top-k retrieval settings.
    """

    def __init__(self, dataset_path: str = "data/evaluation/synthetic_demo.json"):
        self.dataset_path = Path(dataset_path)

    def load_dataset(self) -> dict[str, Any]:
        return json.loads(self.dataset_path.read_text(encoding="utf-8"))

    def run(self) -> dict[str, Any]:
        data = self.load_dataset()
        conv_id = data.get("conversation_id", "eval_001")

        # Configurations - strictly identical embedding and retrieval parameters
        shared_params = dict(
            gemini_api_key="",  # Offline reproducible evaluation
            embedding_provider="hash",  # Consistent deterministic vector representations
            embedding_model="deterministic-hash",
            embedding_dimension=384,
            top_k=5,
            recency_baseline_budget=5,
        )

        lamm_settings = Settings(
            database_url="sqlite:///storage/eval_lamm.db",
            faiss_index_path="storage/eval_lamm_vector_index",
            **shared_params,
        )
        unmanaged_settings = Settings(
            database_url="sqlite:///storage/eval_unmanaged.db",
            faiss_index_path="storage/eval_unmanaged_vector_index",
            **shared_params,
        )
        recency_settings = Settings(
            database_url="sqlite:///storage/eval_recency.db",
            faiss_index_path="storage/eval_recency_vector_index",
            **shared_params,
        )

        # Ensure clean state for evaluation
        for s in [lamm_settings, unmanaged_settings, recency_settings]:
            if s.sqlite_path.exists():
                s.sqlite_path.unlink()
            for p in [Path(s.faiss_index_path).with_suffix(ext) for ext in [".faiss", ".mapping.json", ".npz"]]:
                if p.exists():
                    p.unlink()

        lamm_agent = LammAgent(lamm_settings)
        unmanaged_manager = MemoryManager(unmanaged_settings)
        unmanaged_baseline = UnmanagedMemoryBaseline(unmanaged_manager)
        recency_manager = MemoryManager(recency_settings)
        recency_baseline = RecencyOnlyBaseline(recency_manager, budget=shared_params["recency_baseline_budget"])

        turn_rows: list[dict[str, Any]] = []

        # Ingest identical conversation turns
        for turn_idx, turn in enumerate(data.get("turns", []), start=1):
            if turn["role"] != "user":
                continue
            user_text = turn["text"]

            # 1. LAMM Agent
            t0 = time.perf_counter()
            lamm_res = lamm_agent.chat(user_text, conv_id)
            lamm_e2e = time.perf_counter() - t0
            lamm_stats = lamm_agent.manager.stats()
            turn_rows.append({
                "system": "LAMM",
                "turn": turn_idx,
                "input_text": user_text,
                "total": lamm_stats["total"],
                "active": lamm_stats["active"],
                "archived": lamm_stats["archived"],
                "forgotten": lamm_stats["forgotten"],
                "context_tokens": lamm_res.stats.get("context_tokens", 0),
                "e2e_latency": lamm_e2e,
            })

            # 2. UNMANAGED_MEMORY
            t0 = time.perf_counter()
            unmanaged_baseline.ingest(user_text, conv_id)
            unm_e2e = time.perf_counter() - t0
            unm_stats = unmanaged_manager.stats()
            # Estimate context for unmanaged
            retrieved_unm = unmanaged_manager.retrieve(user_text)
            _, unm_context_stats = lamm_agent.context_builder.build(user_text, retrieved_unm)
            turn_rows.append({
                "system": "UNMANAGED_MEMORY",
                "turn": turn_idx,
                "input_text": user_text,
                "total": unm_stats["total"],
                "active": unm_stats["active"],
                "archived": unm_stats["archived"],
                "forgotten": unm_stats["forgotten"],
                "context_tokens": unm_context_stats.get("context_tokens", 0),
                "e2e_latency": unm_e2e,
            })

            # 3. RECENCY_ONLY (LRU)
            t0 = time.perf_counter()
            recency_baseline.ingest(user_text, conv_id)
            rec_e2e = time.perf_counter() - t0
            rec_stats = recency_manager.stats()
            retrieved_rec = recency_manager.retrieve(user_text)
            _, rec_context_stats = lamm_agent.context_builder.build(user_text, retrieved_rec)
            turn_rows.append({
                "system": "RECENCY_ONLY",
                "turn": turn_idx,
                "input_text": user_text,
                "total": rec_stats["total"],
                "active": rec_stats["active"],
                "archived": rec_stats["archived"],
                "forgotten": rec_stats["forgotten"],
                "context_tokens": rec_context_stats.get("context_tokens", 0),
                "e2e_latency": rec_e2e,
            })

        # Retrieval Query Evaluation (Strict Ground-Truth Compliance)
        retrieval_rows: list[dict[str, Any]] = []
        systems = [
            ("LAMM", lamm_agent.manager),
            ("UNMANAGED_MEMORY", unmanaged_manager),
            ("RECENCY_ONLY", recency_manager),
        ]

        for query_spec in data.get("memory_queries", []):
            q_text = query_spec["query"]
            has_gt = query_spec.get("ground_truth_available", False)
            expected_keywords = query_spec.get("expected_topic_keywords", [])

            for sys_name, mgr in systems:
                t0 = time.perf_counter()
                retrieved = mgr.retrieve(q_text, top_k=mgr.settings.top_k)
                latency = time.perf_counter() - t0

                # Check relevance matching based on expected keywords if ground truth is available
                precision, recall, mrr = None, None, None
                if has_gt and expected_keywords:
                    # Ground truth check: retrieved memory text must match expected topical keywords
                    matched_ranks = []
                    for rank, item in enumerate(retrieved, start=1):
                        lower_text = item.memory.text.lower()
                        # Relevant if it contains the target keyword
                        if any(kw.lower() in lower_text for kw in expected_keywords):
                            matched_ranks.append(rank)

                    num_relevant = len(matched_ranks)
                    k = mgr.settings.top_k
                    precision = num_relevant / float(min(k, max(1, len(retrieved))))
                    # Assume 1 target memory sought for these specific queries
                    recall = min(1.0, num_relevant / 1.0)
                    mrr = 1.0 / matched_ranks[0] if matched_ranks else 0.0

                retrieval_rows.append({
                    "system": sys_name,
                    "query_id": query_spec.get("query_id", ""),
                    "query": q_text,
                    "ground_truth_available": has_gt,
                    "retrieved_count": len(retrieved),
                    "retrieval_latency_seconds": latency,
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "mrr": mrr,
                })

        # Aggregate Level 1 & Level 2 Metrics
        growth_df = pd.DataFrame(turn_rows)
        retrieval_df = pd.DataFrame(retrieval_rows)

        # Lifecycle operations breakdown for LAMM
        events = Counter(e["operation"] for e in lamm_agent.manager.events.list())

        # Final memory stats
        final_lamm = lamm_agent.manager.stats()
        final_unmanaged = unmanaged_manager.stats()
        final_recency = recency_manager.stats()

        # Measured reductions
        mem_reduction_vs_unmanaged = memory_reduction_percentage(final_unmanaged["active"], final_lamm["active"])
        mem_reduction_vs_recency = memory_reduction_percentage(final_recency["active"], final_lamm["active"])

        # Token reductions across turns
        avg_tokens_lamm = growth_df[growth_df["system"] == "LAMM"]["context_tokens"].mean()
        avg_tokens_unm = growth_df[growth_df["system"] == "UNMANAGED_MEMORY"]["context_tokens"].mean()
        tok_reduction = token_reduction_percentage(avg_tokens_unm, avg_tokens_lamm)

        # Retrieval accuracy summaries (strictly on ground-truth annotated queries)
        gt_retrieval = retrieval_df[retrieval_df["ground_truth_available"] == True]
        accuracy_summary: dict[str, Any] = {}
        for sys_name in ["LAMM", "UNMANAGED_MEMORY", "RECENCY_ONLY"]:
            sub = gt_retrieval[gt_retrieval["system"] == sys_name]
            accuracy_summary[sys_name] = {
                "mean_precision_at_k": float(sub["precision_at_k"].mean()) if not sub.empty else None,
                "mean_recall_at_k": float(sub["recall_at_k"].mean()) if not sub.empty else None,
                "mean_mrr": float(sub["mrr"].mean()) if not sub.empty else None,
                "mean_latency": float(retrieval_df[retrieval_df["system"] == sys_name]["retrieval_latency_seconds"].mean()),
            }

        result = {
            "experiment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": str(self.dataset_path),
            "note": "Scenario-based synthetic evaluation data. Metrics are actual measured results.",
            "configuration": shared_params,
            "final_memory_counts": {
                "LAMM": final_lamm,
                "UNMANAGED_MEMORY": final_unmanaged,
                "RECENCY_ONLY": final_recency,
            },
            "comparative_metrics": {
                "measured_active_memory_reduction_vs_unmanaged_pct": round(mem_reduction_vs_unmanaged, 2),
                "measured_context_token_reduction_vs_unmanaged_pct": round(tok_reduction, 2),
                "lamm_lifecycle_operations": dict(events),
                "retrieval_accuracy_summary": accuracy_summary,
            },
            "turn_history": turn_rows,
            "retrieval_history": retrieval_rows,
        }

        self.save(result, growth_df, retrieval_df)
        return result

    def save(self, result: dict[str, Any], growth_df: pd.DataFrame, retrieval_df: pd.DataFrame) -> None:
        Path("results/tables").mkdir(parents=True, exist_ok=True)
        Path("results/figures").mkdir(parents=True, exist_ok=True)
        Path("results/reports").mkdir(parents=True, exist_ok=True)

        # 1. Save CSV tables
        growth_df.to_csv("results/tables/memory_growth.csv", index=False)
        retrieval_df.to_csv("results/tables/retrieval.csv", index=False)

        # Summary table
        summary_rows = []
        for sys_name, stats in result["final_memory_counts"].items():
            acc = result["comparative_metrics"]["retrieval_accuracy_summary"].get(sys_name, {})
            summary_rows.append({
                "System": sys_name,
                "Total Memories": stats["total"],
                "Active Memories": stats["active"],
                "Archived Memories": stats["archived"],
                "Forgotten Memories": stats["forgotten"],
                "Mean Precision@k": acc.get("mean_precision_at_k"),
                "Mean Recall@k": acc.get("mean_recall_at_k"),
                "Mean MRR": acc.get("mean_mrr"),
                "Mean Retrieval Latency (s)": acc.get("mean_latency"),
            })
        pd.DataFrame(summary_rows).to_csv("results/tables/metrics_summary.csv", index=False)

        # 2. Save JSON Reports
        Path("results/reports/experiment_config.json").write_text(
            json.dumps(result["configuration"], indent=2), encoding="utf-8"
        )
        Path("results/reports/evaluation_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

        # 3. Generate All 6 Plots
        if plt is not None:
            self._generate_plots(growth_df, retrieval_df, result["comparative_metrics"]["lamm_lifecycle_operations"])

    def _generate_plots(self, growth_df: pd.DataFrame, retrieval_df: pd.DataFrame, lifecycle_ops: dict[str, int]) -> None:
        # Plot 1: Total Memory Growth over conversation turns
        fig, ax = plt.subplots(figsize=(8, 5))
        for sys_name in ["LAMM", "UNMANAGED_MEMORY", "RECENCY_ONLY"]:
            sub = growth_df[growth_df["system"] == sys_name]
            ax.plot(sub["turn"], sub["total"], marker="o", label=sys_name, linewidth=2)
        ax.set_title("Total Memory Growth Across Conversation Turns", fontsize=12, fontweight="bold")
        ax.set_xlabel("Conversation Turn", fontsize=10)
        ax.set_ylabel("Total Stored Memories", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig("results/figures/memory_growth.png", dpi=150)
        plt.close(fig)

        # Plot 2: Active Memory Count over turns
        fig, ax = plt.subplots(figsize=(8, 5))
        for sys_name in ["LAMM", "UNMANAGED_MEMORY", "RECENCY_ONLY"]:
            sub = growth_df[growth_df["system"] == sys_name]
            ax.plot(sub["turn"], sub["active"], marker="s", label=sys_name, linewidth=2)
        ax.set_title("Active Retrievable Memories Over Turns", fontsize=12, fontweight="bold")
        ax.set_xlabel("Conversation Turn", fontsize=10)
        ax.set_ylabel("Active Memory Count", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig("results/figures/active_memory_count.png", dpi=150)
        plt.close(fig)

        # Plot 3: Token / Context Consumption over turns
        fig, ax = plt.subplots(figsize=(8, 5))
        for sys_name in ["LAMM", "UNMANAGED_MEMORY", "RECENCY_ONLY"]:
            sub = growth_df[growth_df["system"] == sys_name]
            ax.plot(sub["turn"], sub["context_tokens"], marker="^", label=sys_name, linewidth=2)
        ax.set_title("Context Token Consumption Over Conversation Turns", fontsize=12, fontweight="bold")
        ax.set_xlabel("Conversation Turn", fontsize=10)
        ax.set_ylabel("Estimated Context Tokens", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig("results/figures/token_context_consumption.png", dpi=150)
        plt.close(fig)

        # Plot 4: Retrieval Latency Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        lat_data = [
            retrieval_df[retrieval_df["system"] == s]["retrieval_latency_seconds"].values * 1000.0
            for s in ["LAMM", "UNMANAGED_MEMORY", "RECENCY_ONLY"]
        ]
        ax.boxplot(lat_data, tick_labels=["LAMM", "UNMANAGED", "RECENCY_ONLY"], patch_artist=True)
        ax.set_title("Semantic Retrieval Latency Comparison (ms)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Latency (ms)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        fig.savefig("results/figures/retrieval_latency.png", dpi=150)
        plt.close(fig)

        # Plot 5: LAMM Lifecycle Operations Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ops = list(lifecycle_ops.keys())
        counts = list(lifecycle_ops.values())
        ax.bar(ops, counts, color="#2b5c8f", width=0.55)
        ax.set_title("LAMM Adaptive Lifecycle Operation Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Lifecycle Operation", fontsize=10)
        ax.set_ylabel("Count of Decisions", fontsize=10)
        for i, v in enumerate(counts):
            ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        fig.tight_layout()
        fig.savefig("results/figures/lifecycle_operations.png", dpi=150)
        plt.close(fig)

        # Plot 6: Retrieval Quality Comparison (Precision@k, Recall@k, MRR)
        fig, ax = plt.subplots(figsize=(8, 5))
        gt = retrieval_df[retrieval_df["ground_truth_available"] == True]
        if not gt.empty:
            agg = gt.groupby("system")[["precision_at_k", "recall_at_k", "mrr"]].mean()
            agg.plot(kind="bar", ax=ax, width=0.6)
            ax.set_title("Retrieval Quality Comparison on Ground-Truth Queries", fontsize=12, fontweight="bold")
            ax.set_xlabel("System", fontsize=10)
            ax.set_ylabel("Score (0.0 to 1.0)", fontsize=10)
            ax.set_ylim(0.0, 1.15)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            ax.legend(["Precision@k", "Recall@k", "MRR"])
            fig.tight_layout()
            fig.savefig("results/figures/retrieval_quality.png", dpi=150)
            plt.close(fig)
