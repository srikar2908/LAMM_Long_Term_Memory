from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.evaluation.report import write_markdown_report
from app.evaluation.runner import EvaluationRunner

router = APIRouter(tags=["Evaluation"])


@router.post("/evaluation/run", summary="Execute 3-way comparative evaluation experiment")
def run_evaluation() -> dict[str, Any]:
    """
    Executes the fair 3-way evaluation:
      - LAMM vs UNMANAGED_MEMORY vs RECENCY_ONLY (LRU)
    Generates tables in results/tables, figures in results/figures, and reports in results/reports.
    """
    result = EvaluationRunner().run()
    write_markdown_report()
    return result


@router.get("/evaluation/results", summary="Retrieve latest evaluation experiment results")
def evaluation_results() -> dict[str, Any]:
    """Returns the JSON metrics and comparative summary from the most recent evaluation run."""
    path = Path("results/reports/evaluation_report.json")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation results found. Execute POST /evaluation/run first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))
