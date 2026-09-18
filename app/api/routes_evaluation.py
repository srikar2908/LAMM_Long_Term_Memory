from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.evaluation.report import write_markdown_report
from app.evaluation.runner import EvaluationRunner

router = APIRouter()


@router.post("/evaluation/run")
def run_evaluation():
    result = EvaluationRunner().run()
    write_markdown_report()
    return result


@router.get("/evaluation/results")
def evaluation_results():
    path = Path("results/reports/evaluation_report.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No evaluation results found. Run /evaluation/run first.")
    return path.read_text()
