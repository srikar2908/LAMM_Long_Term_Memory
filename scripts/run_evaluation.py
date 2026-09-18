import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.report import write_markdown_report
from app.evaluation.runner import EvaluationRunner


if __name__ == "__main__":
    result = EvaluationRunner().run()
    report = write_markdown_report()
    print("Evaluation complete.")
    print(result["note"])
    print("JSON: results/reports/evaluation_report.json")
    print(f"Markdown: {report}")
