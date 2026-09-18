from __future__ import annotations

import json
from pathlib import Path


def write_markdown_report(json_path: str = "results/reports/evaluation_report.json") -> Path:
    data = json.loads(Path(json_path).read_text())
    output = Path("results/reports/evaluation_report.md")
    lines = [
        "# LAMM Evaluation Report",
        "",
        "## Methodology",
        "The runner compares LAMM with an unmanaged memory baseline on synthetic demo data.",
        "",
        "## Configuration",
        f"Dataset: `{data['dataset']}`",
        "",
        "## Baseline",
        "UNMANAGED_MEMORY stores every extracted candidate without lifecycle management.",
        "",
        "## Measured Results",
        f"LAMM stats: `{data['lamm_stats']}`",
        f"Baseline stats: `{data['baseline_stats']}`",
        f"Lifecycle operations: `{data['lifecycle_operation_counts']}`",
        "",
        "## Observations",
        "These observations are based only on the generated synthetic run.",
        "",
        "## Limitations",
        "Synthetic data is for development and demonstration, not a benchmark.",
    ]
    output.write_text("\n".join(lines))
    return output
