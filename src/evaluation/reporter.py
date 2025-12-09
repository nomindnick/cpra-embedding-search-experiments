"""Output formatting and reporting for evaluation results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .evaluator import EvaluationResult


def format_results_table(results: EvaluationResult) -> str:
    """Format evaluation results as markdown tables.

    Args:
        results: Evaluation results to format

    Returns:
        Markdown formatted string with results tables
    """
    lines = []

    # Header
    lines.append(f"# {results.experiment_name}")
    lines.append("")
    lines.append(f"**Pipeline:** {results.pipeline_name}")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Overall metrics
    lines.append("## Overall Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Precision | {results.overall_precision:.2%} |")
    lines.append(f"| Recall | {results.overall_recall:.2%} |")
    lines.append(f"| F1 | {results.overall_f1:.2%} |")
    lines.append(f"| MAP | {results.mean_average_precision:.4f} |")
    lines.append(f"| Total Emails | {results.total_emails:,} |")
    lines.append(f"| Total Responsive | {results.total_responsive:,} |")
    lines.append(f"| Total Predicted | {results.total_predicted:,} |")
    lines.append("")

    # Per-request breakdown
    lines.append("## Results by CPRA Request")
    lines.append("")
    lines.append("| Request | Precision | Recall | F1 | TP | FP | FN |")
    lines.append("|---------|-----------|--------|----|----|----|----|")
    for req in results.by_request:
        lines.append(
            f"| {req.request_title} | {req.precision:.2%} | {req.recall:.2%} | "
            f"{req.f1:.2%} | {req.true_positives} | {req.false_positives} | "
            f"{req.false_negatives} |"
        )
    lines.append("")

    # Per-challenge breakdown
    if results.by_challenge:
        lines.append("## Results by Challenge Type")
        lines.append("")
        lines.append("| Challenge Type | Precision | Recall | F1 | Total | Correct |")
        lines.append("|----------------|-----------|--------|----| ------|---------|")
        for ch in results.by_challenge:
            lines.append(
                f"| {ch.challenge_type.value} | {ch.precision:.2%} | {ch.recall:.2%} | "
                f"{ch.f1:.2%} | {ch.total_emails} | {ch.correctly_identified} |"
            )
        lines.append("")

    # Ranked metrics (if available)
    if results.k_values and results.by_request:
        lines.append("## Precision@K / Recall@K")
        lines.append("")

        # Build header
        header = "| Request |"
        separator = "|---------|"
        for k in results.k_values:
            header += f" P@{k} | R@{k} |"
            separator += "------|------|"

        lines.append(header)
        lines.append(separator)

        for req in results.by_request:
            row = f"| {req.request_title} |"
            for k in results.k_values:
                p_at_k = req.ranked_metrics.get(f"precision_at_{k}", 0)
                r_at_k = req.ranked_metrics.get(f"recall_at_{k}", 0)
                row += f" {p_at_k:.2%} | {r_at_k:.2%} |"
            lines.append(row)

        lines.append("")

    # Threshold analysis (if available)
    if results.threshold_analysis:
        lines.append("## Threshold Analysis")
        lines.append("")
        lines.append("| Threshold | Precision | Recall | F1 | Predicted | TP | FP | FN |")
        lines.append("|-----------|-----------|--------|----|-----------|----|----|----|")

        best_f1 = max(tm.f1 for tm in results.threshold_analysis)
        for tm in results.threshold_analysis:
            marker = " **" if tm.f1 == best_f1 else ""
            end_marker = "**" if tm.f1 == best_f1 else ""
            lines.append(
                f"| {marker}{tm.threshold:.2f}{end_marker} | {tm.precision:.2%} | "
                f"{tm.recall:.2%} | {marker}{tm.f1:.2%}{end_marker} | {tm.total_predicted} | "
                f"{tm.true_positives} | {tm.false_positives} | {tm.false_negatives} |"
            )

        lines.append("")

        # Add recommendation
        best_tm = max(results.threshold_analysis, key=lambda x: x.f1)
        lines.append(
            f"**Best F1 ({best_tm.f1:.2%}) at threshold {best_tm.threshold:.2f}** "
            f"— Precision: {best_tm.precision:.2%}, Recall: {best_tm.recall:.2%}"
        )
        lines.append("")

    return "\n".join(lines)


def format_log_entry(results: EvaluationResult) -> str:
    """Format results as a log entry for LOG.md.

    Args:
        results: Evaluation results

    Returns:
        Markdown section for the experiment log
    """
    lines = []
    lines.append(f"### {results.experiment_name}")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("**Results:**")
    lines.append("")
    lines.append("| Metric       | Overall |", end="")

    # Add per-request headers
    for req in results.by_request:
        short_name = req.request_title.split()[0]  # First word
        lines[-1] += f" {short_name} |"
    lines.append("")

    lines.append("| ------------ | ------- |", end="")
    for _ in results.by_request:
        lines[-1] += " --- |"
    lines.append("")

    # Precision row
    row = f"| Precision    | {results.overall_precision:.2%} |"
    for req in results.by_request:
        row += f" {req.precision:.2%} |"
    lines.append(row)

    # Recall row
    row = f"| Recall       | {results.overall_recall:.2%} |"
    for req in results.by_request:
        row += f" {req.recall:.2%} |"
    lines.append(row)

    # F1 row
    row = f"| F1           | {results.overall_f1:.2%} |"
    for req in results.by_request:
        row += f" {req.f1:.2%} |"
    lines.append(row)

    lines.append("")

    # Challenge breakdown
    if results.by_challenge:
        lines.append("**By Challenge Type:**")
        lines.append("")
        lines.append("| Challenge Type     | Precision | Recall | F1  |")
        lines.append("| ------------------ | --------- | ------ | --- |")
        for ch in results.by_challenge:
            lines.append(
                f"| {ch.challenge_type.value:18} | {ch.precision:.2%}    | "
                f"{ch.recall:.2%} | {ch.f1:.2%} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def results_to_dict(results: EvaluationResult) -> dict[str, Any]:
    """Convert results to a JSON-serializable dictionary.

    Args:
        results: Evaluation results

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        "experiment_name": results.experiment_name,
        "pipeline_name": results.pipeline_name,
        "timestamp": datetime.now().isoformat(),
        "overall": {
            "precision": results.overall_precision,
            "recall": results.overall_recall,
            "f1": results.overall_f1,
            "map": results.mean_average_precision,
        },
        "totals": {
            "emails": results.total_emails,
            "responsive": results.total_responsive,
            "predicted": results.total_predicted,
        },
        "by_request": [
            {
                "request_id": req.request_id,
                "request_title": req.request_title,
                "precision": req.precision,
                "recall": req.recall,
                "f1": req.f1,
                "true_positives": req.true_positives,
                "false_positives": req.false_positives,
                "false_negatives": req.false_negatives,
                "total_responsive": req.total_responsive,
                "ranked_metrics": req.ranked_metrics,
            }
            for req in results.by_request
        ],
        "by_challenge": [
            {
                "challenge_type": ch.challenge_type.value,
                "precision": ch.precision,
                "recall": ch.recall,
                "f1": ch.f1,
                "total_emails": ch.total_emails,
                "correctly_identified": ch.correctly_identified,
            }
            for ch in results.by_challenge
        ],
        "threshold_analysis": [
            {
                "threshold": tm.threshold,
                "precision": tm.precision,
                "recall": tm.recall,
                "f1": tm.f1,
                "total_predicted": tm.total_predicted,
                "true_positives": tm.true_positives,
                "false_positives": tm.false_positives,
                "false_negatives": tm.false_negatives,
            }
            for tm in results.threshold_analysis
        ] if results.threshold_analysis else [],
        "k_values": results.k_values,
        "config": results.config,
    }


def save_results(
    results: EvaluationResult,
    output_dir: str | Path,
    save_summary: bool = True,
    save_json: bool = True,
) -> dict[str, Path]:
    """Save evaluation results to files.

    Args:
        results: Evaluation results to save
        output_dir: Directory to save results to
        save_summary: Whether to save markdown summary
        save_json: Whether to save JSON results

    Returns:
        Dict mapping file type to path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    if save_summary:
        summary_path = output_dir / "summary.md"
        with open(summary_path, "w") as f:
            f.write(format_results_table(results))
        saved_files["summary"] = summary_path

    if save_json:
        json_path = output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(results_to_dict(results), f, indent=2)
        saved_files["json"] = json_path

    return saved_files
