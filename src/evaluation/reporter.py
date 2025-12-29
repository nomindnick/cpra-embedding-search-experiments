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
    lines.append(f"| Precision | {results.precision:.2%} |")
    lines.append(f"| Recall | {results.recall:.2%} |")
    lines.append(f"| F1 | {results.f1:.2%} |")
    lines.append(f"| Average Precision | {results.average_precision:.4f} |")
    lines.append(f"| True Positives | {results.true_positives:,} |")
    lines.append(f"| False Positives | {results.false_positives:,} |")
    lines.append(f"| False Negatives | {results.false_negatives:,} |")
    lines.append(f"| Total Emails | {results.total_emails:,} |")
    lines.append(f"| Total Documents | {results.total_documents:,} |")
    lines.append(f"| Total Responsive | {results.total_responsive:,} |")
    lines.append(f"| Total Predicted | {results.total_predicted:,} |")
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
    if results.k_values and results.ranked_metrics:
        lines.append("## Precision@K / Recall@K")
        lines.append("")
        lines.append("| K | Precision@K | Recall@K |")
        lines.append("|---|-------------|----------|")

        for k in results.k_values:
            p_at_k = results.ranked_metrics.get(f"precision_at_{k}", 0)
            r_at_k = results.ranked_metrics.get(f"recall_at_{k}", 0)
            lines.append(f"| {k} | {p_at_k:.2%} | {r_at_k:.2%} |")

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
    lines.append(f"**Pipeline:** {results.pipeline_name}")
    lines.append("")
    lines.append("**Results:**")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Precision | {results.precision:.2%} |")
    lines.append(f"| Recall | {results.recall:.2%} |")
    lines.append(f"| F1 | {results.f1:.2%} |")
    lines.append(f"| Average Precision | {results.average_precision:.4f} |")
    lines.append("")

    # Challenge breakdown
    if results.by_challenge:
        lines.append("**By Challenge Type:**")
        lines.append("")
        lines.append("| Challenge Type | Precision | Recall | F1 |")
        lines.append("|----------------|-----------|--------|-----|")
        for ch in results.by_challenge:
            lines.append(
                f"| {ch.challenge_type.value} | {ch.precision:.2%} | "
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
        "metrics": {
            "precision": results.precision,
            "recall": results.recall,
            "f1": results.f1,
            "average_precision": results.average_precision,
            "true_positives": results.true_positives,
            "false_positives": results.false_positives,
            "false_negatives": results.false_negatives,
        },
        "totals": {
            "emails": results.total_emails,
            "documents": results.total_documents,
            "responsive": results.total_responsive,
            "predicted": results.total_predicted,
        },
        "ranked_metrics": results.ranked_metrics,
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
