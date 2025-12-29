#!/usr/bin/env python3
"""Compare results from multiple experiments and generate a summary table."""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


def load_results(results_dir: Path) -> dict | None:
    """Load results.json from an experiment directory."""
    results_file = results_dir / "results.json"
    if not results_file.exists():
        return None
    with open(results_file) as f:
        return json.load(f)


def find_best_threshold(results: dict) -> dict | None:
    """Find the threshold with best F1 from threshold analysis."""
    threshold_analysis = results.get("threshold_analysis", [])
    if not threshold_analysis:
        return None
    return max(threshold_analysis, key=lambda x: x["f1"])


def find_threshold_meeting_recall(results: dict, min_recall: float = 0.94) -> dict | None:
    """Find the best F1 threshold that still meets minimum recall requirement."""
    threshold_analysis = results.get("threshold_analysis", [])
    if not threshold_analysis:
        return None

    # Filter thresholds meeting recall requirement
    meeting_recall = [t for t in threshold_analysis if t["recall"] >= min_recall]
    if not meeting_recall:
        return None

    # Return the one with best F1 among those meeting recall
    return max(meeting_recall, key=lambda x: x["f1"])


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.94,
        help="Minimum recall requirement (default: 0.94)",
    )
    args = parser.parse_args()

    console = Console()
    results_path = Path(args.results_dir)

    if not results_path.exists():
        console.print(f"[red]Results directory not found: {results_path}[/red]")
        return

    # Find all experiment directories
    experiments = []
    for exp_dir in sorted(results_path.iterdir()):
        if exp_dir.is_dir():
            results = load_results(exp_dir)
            if results:
                experiments.append((exp_dir.name, results))

    if not experiments:
        console.print("[yellow]No experiment results found[/yellow]")
        return

    # Main comparison table
    table = Table(title=f"Model Comparison (Corpus: v2 Primary)")
    table.add_column("Experiment", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right", style="green")
    table.add_column("AP", justify="right")
    table.add_column(f"Meets {args.min_recall:.0%}?", justify="center")

    for name, results in experiments:
        precision = results.get("precision", 0)
        recall = results.get("recall", 0)
        f1 = results.get("f1", 0)
        ap = results.get("average_precision", 0)
        meets_recall = recall >= args.min_recall

        table.add_row(
            name,
            f"{precision:.2%}",
            f"{recall:.2%}",
            f"{f1:.2%}",
            f"{ap:.4f}",
            "[green]Yes[/green]" if meets_recall else "[red]No[/red]",
        )

    console.print(table)
    console.print()

    # Best threshold table (for embedding models with threshold analysis)
    threshold_table = Table(title="Best Threshold Analysis (by F1)")
    threshold_table.add_column("Experiment", style="cyan")
    threshold_table.add_column("Best Threshold", justify="right")
    threshold_table.add_column("Precision", justify="right")
    threshold_table.add_column("Recall", justify="right")
    threshold_table.add_column("F1", justify="right", style="green")

    has_threshold_data = False
    for name, results in experiments:
        best = find_best_threshold(results)
        if best:
            has_threshold_data = True
            threshold_table.add_row(
                name,
                f"{best['threshold']:.2f}",
                f"{best['precision']:.2%}",
                f"{best['recall']:.2%}",
                f"{best['f1']:.2%}",
            )

    if has_threshold_data:
        console.print(threshold_table)
        console.print()

    # Threshold meeting recall requirement
    recall_table = Table(
        title=f"Best Threshold Meeting {args.min_recall:.0%} Recall Requirement"
    )
    recall_table.add_column("Experiment", style="cyan")
    recall_table.add_column("Threshold", justify="right")
    recall_table.add_column("Precision", justify="right")
    recall_table.add_column("Recall", justify="right")
    recall_table.add_column("F1", justify="right", style="green")

    has_recall_data = False
    for name, results in experiments:
        best = find_threshold_meeting_recall(results, args.min_recall)
        if best:
            has_recall_data = True
            recall_table.add_row(
                name,
                f"{best['threshold']:.2f}",
                f"{best['precision']:.2%}",
                f"{best['recall']:.2%}",
                f"{best['f1']:.2%}",
            )

    if has_recall_data:
        console.print(recall_table)
        console.print()

    # Challenge type breakdown for best model
    console.print("[bold]Challenge Type Performance (First Experiment):[/bold]")
    first_name, first_results = experiments[0]
    challenge_data = first_results.get("by_challenge", [])

    if challenge_data:
        challenge_table = Table(title=f"Challenge Breakdown: {first_name}")
        challenge_table.add_column("Challenge Type", style="cyan")
        challenge_table.add_column("Precision", justify="right")
        challenge_table.add_column("Recall", justify="right")
        challenge_table.add_column("F1", justify="right")
        challenge_table.add_column("Total", justify="right")
        challenge_table.add_column("Correct", justify="right")

        for ch in challenge_data:
            challenge_table.add_row(
                ch["challenge_type"],
                f"{ch['precision']:.2%}",
                f"{ch['recall']:.2%}",
                f"{ch['f1']:.2%}",
                str(ch["total_emails"]),
                str(ch["correctly_identified"]),
            )

        console.print(challenge_table)


if __name__ == "__main__":
    main()
