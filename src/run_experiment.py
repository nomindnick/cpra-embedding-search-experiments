"""CLI entry point for running experiments."""

import argparse
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from src.data import load_corpus
from src.evaluation import Evaluator, format_results_table, save_results
from src.pipeline import KeywordSearchPipeline

# Default corpus path
DEFAULT_CORPUS = "corpus/primary"


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_pipeline(config: dict):
    """Create the appropriate pipeline based on config."""
    method = config.get("pipeline", {}).get("method", "keyword")

    if method == "keyword":
        kw_config = config.get("pipeline", {}).get("keyword_search", {})
        return KeywordSearchPipeline(
            match_mode=kw_config.get("match_mode", "any"),
            case_sensitive=kw_config.get("case_sensitive", False),
        )
    elif method == "embedding":
        # Import here to avoid loading models if not needed
        from src.pipeline.embedding import EmbeddingSearchPipeline

        model_name = config.get("embedding_model", "st:all-mpnet-base-v2")
        return EmbeddingSearchPipeline(model_name=model_name)
    else:
        raise ValueError(f"Unknown pipeline method: {method}")


def print_threshold_analysis(threshold_metrics, console: Console):
    """Print threshold analysis table."""
    if not threshold_metrics:
        return

    table = Table(title="Threshold Analysis")
    table.add_column("Threshold", style="cyan", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right", style="green")
    table.add_column("Predicted", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")

    # Find best F1 for highlighting
    best_f1 = max(tm.f1 for tm in threshold_metrics)

    for tm in threshold_metrics:
        f1_style = "bold green" if tm.f1 == best_f1 else ""
        table.add_row(
            f"{tm.threshold:.2f}",
            f"{tm.precision:.2%}",
            f"{tm.recall:.2%}",
            f"[{f1_style}]{tm.f1:.2%}[/{f1_style}]" if f1_style else f"{tm.f1:.2%}",
            str(tm.total_predicted),
            str(tm.true_positives),
            str(tm.false_positives),
            str(tm.false_negatives),
        )

    console.print(table)
    console.print()

    # Print recommendation
    best_tm = max(threshold_metrics, key=lambda x: x.f1)
    console.print(
        f"[bold]Best F1 ({best_tm.f1:.2%}) at threshold {best_tm.threshold:.2f}[/bold]"
    )
    console.print(
        f"  Precision: {best_tm.precision:.2%}, Recall: {best_tm.recall:.2%}"
    )


def print_results_summary(results, console: Console):
    """Print a summary of results to console."""
    console.print()
    console.print(f"[bold green]Experiment: {results.experiment_name}[/bold green]")
    console.print(f"Pipeline: {results.pipeline_name}")
    console.print()

    # Overall metrics table
    overall_table = Table(title="Overall Results")
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="green")

    overall_table.add_row("Precision", f"{results.precision:.2%}")
    overall_table.add_row("Recall", f"{results.recall:.2%}")
    overall_table.add_row("F1", f"{results.f1:.2%}")
    overall_table.add_row("Average Precision", f"{results.average_precision:.4f}")
    overall_table.add_row("True Positives", str(results.true_positives))
    overall_table.add_row("False Positives", str(results.false_positives))
    overall_table.add_row("False Negatives", str(results.false_negatives))
    overall_table.add_row("Total Predicted", str(results.total_predicted))
    overall_table.add_row("Total Responsive", str(results.total_responsive))

    console.print(overall_table)
    console.print()

    # Challenge type table
    if results.by_challenge:
        challenge_table = Table(title="Results by Challenge Type")
        challenge_table.add_column("Challenge", style="cyan")
        challenge_table.add_column("Precision", justify="right")
        challenge_table.add_column("Recall", justify="right")
        challenge_table.add_column("F1", justify="right")
        challenge_table.add_column("Total", justify="right")
        challenge_table.add_column("Correct", justify="right")

        for ch in results.by_challenge:
            challenge_table.add_row(
                ch.challenge_type.value,
                f"{ch.precision:.2%}",
                f"{ch.recall:.2%}",
                f"{ch.f1:.2%}",
                str(ch.total_emails),
                str(ch.correctly_identified),
            )

        console.print(challenge_table)


def main():
    parser = argparse.ArgumentParser(description="Run CPRA responsiveness experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=DEFAULT_CORPUS,
        help="Path to corpus directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: results/<experiment_name>)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for binary classification",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )

    args = parser.parse_args()
    console = Console()

    # Load config
    config = load_config(args.config)
    experiment_name = config.get("name", Path(args.config).stem)

    if not args.quiet:
        console.print(f"[bold]Loading corpus from {args.corpus}...[/bold]")

    # Load corpus
    corpus = load_corpus(args.corpus)

    if not args.quiet:
        console.print(f"Loaded {corpus.num_emails:,} emails")
        console.print(f"  {corpus.num_searchable_documents:,} searchable documents")
        console.print(f"  {corpus.num_threads} threads")
        console.print(f"Request: {corpus.request.title}")
        console.print()
        console.print("[bold]Creating pipeline...[/bold]")

    # Create pipeline
    pipeline = create_pipeline(config)

    if not args.quiet:
        console.print(f"Pipeline: {pipeline.name}")
        console.print()
        console.print("[bold]Running evaluation...[/bold]")

    # Get k_values from config
    eval_config = config.get("evaluation", {})
    k_values = eval_config.get("k_values", [50, 100, 200])
    thresholds = eval_config.get("thresholds", [])

    # Run evaluation
    evaluator = Evaluator(corpus, k_values=k_values)
    results = evaluator.evaluate(
        pipeline,
        experiment_name=experiment_name,
        threshold=args.threshold,
    )
    results.config = config

    # Run threshold analysis if thresholds are configured
    if thresholds and config.get("pipeline", {}).get("method") == "embedding":
        if not args.quiet:
            console.print()
            console.print("[bold]Running threshold analysis...[/bold]")
        threshold_metrics = evaluator.evaluate_thresholds(pipeline, thresholds)
        results.threshold_analysis = threshold_metrics

    # Print results
    if not args.quiet:
        print_results_summary(results, console)

        # Print threshold analysis if available
        if results.threshold_analysis:
            console.print()
            print_threshold_analysis(results.threshold_analysis, console)

    # Save results
    output_dir = args.output_dir or f"results/{experiment_name.lower().replace(' ', '_')}"
    saved = save_results(results, output_dir)

    if not args.quiet:
        console.print()
        console.print("[bold green]Results saved:[/bold green]")
        for file_type, path in saved.items():
            console.print(f"  {file_type}: {path}")

    # Also save markdown summary
    summary_path = Path(output_dir) / "summary.md"
    with open(summary_path, "w") as f:
        f.write(format_results_table(results))

    return results


if __name__ == "__main__":
    main()
