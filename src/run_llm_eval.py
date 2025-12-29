#!/usr/bin/env python3
"""Run LLM capability assessment (EXP-000).

This script evaluates local LLMs on various tasks to determine which models
to use for subsequent experiments.

Usage:
    # Evaluate all available text models on all tasks
    python -m src.run_llm_eval --corpus corpus/primary

    # Evaluate specific models
    python -m src.run_llm_eval --corpus corpus/primary --models qwen3:8b gemma3:4b

    # Evaluate only classification tasks
    python -m src.run_llm_eval --corpus corpus/primary --tasks classification_binary classification_ternary

    # Quick test with fewer samples
    python -m src.run_llm_eval --corpus corpus/primary --samples 2
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .llm_eval import LLMEvaluator, TASKS
from .llm_eval.models import get_text_models


console = Console()


def print_summary(result, evaluator):
    """Print a summary of the evaluation results."""
    console.print("\n")
    console.print(Panel.fit("[bold]LLM Evaluation Summary[/bold]", style="blue"))

    # Models evaluated
    console.print(f"\n[bold]Models evaluated:[/bold] {len(result.models_evaluated)}")
    console.print(f"[bold]Sample size:[/bold] {result.sample_size} documents")

    # Results table
    table = Table(title="\nModel Performance by Task Category")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="dim")
    table.add_column("Classification", justify="right")
    table.add_column("Extraction", justify="right")
    table.add_column("Generation", justify="right")
    table.add_column("Avg Latency", justify="right")

    for model_eval in result.model_results:
        summary = model_eval.summary.get("by_task", {})

        # Classification accuracy
        class_tasks = [
            "Binary Classification",
            "Ternary Classification with Confidence",
            "JSON Structured Output",
        ]
        class_accs = []
        for task_name in class_tasks:
            if task_name in summary and "accuracy" in summary[task_name]:
                class_accs.append(summary[task_name]["accuracy"])
        class_str = f"{sum(class_accs)/len(class_accs)*100:.0f}%" if class_accs else "—"

        # Extraction metrics
        ext_tasks = ["Evidence Extraction", "Keyword Extraction"]
        ext_scores = []
        for task_name in ext_tasks:
            if task_name in summary:
                if "avg_extraction_accuracy" in summary[task_name]:
                    ext_scores.append(summary[task_name]["avg_extraction_accuracy"])
                elif "avg_format_complete" in summary[task_name]:
                    ext_scores.append(1.0 if summary[task_name]["avg_format_complete"] else 0.0)
        ext_str = f"{sum(ext_scores)/len(ext_scores)*100:.0f}%" if ext_scores else "—"

        # Generation metrics
        gen_tasks = [
            "Positive Example Generation",
            "Negative Example Generation",
            "Paraphrase Generation",
        ]
        gen_scores = []
        for task_name in gen_tasks:
            if task_name in summary:
                if "avg_structure_complete" in summary[task_name]:
                    gen_scores.append(1.0 if summary[task_name]["avg_structure_complete"] else 0.0)
                elif "avg_meets_target" in summary[task_name]:
                    gen_scores.append(1.0 if summary[task_name]["avg_meets_target"] else 0.0)
        gen_str = f"{sum(gen_scores)/len(gen_scores)*100:.0f}%" if gen_scores else "—"

        # Latency
        avg_latency = model_eval.summary.get("avg_latency", 0)
        latency_str = f"{avg_latency:.2f}s"

        table.add_row(
            model_eval.model_name,
            model_eval.model_size,
            class_str,
            ext_str,
            gen_str,
            latency_str,
        )

    console.print(table)

    # Recommendations
    if result.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for category, models in result.recommendations.items():
            if models:
                console.print(f"  [cyan]{category.title()}:[/cyan] {', '.join(models)}")


def print_detailed_results(result):
    """Print detailed per-task results."""
    console.print("\n")
    console.print(Panel.fit("[bold]Detailed Results by Task[/bold]", style="green"))

    for model_eval in result.model_results:
        console.print(f"\n[bold cyan]{model_eval.model_name}[/bold cyan] ({model_eval.model_size})")

        summary = model_eval.summary.get("by_task", {})
        for task_name, task_summary in summary.items():
            console.print(f"\n  [yellow]{task_name}[/yellow]")

            # Print key metrics
            for key, value in task_summary.items():
                if key in ("count", "errors"):
                    continue
                if isinstance(value, float):
                    if "accuracy" in key or "diversity" in key:
                        console.print(f"    {key}: {value*100:.1f}%")
                    elif "latency" in key:
                        console.print(f"    {key}: {value:.2f}s")
                    else:
                        console.print(f"    {key}: {value:.2f}")
                else:
                    console.print(f"    {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local LLMs for CPRA document classification tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="corpus/primary",
        help="Path to corpus directory (default: corpus/primary)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific models to evaluate (default: all text models)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=list(TASKS.keys()),
        help="Specific tasks to run (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of documents per category to sample (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/llm_eval",
        help="Output directory for results (default: results/llm_eval)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-task results",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available text models and exit",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit",
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_models:
        console.print("[bold]Available Text Models:[/bold]\n")
        models = get_text_models()
        table = Table()
        table.add_column("Model", style="cyan")
        table.add_column("Size", justify="right")
        for m in sorted(models, key=lambda x: x.size_bytes):
            table.add_row(m.name, m.size_human)
        console.print(table)
        return 0

    if args.list_tasks:
        console.print("[bold]Available Tasks:[/bold]\n")
        table = Table()
        table.add_column("Task ID", style="cyan")
        table.add_column("Name")
        table.add_column("Description")
        for task_id, task in TASKS.items():
            table.add_row(task_id, task.name, task.description)
        console.print(table)
        return 0

    # Validate corpus path
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        console.print(f"[red]Error: Corpus path not found: {corpus_path}[/red]")
        return 1

    # Validate models
    if args.models:
        available = {m.name for m in get_text_models()}
        for model in args.models:
            if model not in available:
                console.print(f"[red]Error: Model not found: {model}[/red]")
                console.print(f"Available models: {', '.join(sorted(available))}")
                return 1

    # Run evaluation
    console.print(Panel.fit("[bold]EXP-000: Local LLM Capability Assessment[/bold]"))
    console.print(f"\nCorpus: {corpus_path}")
    console.print(f"Output: {args.output}")

    evaluator = LLMEvaluator(corpus_path, args.output)

    try:
        result = evaluator.run_full_evaluation(
            models=args.models,
            n_per_category=args.samples,
            tasks=args.tasks,
            verbose=True,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted[/yellow]")
        return 1

    # Save results
    output_path = evaluator.save_results(result)
    console.print(f"\n[green]Results saved to: {output_path}[/green]")

    # Print summary
    print_summary(result, evaluator)

    if args.detailed:
        print_detailed_results(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
