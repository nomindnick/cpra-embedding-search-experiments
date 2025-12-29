"""LLM evaluation logic."""

import json
import re
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..data.corpus import Corpus, ChallengeType, Email, load_corpus
from .models import OllamaModel, get_text_models, ModelInfo
from .tasks import Task, TASKS, TaskType


@dataclass
class TaskResult:
    """Result of running a single task."""

    task_name: str
    model_name: str
    document_id: str | None  # None for generation tasks
    expected: Any  # Expected answer (for classification)
    response: str  # Raw model response
    latency: float  # Seconds
    correct: bool | None  # For classification tasks
    metrics: dict[str, Any] = field(default_factory=dict)  # Task-specific metrics
    error: str | None = None


@dataclass
class ModelEvaluation:
    """Aggregated evaluation results for a model."""

    model_name: str
    model_size: str
    results: list[TaskResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def compute_summary(self) -> None:
        """Compute summary statistics."""
        by_task: dict[str, list[TaskResult]] = {}
        for r in self.results:
            if r.task_name not in by_task:
                by_task[r.task_name] = []
            by_task[r.task_name].append(r)

        self.summary = {
            "total_tasks": len(self.results),
            "total_latency": sum(r.latency for r in self.results),
            "avg_latency": (
                sum(r.latency for r in self.results) / len(self.results)
                if self.results
                else 0
            ),
            "by_task": {},
        }

        for task_name, task_results in by_task.items():
            task_summary = {
                "count": len(task_results),
                "avg_latency": (
                    sum(r.latency for r in task_results) / len(task_results)
                    if task_results
                    else 0
                ),
                "errors": sum(1 for r in task_results if r.error),
            }

            # Add accuracy for classification tasks
            classification_results = [r for r in task_results if r.correct is not None]
            if classification_results:
                task_summary["accuracy"] = (
                    sum(1 for r in classification_results if r.correct)
                    / len(classification_results)
                )

            # Aggregate task-specific metrics
            all_metrics: dict[str, list[float]] = {}
            for r in task_results:
                for k, v in r.metrics.items():
                    if isinstance(v, (int, float)):
                        if k not in all_metrics:
                            all_metrics[k] = []
                        all_metrics[k].append(v)

            for k, values in all_metrics.items():
                task_summary[f"avg_{k}"] = sum(values) / len(values) if values else 0

            self.summary["by_task"][task_name] = task_summary


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    timestamp: str
    corpus_path: str
    models_evaluated: list[str]
    sample_size: int
    model_results: list[ModelEvaluation]
    recommendations: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "corpus_path": self.corpus_path,
            "models_evaluated": self.models_evaluated,
            "sample_size": self.sample_size,
            "model_results": [
                {
                    "model_name": m.model_name,
                    "model_size": m.model_size,
                    "summary": m.summary,
                    "results": [
                        {
                            "task_name": r.task_name,
                            "document_id": r.document_id,
                            "expected": r.expected,
                            "response": r.response[:500],  # Truncate for storage
                            "latency": r.latency,
                            "correct": r.correct,
                            "metrics": r.metrics,
                            "error": r.error,
                        }
                        for r in m.results
                    ],
                }
                for m in self.model_results
            ],
            "recommendations": self.recommendations,
        }


def select_sample_documents(
    corpus: Corpus,
    n_per_category: int = 5,
    seed: int = 42,
) -> list[tuple[Email, bool, ChallengeType]]:
    """Select a balanced sample of documents for evaluation.

    Args:
        corpus: The corpus to sample from
        n_per_category: Number of documents per category
        seed: Random seed for reproducibility

    Returns:
        List of (email, is_responsive, challenge_type) tuples
    """
    random.seed(seed)
    samples = []

    # Categories to sample
    categories = {
        # Clear responsive
        "clear_responsive": [ChallengeType.DIRECT_MATCH],
        # Tricky responsive
        "tricky_responsive": [
            ChallengeType.INDIRECT_REFERENCE,
            ChallengeType.TECHNICAL_JARGON,
            ChallengeType.BURIED_IN_THREAD,
            ChallengeType.AMBIGUOUS_TERMS,
            ChallengeType.TEMPORAL_REFERENCE,
        ],
        # Tricky non-responsive
        "tricky_nonresponsive": [
            ChallengeType.KEYWORD_FALSE_POSITIVE,
            ChallengeType.ADJACENT_TOPIC,
        ],
        # Clear non-responsive
        "clear_nonresponsive": [ChallengeType.TRUE_NEGATIVE],
    }

    for category_name, challenge_types in categories.items():
        # Get all emails matching these challenge types
        category_emails = []
        for ct in challenge_types:
            emails = corpus.get_emails_by_challenge(ct)
            for email in emails:
                is_responsive = corpus.is_responsive(email.id)
                category_emails.append((email, is_responsive, ct))

        # Sample from this category
        if len(category_emails) <= n_per_category:
            samples.extend(category_emails)
        else:
            samples.extend(random.sample(category_emails, n_per_category))

    return samples


# Response evaluation functions


def evaluate_binary_classification(
    response: str,
    expected_responsive: bool,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate binary classification response.

    Returns (is_correct, metrics)
    """
    response_clean = response.strip().upper()

    # Try to extract YES or NO
    if "YES" in response_clean and "NO" not in response_clean:
        predicted = True
    elif "NO" in response_clean and "YES" not in response_clean:
        predicted = False
    elif response_clean.startswith("YES"):
        predicted = True
    elif response_clean.startswith("NO"):
        predicted = False
    else:
        # Unclear response
        return False, {"parse_error": True, "raw_response": response[:100]}

    is_correct = predicted == expected_responsive
    return is_correct, {
        "predicted": predicted,
        "expected": expected_responsive,
        "parse_error": False,
    }


def evaluate_ternary_classification(
    response: str,
    expected_responsive: bool,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate ternary classification response.

    Returns (is_correct, metrics)
    """
    lines = response.strip().split("\n")
    metrics: dict[str, Any] = {"parse_error": False}

    # Parse classification
    if lines:
        class_line = lines[0].strip().upper()
        if "YES" in class_line:
            predicted = "yes"
        elif "MAYBE" in class_line:
            predicted = "maybe"
        elif "NO" in class_line:
            predicted = "no"
        else:
            metrics["parse_error"] = True
            return False, metrics
    else:
        metrics["parse_error"] = True
        return False, metrics

    metrics["predicted_class"] = predicted

    # Parse confidence
    if len(lines) > 1:
        try:
            confidence_str = re.search(r"\d+", lines[1])
            if confidence_str:
                metrics["confidence"] = int(confidence_str.group())
        except (ValueError, IndexError):
            pass

    # Evaluate: yes/maybe → responsive, no → non-responsive
    predicted_responsive = predicted in ("yes", "maybe")
    is_correct = predicted_responsive == expected_responsive
    metrics["expected"] = expected_responsive

    return is_correct, metrics


def evaluate_json_output(
    response: str,
    expected_responsive: bool,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate JSON output response.

    Returns (is_correct, metrics)
    """
    metrics: dict[str, Any] = {"valid_json": False, "parse_error": False}

    # Try to extract JSON from response
    try:
        # Try direct parse first
        data = json.loads(response.strip())
        metrics["valid_json"] = True
    except json.JSONDecodeError:
        # Try to find JSON in response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                metrics["valid_json"] = True
            except json.JSONDecodeError:
                metrics["parse_error"] = True
                return False, metrics
        else:
            metrics["parse_error"] = True
            return False, metrics

    # Check required fields
    metrics["has_responsive"] = "responsive" in data
    metrics["has_confidence"] = "confidence" in data
    metrics["has_category"] = "category" in data
    metrics["has_reasoning"] = "reasoning" in data

    # Evaluate classification
    responsive_val = str(data.get("responsive", "")).lower()
    if responsive_val in ("yes", "true", "1"):
        predicted = True
    elif responsive_val == "maybe":
        predicted = True  # Treat maybe as responsive for recall
    else:
        predicted = False

    is_correct = predicted == expected_responsive
    metrics["predicted"] = predicted
    metrics["expected"] = expected_responsive

    return is_correct, metrics


def evaluate_evidence_extraction(
    response: str,
    document_text: str,
) -> tuple[bool | None, dict[str, Any]]:
    """Evaluate evidence extraction response.

    Returns (None, metrics) - no correct/incorrect for this task
    """
    metrics: dict[str, Any] = {
        "quotes_found": 0,
        "quotes_valid": 0,
        "no_content_response": False,
    }

    if "NO RELEVANT CONTENT" in response.upper():
        metrics["no_content_response"] = True
        return None, metrics

    # Find quoted text
    quotes = re.findall(r'"([^"]+)"', response)
    metrics["quotes_found"] = len(quotes)

    # Check if quotes appear in document
    doc_lower = document_text.lower()
    valid_quotes = 0
    for quote in quotes:
        # Allow some flexibility (lowercase comparison, whitespace normalization)
        quote_normalized = " ".join(quote.lower().split())
        if len(quote_normalized) > 10:  # Ignore very short quotes
            if quote_normalized in doc_lower or quote.lower() in doc_lower:
                valid_quotes += 1

    metrics["quotes_valid"] = valid_quotes
    metrics["extraction_accuracy"] = (
        valid_quotes / len(quotes) if quotes else 0
    )

    return None, metrics


def evaluate_paraphrase_generation(
    response: str,
) -> tuple[bool | None, dict[str, Any]]:
    """Evaluate paraphrase generation response.

    Returns (None, metrics) - no correct/incorrect for this task
    """
    # Count numbered items
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    numbered_lines = [l for l in lines if re.match(r"^\d+[\.\):]", l)]

    metrics = {
        "paraphrases_generated": len(numbered_lines),
        "target_count": 5,
        "meets_target": len(numbered_lines) >= 5,
    }

    # Check diversity (simple: unique words across paraphrases)
    if numbered_lines:
        all_words = set()
        per_paraphrase_words = []
        for line in numbered_lines:
            words = set(re.findall(r"\w+", line.lower()))
            per_paraphrase_words.append(words)
            all_words.update(words)

        # Diversity: average unique words per paraphrase
        if len(numbered_lines) > 1:
            diversity_scores = []
            for i, words in enumerate(per_paraphrase_words):
                other_words = set()
                for j, other in enumerate(per_paraphrase_words):
                    if i != j:
                        other_words.update(other)
                unique_to_this = len(words - other_words)
                diversity_scores.append(unique_to_this / len(words) if words else 0)
            metrics["avg_diversity"] = sum(diversity_scores) / len(diversity_scores)
        else:
            metrics["avg_diversity"] = 0

    return None, metrics


def evaluate_example_generation(
    response: str,
) -> tuple[bool | None, dict[str, Any]]:
    """Evaluate example email generation.

    Returns (None, metrics) - no correct/incorrect for this task
    """
    metrics: dict[str, Any] = {}

    # Check for email structure
    metrics["has_from"] = bool(re.search(r"from:", response, re.IGNORECASE))
    metrics["has_to"] = bool(re.search(r"to:", response, re.IGNORECASE))
    metrics["has_subject"] = bool(re.search(r"subject:", response, re.IGNORECASE))

    # Check body length
    lines = response.split("\n")
    body_start = None
    for i, line in enumerate(lines):
        if line.strip() == "" and i > 0:
            body_start = i + 1
            break

    if body_start and body_start < len(lines):
        body = "\n".join(lines[body_start:])
        metrics["body_word_count"] = len(body.split())
    else:
        metrics["body_word_count"] = len(response.split())

    metrics["structure_complete"] = all(
        [metrics["has_from"], metrics["has_to"], metrics["has_subject"]]
    )

    return None, metrics


def evaluate_keyword_extraction(
    response: str,
) -> tuple[bool | None, dict[str, Any]]:
    """Evaluate keyword extraction response.

    Returns (None, metrics) - no correct/incorrect for this task
    """
    metrics: dict[str, Any] = {}

    # Check for expected sections
    metrics["has_keywords"] = bool(re.search(r"keywords?:", response, re.IGNORECASE))
    metrics["has_entities"] = bool(re.search(r"entit", response, re.IGNORECASE))
    metrics["has_acronyms"] = bool(re.search(r"acronym", response, re.IGNORECASE))

    # Count extracted items
    keywords_match = re.search(r"keywords?:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if keywords_match:
        keywords = [k.strip() for k in keywords_match.group(1).split(",")]
        metrics["keyword_count"] = len([k for k in keywords if k])
    else:
        metrics["keyword_count"] = 0

    metrics["format_complete"] = all(
        [metrics["has_keywords"], metrics["has_entities"], metrics["has_acronyms"]]
    )

    return None, metrics


class LLMEvaluator:
    """Main evaluator for running LLM assessments."""

    def __init__(
        self,
        corpus_path: str | Path,
        output_dir: str | Path = "results/llm_eval",
    ):
        self.corpus = load_corpus(corpus_path)
        self.corpus_path = str(corpus_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model_name: str,
        sample_docs: list[tuple[Email, bool, ChallengeType]],
        tasks: list[str] | None = None,
        verbose: bool = True,
    ) -> ModelEvaluation:
        """Evaluate a single model on all tasks.

        Args:
            model_name: Ollama model name
            sample_docs: Sample documents with labels
            tasks: List of task names to run (None = all)
            verbose: Print progress

        Returns:
            ModelEvaluation with all results
        """
        model = OllamaModel(model_name)
        model_info = next(
            (m for m in get_text_models() if m.name == model_name),
            ModelInfo(model_name, 0, "unknown", ""),
        )

        if verbose:
            print(f"\nEvaluating {model_name} ({model_info.size_human})...")

        results: list[TaskResult] = []
        tasks_to_run = tasks or list(TASKS.keys())

        for task_name in tasks_to_run:
            task = TASKS[task_name]

            if verbose:
                print(f"  Task: {task.name}...", end=" ", flush=True)

            task_results = self._run_task(model, task, sample_docs)
            results.extend(task_results)

            if verbose:
                # Quick summary
                if task.task_type in (
                    TaskType.CLASSIFICATION_BINARY,
                    TaskType.CLASSIFICATION_TERNARY,
                    TaskType.JSON_OUTPUT,
                ):
                    correct = sum(1 for r in task_results if r.correct)
                    print(f"{correct}/{len(task_results)} correct")
                else:
                    avg_latency = (
                        sum(r.latency for r in task_results) / len(task_results)
                        if task_results
                        else 0
                    )
                    print(f"avg {avg_latency:.2f}s")

        evaluation = ModelEvaluation(
            model_name=model_name,
            model_size=model_info.size_human,
            results=results,
        )
        evaluation.compute_summary()

        return evaluation

    def _run_task(
        self,
        model: OllamaModel,
        task: Task,
        sample_docs: list[tuple[Email, bool, ChallengeType]],
    ) -> list[TaskResult]:
        """Run a single task across sample documents."""
        results = []
        request_text = self.corpus.request.request_text

        if task.task_type in (
            TaskType.PARAPHRASE_GENERATION,
            TaskType.EXAMPLE_GENERATION,
        ):
            # These tasks don't need per-document evaluation
            if task.task_type == TaskType.PARAPHRASE_GENERATION:
                prompt = task.format_prompt(request_text=request_text)
                try:
                    response, latency = model.generate(
                        prompt, system_prompt=task.system_prompt
                    )
                    _, metrics = evaluate_paraphrase_generation(response)
                    results.append(
                        TaskResult(
                            task_name=task.name,
                            model_name=model.model_name,
                            document_id=None,
                            expected=None,
                            response=response,
                            latency=latency,
                            correct=None,
                            metrics=metrics,
                        )
                    )
                except Exception as e:
                    results.append(
                        TaskResult(
                            task_name=task.name,
                            model_name=model.model_name,
                            document_id=None,
                            expected=None,
                            response="",
                            latency=0,
                            correct=None,
                            error=str(e),
                        )
                    )
            else:
                # Example generation - run once for positive, once for negative
                prompt = task.format_prompt(request_text=request_text)
                try:
                    response, latency = model.generate(
                        prompt, system_prompt=task.system_prompt
                    )
                    _, metrics = evaluate_example_generation(response)
                    results.append(
                        TaskResult(
                            task_name=task.name,
                            model_name=model.model_name,
                            document_id=None,
                            expected=None,
                            response=response,
                            latency=latency,
                            correct=None,
                            metrics=metrics,
                        )
                    )
                except Exception as e:
                    results.append(
                        TaskResult(
                            task_name=task.name,
                            model_name=model.model_name,
                            document_id=None,
                            expected=None,
                            response="",
                            latency=0,
                            correct=None,
                            error=str(e),
                        )
                    )
        else:
            # Per-document tasks
            for email, is_responsive, challenge_type in sample_docs:
                prompt = task.format_prompt(
                    request_text=request_text,
                    document_text=email.text,
                )

                try:
                    response, latency = model.generate(
                        prompt, system_prompt=task.system_prompt
                    )

                    # Evaluate based on task type
                    if task.task_type == TaskType.CLASSIFICATION_BINARY:
                        correct, metrics = evaluate_binary_classification(
                            response, is_responsive
                        )
                    elif task.task_type == TaskType.CLASSIFICATION_TERNARY:
                        correct, metrics = evaluate_ternary_classification(
                            response, is_responsive
                        )
                    elif task.task_type == TaskType.JSON_OUTPUT:
                        correct, metrics = evaluate_json_output(response, is_responsive)
                    elif task.task_type == TaskType.EVIDENCE_EXTRACTION:
                        correct, metrics = evaluate_evidence_extraction(
                            response, email.text
                        )
                    elif task.task_type == TaskType.KEYWORD_EXTRACTION:
                        correct, metrics = evaluate_keyword_extraction(response)
                    else:
                        correct, metrics = None, {}

                    results.append(
                        TaskResult(
                            task_name=task.name,
                            model_name=model.model_name,
                            document_id=email.id,
                            expected=is_responsive,
                            response=response,
                            latency=latency,
                            correct=correct,
                            metrics=metrics,
                        )
                    )

                except Exception as e:
                    results.append(
                        TaskResult(
                            task_name=task.name,
                            model_name=model.model_name,
                            document_id=email.id,
                            expected=is_responsive,
                            response="",
                            latency=0,
                            correct=False,
                            error=str(e),
                        )
                    )

        return results

    def run_full_evaluation(
        self,
        models: list[str] | None = None,
        n_per_category: int = 5,
        tasks: list[str] | None = None,
        verbose: bool = True,
    ) -> EvaluationResult:
        """Run full evaluation across all models and tasks.

        Args:
            models: List of model names (None = all text models)
            n_per_category: Sample size per challenge category
            tasks: List of tasks to run (None = all)
            verbose: Print progress

        Returns:
            Complete EvaluationResult
        """
        # Select models
        if models is None:
            available = get_text_models()
            models = [m.name for m in available]

        if verbose:
            print(f"Evaluating {len(models)} models...")

        # Select sample documents
        sample_docs = select_sample_documents(self.corpus, n_per_category)
        if verbose:
            print(f"Selected {len(sample_docs)} sample documents")
            responsive = sum(1 for _, r, _ in sample_docs if r)
            print(f"  Responsive: {responsive}, Non-responsive: {len(sample_docs) - responsive}")

        # Evaluate each model
        model_results = []
        for model_name in models:
            try:
                result = self.evaluate_model(model_name, sample_docs, tasks, verbose)
                model_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Error evaluating {model_name}: {e}")

        # Build result
        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            corpus_path=self.corpus_path,
            models_evaluated=models,
            sample_size=len(sample_docs),
            model_results=model_results,
        )

        # Generate recommendations
        result.recommendations = self._generate_recommendations(model_results)

        return result

    def _generate_recommendations(
        self, model_results: list[ModelEvaluation]
    ) -> dict[str, list[str]]:
        """Generate recommendations based on evaluation results."""
        recommendations: dict[str, list[str]] = {
            "classification": [],
            "generation": [],
            "extraction": [],
            "fast": [],
        }

        # Score models by task category
        classification_scores: list[tuple[str, float, float]] = []
        generation_scores: list[tuple[str, float, float]] = []
        extraction_scores: list[tuple[str, float, float]] = []

        for model in model_results:
            summary = model.summary.get("by_task", {})

            # Classification score (accuracy)
            class_tasks = ["Binary Classification", "Ternary Classification with Confidence", "JSON Structured Output"]
            class_accs = []
            class_latencies = []
            for task_name in class_tasks:
                if task_name in summary:
                    if "accuracy" in summary[task_name]:
                        class_accs.append(summary[task_name]["accuracy"])
                    class_latencies.append(summary[task_name].get("avg_latency", 0))

            if class_accs:
                avg_acc = sum(class_accs) / len(class_accs)
                avg_lat = sum(class_latencies) / len(class_latencies) if class_latencies else 0
                classification_scores.append((model.model_name, avg_acc, avg_lat))

            # Generation score (structure completeness)
            gen_tasks = ["Positive Example Generation", "Negative Example Generation", "Paraphrase Generation"]
            gen_scores_list = []
            gen_latencies = []
            for task_name in gen_tasks:
                if task_name in summary:
                    if "avg_structure_complete" in summary[task_name]:
                        gen_scores_list.append(summary[task_name]["avg_structure_complete"])
                    elif "avg_meets_target" in summary[task_name]:
                        gen_scores_list.append(summary[task_name]["avg_meets_target"])
                    gen_latencies.append(summary[task_name].get("avg_latency", 0))

            if gen_scores_list:
                avg_score = sum(gen_scores_list) / len(gen_scores_list)
                avg_lat = sum(gen_latencies) / len(gen_latencies) if gen_latencies else 0
                generation_scores.append((model.model_name, avg_score, avg_lat))

            # Extraction score
            ext_tasks = ["Evidence Extraction", "Keyword Extraction"]
            ext_scores_list = []
            ext_latencies = []
            for task_name in ext_tasks:
                if task_name in summary:
                    if "avg_extraction_accuracy" in summary[task_name]:
                        ext_scores_list.append(summary[task_name]["avg_extraction_accuracy"])
                    elif "avg_format_complete" in summary[task_name]:
                        ext_scores_list.append(summary[task_name]["avg_format_complete"])
                    ext_latencies.append(summary[task_name].get("avg_latency", 0))

            if ext_scores_list:
                avg_score = sum(ext_scores_list) / len(ext_scores_list)
                avg_lat = sum(ext_latencies) / len(ext_latencies) if ext_latencies else 0
                extraction_scores.append((model.model_name, avg_score, avg_lat))

        # Sort and recommend top 3
        classification_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations["classification"] = [m[0] for m in classification_scores[:3]]

        generation_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations["generation"] = [m[0] for m in generation_scores[:3]]

        extraction_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations["extraction"] = [m[0] for m in extraction_scores[:3]]

        # Fast models (lowest latency with acceptable accuracy)
        all_scores = classification_scores + generation_scores + extraction_scores
        if all_scores:
            # Filter for reasonable performance, then sort by latency
            acceptable = [s for s in all_scores if s[1] >= 0.5]
            if acceptable:
                acceptable.sort(key=lambda x: x[2])
                recommendations["fast"] = list(dict.fromkeys([m[0] for m in acceptable[:3]]))

        return recommendations

    def save_results(self, result: EvaluationResult, filename: str = "llm_eval_results.json") -> Path:
        """Save evaluation results to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        return output_path
