"""Evaluation orchestration for CPRA experiments."""

from dataclasses import dataclass, field
from typing import Any

from src.data import ChallengeType, Corpus
from src.pipeline import SearchPipeline, SearchResult

from .metrics import (
    compute_binary_metrics,
    compute_ranked_metrics,
    mean_average_precision,
)


@dataclass
class RequestMetrics:
    """Metrics for a single CPRA request."""

    request_id: str
    request_title: str
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_responsive: int
    ranked_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ChallengeMetrics:
    """Metrics broken down by challenge type."""

    challenge_type: ChallengeType
    precision: float
    recall: float
    f1: float
    total_emails: int
    correctly_identified: int


@dataclass
class ThresholdMetrics:
    """Metrics at a specific threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    total_predicted: int
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    experiment_name: str
    pipeline_name: str

    # Overall metrics
    overall_precision: float
    overall_recall: float
    overall_f1: float
    mean_average_precision: float

    # Per-request breakdown
    by_request: list[RequestMetrics]

    # Per-challenge-type breakdown
    by_challenge: list[ChallengeMetrics]

    # Raw numbers
    total_emails: int
    total_responsive: int
    total_predicted: int

    # Threshold analysis (optional)
    threshold_analysis: list[ThresholdMetrics] = field(default_factory=list)

    # Metadata
    k_values: list[int] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """Evaluates search pipeline results against ground truth."""

    def __init__(
        self,
        corpus: Corpus,
        k_values: list[int] | None = None,
    ):
        """Initialize evaluator.

        Args:
            corpus: Corpus with ground truth
            k_values: K values for precision@k, recall@k (default: [50, 100, 200, 375])
        """
        self.corpus = corpus
        self.k_values = k_values or [50, 100, 200, 375]

    def evaluate(
        self,
        pipeline: SearchPipeline,
        experiment_name: str = "",
        threshold: float = 0.5,
    ) -> EvaluationResult:
        """Run full evaluation of a pipeline.

        Args:
            pipeline: Search pipeline to evaluate
            experiment_name: Name for this experiment
            threshold: Score threshold for binary classification

        Returns:
            Complete evaluation results
        """
        # Run pipeline on all requests
        all_results = pipeline.search_all(self.corpus.requests, self.corpus.emails)

        # Compute per-request metrics
        request_metrics = []
        all_predictions: dict[str, set[str]] = {}
        all_rankings: dict[str, list[str]] = {}
        all_actuals: dict[str, set[str]] = {}

        for request in self.corpus.requests:
            results = all_results[request.id]
            predictions = pipeline.get_predictions(results, threshold)
            ranked_ids = pipeline.get_ranked_ids(results)
            actual = self.corpus.get_responsive_emails(request.id)

            all_predictions[request.id] = predictions
            all_rankings[request.id] = ranked_ids
            all_actuals[request.id] = actual

            # Binary metrics
            binary = compute_binary_metrics(predictions, actual)

            # Ranked metrics
            ranked = compute_ranked_metrics(ranked_ids, actual, self.k_values)

            request_metrics.append(
                RequestMetrics(
                    request_id=request.id,
                    request_title=request.title,
                    precision=binary["precision"],
                    recall=binary["recall"],
                    f1=binary["f1"],
                    true_positives=binary["true_positives"],
                    false_positives=binary["false_positives"],
                    false_negatives=binary["false_negatives"],
                    total_responsive=len(actual),
                    ranked_metrics=ranked,
                )
            )

        # Compute challenge type breakdown
        challenge_metrics = self._compute_challenge_breakdown(
            all_predictions, all_actuals
        )

        # Compute overall metrics (aggregate across requests)
        all_pred = set().union(*all_predictions.values()) if all_predictions else set()
        all_actual = set().union(*all_actuals.values()) if all_actuals else set()
        overall = compute_binary_metrics(all_pred, all_actual)
        map_score = mean_average_precision(all_rankings, all_actuals)

        return EvaluationResult(
            experiment_name=experiment_name,
            pipeline_name=pipeline.name,
            overall_precision=overall["precision"],
            overall_recall=overall["recall"],
            overall_f1=overall["f1"],
            mean_average_precision=map_score,
            by_request=request_metrics,
            by_challenge=challenge_metrics,
            total_emails=self.corpus.num_emails,
            total_responsive=len(all_actual),
            total_predicted=len(all_pred),
            k_values=self.k_values,
        )

    def evaluate_thresholds(
        self,
        pipeline: SearchPipeline,
        thresholds: list[float],
    ) -> list[ThresholdMetrics]:
        """Evaluate pipeline at multiple thresholds.

        Args:
            pipeline: Search pipeline to evaluate
            thresholds: List of thresholds to evaluate

        Returns:
            List of ThresholdMetrics for each threshold
        """
        # Run pipeline once to get all scores
        all_results = pipeline.search_all(self.corpus.requests, self.corpus.emails)

        # Get all actual responsive emails
        all_actuals: dict[str, set[str]] = {}
        for request in self.corpus.requests:
            all_actuals[request.id] = self.corpus.get_responsive_emails(request.id)

        all_actual = set().union(*all_actuals.values()) if all_actuals else set()

        threshold_metrics = []
        for threshold in sorted(thresholds):
            # Get predictions at this threshold
            all_predictions: dict[str, set[str]] = {}
            for request in self.corpus.requests:
                results = all_results[request.id]
                predictions = pipeline.get_predictions(results, threshold)
                all_predictions[request.id] = predictions

            all_pred = set().union(*all_predictions.values()) if all_predictions else set()

            # Compute metrics
            tp = len(all_pred & all_actual)
            fp = len(all_pred - all_actual)
            fn = len(all_actual - all_pred)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            threshold_metrics.append(
                ThresholdMetrics(
                    threshold=threshold,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    total_predicted=len(all_pred),
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                )
            )

        return threshold_metrics

    def _compute_challenge_breakdown(
        self,
        predictions: dict[str, set[str]],
        actuals: dict[str, set[str]],
    ) -> list[ChallengeMetrics]:
        """Compute metrics broken down by challenge type.

        For each challenge type, we look at responsive emails that have that
        challenge pattern and compute how many were correctly identified.
        """
        results = []

        for challenge_type in ChallengeType:
            # Get all emails with this challenge type
            challenge_emails = self.corpus.get_emails_by_challenge(challenge_type)
            challenge_email_ids = {e.id for e in challenge_emails}

            if not challenge_email_ids:
                continue

            # For each request, get responsive emails with this challenge
            total_responsive = 0
            correctly_identified = 0
            false_positives_for_challenge = 0

            for request_id, actual in actuals.items():
                # Responsive emails with this challenge type
                responsive_with_challenge = actual & challenge_email_ids
                total_responsive += len(responsive_with_challenge)

                # How many did we correctly predict?
                predicted = predictions.get(request_id, set())
                correctly_identified += len(predicted & responsive_with_challenge)

                # False positives: predicted but not responsive, has this challenge
                # (these are the "near miss" type errors we want to catch)
                false_pos = predicted - actual
                false_positives_for_challenge += len(false_pos & challenge_email_ids)

            # Compute metrics for this challenge type
            if total_responsive > 0:
                recall = correctly_identified / total_responsive
            else:
                recall = 0.0

            predicted_with_challenge = correctly_identified + false_positives_for_challenge
            if predicted_with_challenge > 0:
                precision = correctly_identified / predicted_with_challenge
            else:
                precision = 0.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            results.append(
                ChallengeMetrics(
                    challenge_type=challenge_type,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    total_emails=len(challenge_email_ids),
                    correctly_identified=correctly_identified,
                )
            )

        return results
