"""Evaluation orchestration for CPRA experiments."""

from dataclasses import dataclass, field
from typing import Any

from src.data import ChallengeType, Corpus
from src.pipeline import SearchPipeline, SearchResult

from .metrics import (
    average_precision,
    compute_binary_metrics,
    compute_ranked_metrics,
)


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
    precision: float
    recall: float
    f1: float
    average_precision: float

    # Per-challenge-type breakdown
    by_challenge: list[ChallengeMetrics]

    # Raw numbers
    total_emails: int
    total_documents: int
    total_responsive: int
    total_predicted: int
    true_positives: int
    false_positives: int
    false_negatives: int

    # Ranked metrics
    ranked_metrics: dict[str, float] = field(default_factory=dict)

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
            k_values: K values for precision@k, recall@k (default: [50, 100, 200])
        """
        self.corpus = corpus
        self.k_values = k_values or [50, 100, 200]

    def _doc_predictions_to_email_ids(self, doc_ids: set[str]) -> set[str]:
        """Map document IDs (which may include thread IDs) to email IDs."""
        email_ids: set[str] = set()
        for doc_id in doc_ids:
            email_ids.update(self.corpus.document_to_email_ids(doc_id))
        return email_ids

    def _doc_rankings_to_email_ids(self, doc_ids: list[str]) -> list[str]:
        """Map ranked document IDs to email IDs (preserving order)."""
        email_ids: list[str] = []
        for doc_id in doc_ids:
            email_ids.extend(self.corpus.document_to_email_ids(doc_id))
        return email_ids

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
        # Get searchable documents and run pipeline
        documents = self.corpus.get_searchable_documents()
        results = pipeline.search(self.corpus.request, documents)

        # Get document-level predictions and rankings
        doc_predictions = pipeline.get_predictions(results, threshold)
        doc_rankings = pipeline.get_ranked_ids(results)

        # Map document predictions back to email IDs
        predicted_email_ids = self._doc_predictions_to_email_ids(doc_predictions)
        ranked_email_ids = self._doc_rankings_to_email_ids(doc_rankings)

        # Get actual responsive emails
        actual_email_ids = self.corpus.get_responsive_emails()

        # Compute binary metrics at email level
        binary = compute_binary_metrics(predicted_email_ids, actual_email_ids)

        # Compute ranked metrics at email level
        ranked = compute_ranked_metrics(ranked_email_ids, actual_email_ids, self.k_values)

        # Compute average precision
        ap = average_precision(ranked_email_ids, actual_email_ids)

        # Compute challenge type breakdown
        challenge_metrics = self._compute_challenge_breakdown(
            predicted_email_ids, actual_email_ids
        )

        return EvaluationResult(
            experiment_name=experiment_name,
            pipeline_name=pipeline.name,
            precision=binary["precision"],
            recall=binary["recall"],
            f1=binary["f1"],
            average_precision=ap,
            by_challenge=challenge_metrics,
            total_emails=self.corpus.num_emails,
            total_documents=self.corpus.num_searchable_documents,
            total_responsive=len(actual_email_ids),
            total_predicted=len(predicted_email_ids),
            true_positives=binary["true_positives"],
            false_positives=binary["false_positives"],
            false_negatives=binary["false_negatives"],
            ranked_metrics=ranked,
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
        documents = self.corpus.get_searchable_documents()
        results = pipeline.search(self.corpus.request, documents)

        # Get actual responsive emails
        actual_email_ids = self.corpus.get_responsive_emails()

        threshold_metrics = []
        for threshold in sorted(thresholds):
            # Get predictions at this threshold
            doc_predictions = pipeline.get_predictions(results, threshold)
            predicted_email_ids = self._doc_predictions_to_email_ids(doc_predictions)

            # Compute metrics
            tp = len(predicted_email_ids & actual_email_ids)
            fp = len(predicted_email_ids - actual_email_ids)
            fn = len(actual_email_ids - predicted_email_ids)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            threshold_metrics.append(
                ThresholdMetrics(
                    threshold=threshold,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    total_predicted=len(predicted_email_ids),
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                )
            )

        return threshold_metrics

    def _compute_challenge_breakdown(
        self,
        predictions: set[str],
        actuals: set[str],
    ) -> list[ChallengeMetrics]:
        """Compute metrics broken down by challenge type.

        For each challenge type, we look at emails with that challenge type
        and compute how many were correctly identified.
        """
        results = []

        for challenge_type in ChallengeType:
            # Get all emails with this challenge type
            challenge_emails = self.corpus.get_emails_by_challenge(challenge_type)
            challenge_email_ids = {e.id for e in challenge_emails}

            if not challenge_email_ids:
                continue

            # Responsive emails with this challenge type
            responsive_with_challenge = actuals & challenge_email_ids

            # How many did we correctly predict?
            correctly_identified = len(predictions & responsive_with_challenge)

            # False positives: predicted but not responsive, has this challenge
            false_pos = predictions - actuals
            false_positives_for_challenge = len(false_pos & challenge_email_ids)

            # Compute metrics for this challenge type
            total_responsive = len(responsive_with_challenge)
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
