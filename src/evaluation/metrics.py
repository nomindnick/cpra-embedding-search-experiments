"""Evaluation metrics for CPRA responsiveness detection."""


def precision(predicted: set[str], actual: set[str]) -> float:
    """Calculate precision: TP / (TP + FP).

    Args:
        predicted: Set of predicted positive IDs
        actual: Set of actual positive IDs

    Returns:
        Precision score (0.0 to 1.0)
    """
    if not predicted:
        return 0.0
    true_positives = len(predicted & actual)
    return true_positives / len(predicted)


def recall(predicted: set[str], actual: set[str]) -> float:
    """Calculate recall: TP / (TP + FN).

    Args:
        predicted: Set of predicted positive IDs
        actual: Set of actual positive IDs

    Returns:
        Recall score (0.0 to 1.0)
    """
    if not actual:
        return 0.0
    true_positives = len(predicted & actual)
    return true_positives / len(actual)


def f1_score(prec: float, rec: float) -> float:
    """Calculate F1 score: 2 * (precision * recall) / (precision + recall).

    Args:
        prec: Precision score
        rec: Recall score

    Returns:
        F1 score (0.0 to 1.0)
    """
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def precision_at_k(ranked_ids: list[str], actual: set[str], k: int) -> float:
    """Calculate precision at K.

    Args:
        ranked_ids: List of IDs ranked by score (highest first)
        actual: Set of actual positive IDs
        k: Number of top results to consider

    Returns:
        Precision at K
    """
    if k <= 0 or not ranked_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return precision(top_k, actual)


def recall_at_k(ranked_ids: list[str], actual: set[str], k: int) -> float:
    """Calculate recall at K.

    Args:
        ranked_ids: List of IDs ranked by score (highest first)
        actual: Set of actual positive IDs
        k: Number of top results to consider

    Returns:
        Recall at K
    """
    if k <= 0 or not ranked_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return recall(top_k, actual)


def average_precision(ranked_ids: list[str], actual: set[str]) -> float:
    """Calculate average precision (AP).

    AP is the area under the precision-recall curve, computed as the
    average of precisions at each relevant document.

    Args:
        ranked_ids: List of IDs ranked by score (highest first)
        actual: Set of actual positive IDs

    Returns:
        Average precision score
    """
    if not actual or not ranked_ids:
        return 0.0

    num_relevant = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(ranked_ids):
        if doc_id in actual:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            precision_sum += precision_at_i

    if num_relevant == 0:
        return 0.0

    return precision_sum / len(actual)


def mean_average_precision(
    rankings: dict[str, list[str]], actuals: dict[str, set[str]]
) -> float:
    """Calculate mean average precision (MAP) across multiple queries.

    Args:
        rankings: Dict mapping query_id -> ranked list of document IDs
        actuals: Dict mapping query_id -> set of relevant document IDs

    Returns:
        Mean average precision across all queries
    """
    if not rankings:
        return 0.0

    ap_sum = 0.0
    for query_id, ranked_ids in rankings.items():
        actual = actuals.get(query_id, set())
        ap_sum += average_precision(ranked_ids, actual)

    return ap_sum / len(rankings)


def compute_binary_metrics(
    predicted: set[str], actual: set[str]
) -> dict[str, float]:
    """Compute all binary classification metrics.

    Args:
        predicted: Set of predicted positive IDs
        actual: Set of actual positive IDs

    Returns:
        Dict with precision, recall, f1 scores
    """
    prec = precision(predicted, actual)
    rec = recall(predicted, actual)
    f1 = f1_score(prec, rec)

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "true_positives": len(predicted & actual),
        "false_positives": len(predicted - actual),
        "false_negatives": len(actual - predicted),
    }


def compute_ranked_metrics(
    ranked_ids: list[str], actual: set[str], k_values: list[int]
) -> dict[str, float]:
    """Compute all ranking metrics at various K values.

    Args:
        ranked_ids: List of IDs ranked by score (highest first)
        actual: Set of actual positive IDs
        k_values: List of K values to compute metrics at

    Returns:
        Dict with precision@k, recall@k for each k, plus MAP
    """
    metrics = {
        "average_precision": average_precision(ranked_ids, actual),
    }

    for k in k_values:
        metrics[f"precision_at_{k}"] = precision_at_k(ranked_ids, actual, k)
        metrics[f"recall_at_{k}"] = recall_at_k(ranked_ids, actual, k)

    return metrics
