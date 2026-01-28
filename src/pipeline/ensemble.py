"""Ensemble pipeline combining multiple search pipelines."""

from dataclasses import dataclass

import numpy as np

from src.data import CPRARequest, SearchableDocument

from .base import SearchPipeline, SearchResult


@dataclass
class PipelineWeight:
    """A pipeline with optional weight for ensemble."""

    pipeline: SearchPipeline
    weight: float = 1.0


class EnsemblePipeline(SearchPipeline):
    """Combine multiple pipelines using score aggregation or RRF."""

    def __init__(
        self,
        pipelines: list[SearchPipeline | PipelineWeight],
        method: str = "rrf",  # "rrf", "mean", "weighted_mean", "max"
        rrf_k: int = 60,
        normalize_scores: bool = True,
    ):
        self.pipelines = []
        self.weights = []

        for p in pipelines:
            if isinstance(p, PipelineWeight):
                self.pipelines.append(p.pipeline)
                self.weights.append(p.weight)
            else:
                self.pipelines.append(p)
                self.weights.append(1.0)

        self.method = method
        self.rrf_k = rrf_k
        self.normalize_scores = normalize_scores

    @property
    def name(self) -> str:
        names = [p.name for p in self.pipelines]
        return f"Ensemble[{self.method}]({', '.join(names)})"

    def search(
        self,
        request: CPRARequest,
        documents: list[SearchableDocument],
    ) -> list[SearchResult]:
        # 1. Run each pipeline
        all_results = []
        for pipeline in self.pipelines:
            results = pipeline.search(request, documents)
            all_results.append({r.doc_id: r.score for r in results})

        # 2. Get all document IDs
        all_doc_ids = set()
        for results in all_results:
            all_doc_ids.update(results.keys())
        doc_ids = sorted(all_doc_ids)

        # 3. Build score matrix
        scores = np.zeros((len(self.pipelines), len(doc_ids)))
        for i, results in enumerate(all_results):
            for j, did in enumerate(doc_ids):
                scores[i, j] = results.get(did, 0.0)

        # 4. Normalize if requested
        if self.normalize_scores and self.method != "rrf":
            # Min-max normalize each pipeline's scores
            for i in range(len(self.pipelines)):
                min_s, max_s = scores[i].min(), scores[i].max()
                if max_s > min_s:
                    scores[i] = (scores[i] - min_s) / (max_s - min_s)

        # 5. Aggregate
        if self.method == "rrf":
            final_scores = self._rrf_aggregate(scores)
        elif self.method == "mean":
            final_scores = np.mean(scores, axis=0)
        elif self.method == "weighted_mean":
            weights = np.array(self.weights)
            final_scores = np.average(scores, axis=0, weights=weights)
        elif self.method == "max":
            final_scores = np.max(scores, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 6. Build results
        results = [
            SearchResult(doc_id=did, score=float(score))
            for did, score in zip(doc_ids, final_scores)
        ]
        results.sort(reverse=True)

        return results

    def _rrf_aggregate(self, scores: np.ndarray) -> np.ndarray:
        """Reciprocal Rank Fusion across pipelines."""
        num_docs = scores.shape[1]
        rrf_scores = np.zeros(num_docs)

        for i, p_scores in enumerate(scores):
            ranks = np.argsort(np.argsort(-p_scores))
            rrf_scores += self.weights[i] / (self.rrf_k + ranks + 1)

        return rrf_scores
