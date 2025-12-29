"""Cross-encoder search pipeline for direct relevance scoring."""

import numpy as np
from sentence_transformers import CrossEncoder

from src.data import CPRARequest, SearchableDocument

from .base import SearchPipeline, SearchResult


# Available cross-encoder models
CROSS_ENCODER_MODELS = {
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-large": "BAAI/bge-reranker-large",
}

# NLI models output 3 classes: [contradiction, neutral, entailment]
# We use entailment score (index 2) for relevance
NLI_MODELS = {
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-large",
    "cross-encoder/nli-MiniLM2-L6-H768",
}


class CrossEncoderSearchPipeline(SearchPipeline):
    """Search pipeline using cross-encoder for direct query-document scoring.

    Unlike bi-encoders that encode query and documents separately,
    cross-encoders process query and document together, allowing
    direct attention between them for more accurate relevance scoring.

    Trade-off: More accurate but slower (can't pre-compute document encodings).
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder search pipeline.

        Args:
            model_name: Cross-encoder model name (short name or full HF path)
        """
        self.model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return f"Cross-Encoder ({self.model_name})"

    @property
    def model(self) -> CrossEncoder:
        """Lazy load cross-encoder model."""
        if self._model is None:
            # Resolve short name to full model path
            model_path = CROSS_ENCODER_MODELS.get(self.model_name, self.model_name)
            self._model = CrossEncoder(model_path)
        return self._model

    def _get_query_text(self, request: CPRARequest) -> str:
        """Extract query text from CPRA request."""
        return request.search_text

    def _is_nli_model(self) -> bool:
        """Check if current model is an NLI model with multi-class output."""
        model_path = CROSS_ENCODER_MODELS.get(self.model_name, self.model_name)
        return model_path in NLI_MODELS

    def search(
        self, request: CPRARequest, documents: list[SearchableDocument]
    ) -> list[SearchResult]:
        """Score all documents against the query using cross-encoder.

        Args:
            request: CPRA request to search for
            documents: Searchable documents (emails or threads)

        Returns:
            List of SearchResult sorted by relevance score (highest first)
        """
        query = self._get_query_text(request)

        # Create query-document pairs for cross-encoder
        pairs = [(query, doc.text) for doc in documents]

        # Score all pairs (this is the slow part - no pre-computation possible)
        raw_scores = self.model.predict(pairs, show_progress_bar=True)

        # Handle NLI models which output [contradiction, neutral, entailment] logits
        if self._is_nli_model():
            # Use entailment score (index 2) as relevance indicator
            scores = np.array(raw_scores)[:, 2]
        else:
            scores = raw_scores

        # Build results
        results = [
            SearchResult(doc_id=doc.id, score=float(score))
            for doc, score in zip(documents, scores)
        ]

        # Sort by score (highest first)
        results.sort(reverse=True)

        return results
