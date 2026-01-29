"""Contrastive scoring pipeline using positive/negative prototypes."""

from dataclasses import dataclass

import numpy as np

from src.data import CPRARequest, SearchableDocument
from src.llm_eval.models import OllamaModel
from src.models import cosine_similarity

from .base import SearchPipeline, SearchResult
from .embedding import EmbeddingSearchPipeline


@dataclass
class Prototype:
    """A prototype document (positive or negative)."""

    text: str
    label: str  # "positive" or "negative"
    source: str  # "llm_generated", "corpus", "centroid"


class ContrastivePipeline(SearchPipeline):
    """Score documents by similarity to positive/negative prototypes."""

    def __init__(
        self,
        model_name: str = "st:all-mpnet-base-v2",
        positive_prototypes: list[Prototype] | None = None,
        negative_prototypes: list[Prototype] | None = None,
        lambda_negative: float = 0.5,  # Weight for negative similarity penalty
        scoring_method: str = "max",  # "max", "mean"
        cache_dir: str | None = None,
    ):
        self.embedding_pipeline = EmbeddingSearchPipeline(
            model_name=model_name,
            cache_dir=cache_dir,
        )
        self.positive_prototypes = positive_prototypes or []
        self.negative_prototypes = negative_prototypes or []
        self.lambda_negative = lambda_negative
        self.scoring_method = scoring_method

        # Cache for prototype embeddings
        self._pos_embeddings: np.ndarray | None = None
        self._neg_embeddings: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"Contrastive ({len(self.positive_prototypes)}+/{len(self.negative_prototypes)}-)"

    def set_prototypes(
        self,
        positive: list[Prototype],
        negative: list[Prototype],
    ) -> None:
        """Set prototypes and clear cached embeddings."""
        self.positive_prototypes = positive
        self.negative_prototypes = negative
        self._pos_embeddings = None
        self._neg_embeddings = None

    def _embed_prototypes(self) -> None:
        """Embed all prototypes."""
        model = self.embedding_pipeline.model

        if self.positive_prototypes and self._pos_embeddings is None:
            pos_texts = [p.text for p in self.positive_prototypes]
            self._pos_embeddings = model.embed(pos_texts)

        if self.negative_prototypes and self._neg_embeddings is None:
            neg_texts = [p.text for p in self.negative_prototypes]
            self._neg_embeddings = model.embed(neg_texts)

    def search(
        self,
        request: CPRARequest,
        documents: list[SearchableDocument],
    ) -> list[SearchResult]:
        # Ensure prototypes are embedded
        self._embed_prototypes()

        # Get document embeddings
        doc_embeddings = self.embedding_pipeline._embed_documents(documents)
        doc_ids = [d.id for d in documents]
        doc_matrix = np.array([doc_embeddings[did] for did in doc_ids])

        results = []

        for i, doc_id in enumerate(doc_ids):
            doc_emb = doc_matrix[i]

            # Compute positive similarity
            pos_score = 0.0
            if self._pos_embeddings is not None:
                pos_sims = cosine_similarity(doc_emb, self._pos_embeddings)
                if self.scoring_method == "max":
                    pos_score = float(np.max(pos_sims))
                else:
                    pos_score = float(np.mean(pos_sims))

            # Compute negative similarity
            neg_score = 0.0
            if self._neg_embeddings is not None:
                neg_sims = cosine_similarity(doc_emb, self._neg_embeddings)
                if self.scoring_method == "max":
                    neg_score = float(np.max(neg_sims))
                else:
                    neg_score = float(np.mean(neg_sims))

            # Contrastive score
            score = pos_score - self.lambda_negative * neg_score

            results.append(SearchResult(doc_id=doc_id, score=score))

        results.sort(reverse=True)
        return results


class LLMPrototypeGenerator:
    """Generate prototypes using LLM (EXP-025)."""

    def __init__(
        self,
        model_name: str = "ministral-3:3b",
        num_positive: int = 5,
        num_negative: int = 5,
    ):
        self.model = OllamaModel(model_name, timeout=180)
        self.num_positive = num_positive
        self.num_negative = num_negative

    def generate(
        self, request: CPRARequest
    ) -> tuple[list[Prototype], list[Prototype]]:
        """Generate positive and negative prototypes."""
        positives = self._generate_positive(request)
        negatives = self._generate_negative(request)
        return positives, negatives

    def _generate_positive(self, request: CPRARequest) -> list[Prototype]:
        """Generate positive example emails."""
        prompt = f"""Write {self.num_positive} realistic work emails that would be responsive to this public records request.
Do not use obvious keywords—show relevance through context.
Separate each email with "---".

REQUEST: {request.request_text}

EMAILS:"""

        response, _ = self.model.generate(prompt, max_tokens=2048)
        emails = response.split("---")

        return [
            Prototype(text=email.strip(), label="positive", source="llm_generated")
            for email in emails
            if email.strip()
        ][: self.num_positive]

    def _generate_negative(self, request: CPRARequest) -> list[Prototype]:
        """Generate negative (false positive) example emails."""
        prompt = f"""Write {self.num_negative} realistic work emails that are NOT responsive to this public records request,
but might be confused for responsive because they share vocabulary or general topic.

Target these patterns:
- Keywords used in unrelated contexts
- Same domain but different specific topic
- Administrative content tangentially related

Separate each email with "---".

REQUEST: {request.request_text}

EMAILS:"""

        response, _ = self.model.generate(prompt, max_tokens=2048)
        emails = response.split("---")

        return [
            Prototype(text=email.strip(), label="negative", source="llm_generated")
            for email in emails
            if email.strip()
        ][: self.num_negative]
