"""Contrastive scoring pipeline using positive/negative prototypes."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

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

    def to_dict(self) -> dict:
        return asdict(self)


def save_prototypes(
    positives: list[Prototype],
    negatives: list[Prototype],
    output_path: str | Path,
    model_name: str | None = None,
) -> None:
    """Save prototypes to a JSON file for inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model": model_name,
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "positives": [p.to_dict() for p in positives],
        "negatives": [p.to_dict() for p in negatives],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def print_prototypes(positives: list[Prototype], negatives: list[Prototype]) -> None:
    """Print prototypes to console for inspection."""
    print(f"\n{'='*80}")
    print(f"POSITIVE PROTOTYPES ({len(positives)})")
    print("=" * 80)
    for i, p in enumerate(positives, 1):
        print(f"\n--- Positive #{i} ---")
        print(p.text[:600] + "..." if len(p.text) > 600 else p.text)

    print(f"\n{'='*80}")
    print(f"NEGATIVE PROTOTYPES ({len(negatives)})")
    print("=" * 80)
    for i, p in enumerate(negatives, 1):
        print(f"\n--- Negative #{i} ---")
        print(p.text[:600] + "..." if len(p.text) > 600 else p.text)


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
        timeout: int = 300,
    ):
        self.model = OllamaModel(model_name, timeout=timeout)
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
        prompt = f"""Write {self.num_positive} realistic government work emails that ARE responsive to this public records request.

IMPORTANT: Vary how the topic is discussed:
- Some should use technical jargon (ppb, action levels, LSL, CCT)
- Some should reference the topic indirectly without obvious keywords
- Some should discuss historical incidents or future planning
- Some should bury the relevant content in routine updates

Do NOT include any preamble or explanation. Start directly with the first email.
Separate each email with exactly "---" on its own line.

REQUEST: {request.request_text}

EMAILS:"""

        response, _ = self.model.generate(prompt, max_tokens=2048)
        emails = response.split("---")

        return [
            Prototype(text=email.strip(), label="positive", source="llm_generated")
            for email in emails
            if email.strip() and not email.strip().startswith("Here")
        ][: self.num_positive]

    def _generate_negative(self, request: CPRARequest) -> list[Prototype]:
        """Generate negative (false positive) example emails."""
        prompt = f"""Write {self.num_negative} realistic government work emails that are NOT responsive to this public records request.

These emails should be FALSE POSITIVES - they would match keyword searches but are NOT actually about the topic.

CRITICAL: Focus on POLYSEMY (same word, different meaning):
- "lead" meaning LEADERSHIP (project lead, team lead, leading the effort)
- "lead" meaning TO GUIDE (lead the meeting, lead the discussion)
- "lead" meaning FIRST/PRIMARY (lead contractor, lead agency)

Also include:
- Water/infrastructure emails NOT about contamination (pipe repairs, billing, general maintenance)
- Environmental emails about OTHER contaminants (not lead)

Do NOT include any preamble. Start directly with the first email.
Do NOT write emails that are actually about lead contamination - those would be responsive!
Separate each email with exactly "---" on its own line.

REQUEST: {request.request_text}

EMAILS:"""

        response, _ = self.model.generate(prompt, max_tokens=2048)
        emails = response.split("---")

        return [
            Prototype(text=email.strip(), label="negative", source="llm_generated")
            for email in emails
            if email.strip() and not email.strip().startswith("Here")
        ][: self.num_negative]


class CorpusPrototypeGenerator:
    """Generate prototypes from actual corpus emails (ceiling test)."""

    def __init__(
        self,
        num_positive: int = 5,
        num_negative: int = 5,
        positive_categories: list[str] | None = None,
        negative_categories: list[str] | None = None,
        seed: int = 42,
    ):
        self.num_positive = num_positive
        self.num_negative = num_negative
        # Default: sample from diverse positive categories
        self.positive_categories = positive_categories or [
            "DIRECT_MATCH",
            "TECHNICAL_JARGON",
            "INDIRECT_REFERENCE",
            "AMBIGUOUS_TERMS",
            "TEMPORAL_REFERENCE",
        ]
        # Default: sample from false positive category
        self.negative_categories = negative_categories or [
            "KEYWORD_FALSE_POSITIVE",
        ]
        self.seed = seed

    def generate(
        self, corpus: "Corpus"  # noqa: F821
    ) -> tuple[list[Prototype], list[Prototype]]:
        """Generate prototypes by sampling from corpus."""
        import random
        from src.data import ChallengeType

        random.seed(self.seed)

        # Sample positive prototypes from responsive categories
        positive_emails = []
        for cat_name in self.positive_categories:
            cat = ChallengeType(cat_name)
            emails = corpus.get_emails_by_challenge(cat)
            positive_emails.extend(emails)

        # Shuffle and sample
        random.shuffle(positive_emails)
        sampled_positive = positive_emails[: self.num_positive]

        positives = [
            Prototype(
                text=email.text,
                label="positive",
                source=f"corpus:{corpus.get_challenge_type(email.id).value}",
            )
            for email in sampled_positive
        ]

        # Sample negative prototypes from false positive categories
        negative_emails = []
        for cat_name in self.negative_categories:
            cat = ChallengeType(cat_name)
            emails = corpus.get_emails_by_challenge(cat)
            negative_emails.extend(emails)

        random.shuffle(negative_emails)
        sampled_negative = negative_emails[: self.num_negative]

        negatives = [
            Prototype(
                text=email.text,
                label="negative",
                source=f"corpus:{corpus.get_challenge_type(email.id).value}",
            )
            for email in sampled_negative
        ]

        return positives, negatives
