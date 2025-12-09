"""Embedding-based search pipeline."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data import CPRARequest, Email
from src.models import cosine_similarity, get_embedding_model

from .base import SearchPipeline, SearchResult


class EmbeddingSearchPipeline(SearchPipeline):
    """Search pipeline using embedding similarity."""

    def __init__(
        self,
        model_name: str = "st:all-mpnet-base-v2",
        embed_fields: list[str] | None = None,
        cache_dir: str | Path | None = None,
    ):
        """Initialize embedding search pipeline.

        Args:
            model_name: Embedding model key from config
            embed_fields: Which email fields to embed (default: ['subject', 'body'])
            cache_dir: Directory to cache embeddings (default: .cache/embeddings)
        """
        self.model_name = model_name
        self.embed_fields = embed_fields or ["subject", "body"]
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._email_embeddings: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return f"Embedding Search ({self.model_name})"

    @property
    def model(self):
        """Lazy load embedding model."""
        if self._model is None:
            self._model = get_embedding_model(self.model_name)
        return self._model

    def _get_email_text(self, email: Email) -> str:
        """Extract text from email based on embed_fields."""
        parts = []
        for field in self.embed_fields:
            if field == "subject":
                parts.append(email.subject)
            elif field == "body":
                parts.append(email.body)
        return "\n\n".join(parts)

    def _get_request_text(self, request: CPRARequest) -> str:
        """Extract text from request for embedding."""
        return request.search_text

    def _get_cache_key(self, emails: list[Email]) -> str:
        """Generate cache key for a set of emails."""
        # Use model name and email IDs to generate cache key
        content = f"{self.model_name}:{':'.join(sorted(e.id for e in emails))}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _load_cached_embeddings(
        self, cache_key: str
    ) -> dict[str, np.ndarray] | None:
        """Load embeddings from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.npz"
        meta_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists() and meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

            data = np.load(cache_file)

            embeddings = {}
            for email_id in meta["email_ids"]:
                if email_id in data:
                    embeddings[email_id] = data[email_id]

            return embeddings

        return None

    def _save_embeddings_cache(
        self, cache_key: str, embeddings: dict[str, np.ndarray]
    ) -> None:
        """Save embeddings to cache."""
        cache_file = self.cache_dir / f"{cache_key}.npz"
        meta_file = self.cache_dir / f"{cache_key}.json"

        # Save embeddings as npz
        np.savez(cache_file, **embeddings)

        # Save metadata
        meta = {
            "model_name": self.model_name,
            "embed_fields": self.embed_fields,
            "email_ids": list(embeddings.keys()),
        }
        with open(meta_file, "w") as f:
            json.dump(meta, f)

    def _embed_emails(self, emails: list[Email]) -> dict[str, np.ndarray]:
        """Embed all emails, using cache if available."""
        cache_key = self._get_cache_key(emails)

        # Try to load from cache
        cached = self._load_cached_embeddings(cache_key)
        if cached is not None:
            return cached

        # Embed all emails
        texts = [self._get_email_text(e) for e in emails]
        embeddings_array = self.model.embed(texts)

        embeddings = {
            email.id: embeddings_array[i] for i, email in enumerate(emails)
        }

        # Cache for future use
        self._save_embeddings_cache(cache_key, embeddings)

        return embeddings

    def search(
        self, request: CPRARequest, emails: list[Email]
    ) -> list[SearchResult]:
        """Search for emails similar to CPRA request.

        Args:
            request: CPRA request to search for
            emails: Emails to search through

        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        # Get email embeddings
        if not self._email_embeddings:
            self._email_embeddings = self._embed_emails(emails)

        # Embed the request
        request_text = self._get_request_text(request)
        request_embedding = self.model.embed_single(request_text)

        # Build email embedding matrix
        email_ids = [e.id for e in emails]
        email_embeddings = np.array([self._email_embeddings[eid] for eid in email_ids])

        # Compute similarities
        similarities = cosine_similarity(request_embedding, email_embeddings)

        # Build results
        results = [
            SearchResult(email_id=eid, score=float(sim))
            for eid, sim in zip(email_ids, similarities)
        ]

        # Sort by score (highest first)
        results.sort(reverse=True)

        return results
