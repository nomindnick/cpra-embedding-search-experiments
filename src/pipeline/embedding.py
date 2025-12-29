"""Embedding-based search pipeline."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data import CPRARequest, SearchableDocument
from src.models import cosine_similarity, get_embedding_model

from .base import SearchPipeline, SearchResult


class EmbeddingSearchPipeline(SearchPipeline):
    """Search pipeline using embedding similarity."""

    def __init__(
        self,
        model_name: str = "st:all-mpnet-base-v2",
        cache_dir: str | Path | None = None,
    ):
        """Initialize embedding search pipeline.

        Args:
            model_name: Embedding model key from config
            cache_dir: Directory to cache embeddings (default: .cache/embeddings)
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._doc_embeddings: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return f"Embedding Search ({self.model_name})"

    @property
    def model(self):
        """Lazy load embedding model."""
        if self._model is None:
            self._model = get_embedding_model(self.model_name)
        return self._model

    def _get_request_text(self, request: CPRARequest) -> str:
        """Extract text from request for embedding."""
        return request.search_text

    def _get_cache_key(self, documents: list[SearchableDocument]) -> str:
        """Generate cache key for a set of documents."""
        # Use model name and document IDs to generate cache key
        content = f"{self.model_name}:{':'.join(sorted(d.id for d in documents))}"
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
            for doc_id in meta["doc_ids"]:
                if doc_id in data:
                    embeddings[doc_id] = data[doc_id]

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
            "doc_ids": list(embeddings.keys()),
        }
        with open(meta_file, "w") as f:
            json.dump(meta, f)

    def _embed_documents(
        self, documents: list[SearchableDocument]
    ) -> dict[str, np.ndarray]:
        """Embed all documents, using cache if available."""
        cache_key = self._get_cache_key(documents)

        # Try to load from cache
        cached = self._load_cached_embeddings(cache_key)
        if cached is not None:
            return cached

        # Embed all documents using their .text property
        texts = [doc.text for doc in documents]
        embeddings_array = self.model.embed(texts)

        embeddings = {doc.id: embeddings_array[i] for i, doc in enumerate(documents)}

        # Cache for future use
        self._save_embeddings_cache(cache_key, embeddings)

        return embeddings

    def search(
        self, request: CPRARequest, documents: list[SearchableDocument]
    ) -> list[SearchResult]:
        """Search for documents similar to CPRA request.

        Args:
            request: CPRA request to search for
            documents: Searchable documents (emails or threads)

        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        # Get document embeddings
        if not self._doc_embeddings:
            self._doc_embeddings = self._embed_documents(documents)

        # Embed the request
        request_text = self._get_request_text(request)
        request_embedding = self.model.embed_single(request_text)

        # Build document embedding matrix
        doc_ids = [d.id for d in documents]
        doc_embeddings = np.array([self._doc_embeddings[did] for did in doc_ids])

        # Compute similarities
        similarities = cosine_similarity(request_embedding, doc_embeddings)

        # Build results
        results = [
            SearchResult(doc_id=did, score=float(sim))
            for did, sim in zip(doc_ids, similarities)
        ]

        # Sort by score (highest first)
        results.sort(reverse=True)

        return results
