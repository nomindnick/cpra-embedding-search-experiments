"""Embedding model abstractions."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensionality."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            np.ndarray of shape (len(texts), dimensions)
        """
        pass

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text.

        Args:
            text: String to embed

        Returns:
            np.ndarray of shape (dimensions,)
        """
        return self.embed([text])[0]


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model using sentence-transformers library."""

    def __init__(self, model_name: str):
        """Initialize with a sentence-transformers model.

        Args:
            model_name: Model name (e.g., 'all-MiniLM-L6-v2', 'all-mpnet-base-v2')
        """
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"st:{self._model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using sentence-transformers.

        Args:
            texts: List of strings to embed

        Returns:
            np.ndarray of embeddings
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings


class OllamaEmbedding(EmbeddingModel):
    """Embedding model using Ollama."""

    def __init__(
        self,
        model_name: str,
        dimensions: int | None = None,
        batch_size: int | None = None,
    ):
        """Initialize with an Ollama embedding model.

        Args:
            model_name: Ollama model name (e.g., 'nomic-embed-text', 'mxbai-embed-large')
            dimensions: Expected embedding dimensions (optional, will be detected if not provided)
            batch_size: Number of texts to embed per API call (auto-detected based on model)
        """
        import httpx

        self._model_name = model_name
        self._client = httpx.Client(timeout=600.0)  # 10 min timeout for large models
        self._base_url = "http://localhost:11434"

        # Auto-detect batch size based on model size
        if batch_size is None:
            if "8b" in model_name or "4b" in model_name:
                self._batch_size = 10  # Smaller batches for large models
            elif "large" in model_name or "1b" in model_name or "2b" in model_name:
                self._batch_size = 25
            else:
                self._batch_size = 50  # Default for smaller models
        else:
            self._batch_size = batch_size

        # Detect dimensions by embedding a test string if not provided
        if dimensions is None:
            test_embedding = self._embed_batch_raw(["test"])[0]
            self._dimensions = len(test_embedding)
        else:
            self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"ollama:{self._model_name}"

    def _embed_batch_raw(self, texts: list[str]) -> list[list[float]]:
        """Get raw embeddings for a batch of texts from Ollama API."""
        response = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model_name, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using Ollama with batching.

        Args:
            texts: List of strings to embed

        Returns:
            np.ndarray of embeddings
        """
        all_embeddings = []

        # Process in batches to avoid timeouts
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_embeddings = self._embed_batch_raw(batch)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)


def load_model_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load model configuration from YAML.

    Args:
        config_path: Path to models.yaml (default: configs/models.yaml)

    Returns:
        Model configuration dictionary
    """
    if config_path is None:
        config_path = Path("configs/models.yaml")
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_embedding_model(
    model_key: str,
    config_path: str | Path | None = None,
) -> EmbeddingModel:
    """Get an embedding model by key from config.

    Args:
        model_key: Model key from config (e.g., 'st:all-mpnet-base-v2')
        config_path: Optional path to models.yaml

    Returns:
        Initialized EmbeddingModel
    """
    config = load_model_config(config_path)
    embedding_models = config.get("embedding_models", {})

    if model_key not in embedding_models:
        raise ValueError(
            f"Unknown embedding model: {model_key}. "
            f"Available: {list(embedding_models.keys())}"
        )

    model_config = embedding_models[model_key]
    provider = model_config["provider"]

    if provider == "sentence-transformers":
        return SentenceTransformerEmbedding(model_config["model_name"])
    elif provider == "ollama":
        return OllamaEmbedding(
            model_config["model_name"],
            dimensions=model_config.get("dimensions"),
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vectors.

    Args:
        a: Query vector(s) of shape (d,) or (n, d)
        b: Document vectors of shape (m, d)

    Returns:
        Similarity scores of shape (m,) or (n, m)
    """
    # Normalize vectors
    if a.ndim == 1:
        a = a.reshape(1, -1)

    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)

    # Compute cosine similarity
    similarities = np.dot(a_norm, b_norm.T)

    # If query was 1D, return 1D result
    if similarities.shape[0] == 1:
        return similarities[0]

    return similarities
