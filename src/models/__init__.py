# Model abstractions for embeddings and LLMs
from .embeddings import (
    EmbeddingModel,
    OllamaEmbedding,
    SentenceTransformerEmbedding,
    cosine_similarity,
    get_embedding_model,
)

__all__ = [
    "EmbeddingModel",
    "OllamaEmbedding",
    "SentenceTransformerEmbedding",
    "cosine_similarity",
    "get_embedding_model",
]
