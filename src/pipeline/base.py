"""Base classes for search pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.data import CPRARequest, SearchableDocument


@dataclass
class SearchResult:
    """A single search result with score."""

    doc_id: str  # Document ID (email_id or thread_id)
    score: float
    matched_terms: list[str] = field(default_factory=list)

    def __lt__(self, other: "SearchResult") -> bool:
        """Compare by score (for sorting, higher is better)."""
        return self.score < other.score


class SearchPipeline(ABC):
    """Abstract base class for search pipelines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the pipeline."""
        pass

    @abstractmethod
    def search(
        self, request: CPRARequest, documents: list[SearchableDocument]
    ) -> list[SearchResult]:
        """Search for documents responsive to a CPRA request.

        Args:
            request: The CPRA request to search for
            documents: List of searchable documents (emails or concatenated threads)

        Returns:
            List of SearchResult objects, sorted by score (highest first)
        """
        pass

    def get_predictions(
        self, results: list[SearchResult], threshold: float = 0.5
    ) -> set[str]:
        """Convert search results to binary predictions using a threshold.

        Args:
            results: List of SearchResult objects
            threshold: Score threshold for positive prediction

        Returns:
            Set of document IDs predicted as responsive
        """
        return {r.doc_id for r in results if r.score >= threshold}

    def get_ranked_ids(self, results: list[SearchResult]) -> list[str]:
        """Extract ranked document IDs from results.

        Args:
            results: List of SearchResult objects (should be sorted by score)

        Returns:
            List of document IDs in ranked order
        """
        return [r.doc_id for r in results]
