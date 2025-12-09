"""Base classes for search pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.data import CPRARequest, Email


@dataclass
class SearchResult:
    """A single search result with score."""

    email_id: str
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
        self, request: CPRARequest, emails: list[Email]
    ) -> list[SearchResult]:
        """Search for emails responsive to a CPRA request.

        Args:
            request: The CPRA request to search for
            emails: List of emails to search through

        Returns:
            List of SearchResult objects, sorted by score (highest first)
        """
        pass

    def search_all(
        self, requests: list[CPRARequest], emails: list[Email]
    ) -> dict[str, list[SearchResult]]:
        """Search for all requests against all emails.

        Args:
            requests: List of CPRA requests
            emails: List of emails to search through

        Returns:
            Dict mapping request_id -> list of SearchResults (sorted by score)
        """
        results = {}
        for request in requests:
            results[request.id] = self.search(request, emails)
        return results

    def get_predictions(
        self, results: list[SearchResult], threshold: float = 0.5
    ) -> set[str]:
        """Convert search results to binary predictions using a threshold.

        Args:
            results: List of SearchResult objects
            threshold: Score threshold for positive prediction

        Returns:
            Set of email IDs predicted as responsive
        """
        return {r.email_id for r in results if r.score >= threshold}

    def get_ranked_ids(self, results: list[SearchResult]) -> list[str]:
        """Extract ranked email IDs from results.

        Args:
            results: List of SearchResult objects (should be sorted by score)

        Returns:
            List of email IDs in ranked order
        """
        return [r.email_id for r in results]
