"""Data models for CPRA requests."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum


class RequestComplexity(Enum):
    """Complexity level of CPRA request for testing purposes."""
    SIMPLE = "simple"       # Single clear topic
    MODERATE = "moderate"   # Multiple related topics
    COMPLEX = "complex"     # Multiple topics with temporal/contextual constraints


class ChallengeType(Enum):
    """Types of challenges for testing detection algorithms."""
    # Original challenge types
    AMBIGUOUS_TERMS = "ambiguous_terms"
    NEAR_MISS = "near_miss"
    INDIRECT_REFERENCE = "indirect_reference"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    PARTIAL_MATCH = "partial_match"

    # New challenge types for harder corpus
    KEYWORD_FREE = "keyword_free"           # Responsive but contains zero request keywords
    BURIED_IN_THREAD = "buried_in_thread"   # Responsive content 2-3 replies deep
    ABBREVIATION_HEAVY = "abbreviation_heavy"  # Uses abbreviations/acronyms not in keyword list
    EUPHEMISM = "euphemism"                 # Uses euphemisms to avoid direct mention
    ATTACHMENT_ONLY = "attachment_only"     # Email body is benign, attachment is responsive
    TEMPORAL_CONTEXT = "temporal_context"   # Date relevance requires understanding context


@dataclass
class CPRARequest:
    """Represents a CPRA request for document retrieval."""
    id: str
    title: str
    description: str
    request_text: str  # The actual CPRA request as submitted
    date_submitted: datetime
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None

    # Keywords and concepts for testing
    primary_keywords: List[str] = field(default_factory=list)
    secondary_keywords: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)

    # Semantic concepts (for embedding-based approaches)
    concepts: List[str] = field(default_factory=list)

    # Testing metadata
    complexity: RequestComplexity = RequestComplexity.MODERATE
    challenge_types: List[ChallengeType] = field(default_factory=list)

    # Ground truth helpers
    responsive_email_patterns: List[str] = field(default_factory=list)
    non_responsive_patterns: List[str] = field(default_factory=list)

    # Additional metadata
    requester_name: str = "Public Records Requester"
    department_targets: List[str] = field(default_factory=list)  # Which departments likely have responsive docs

    # Difficulty settings for realistic corpus generation
    keyword_free_rate: float = 0.0  # Percentage of responsive emails without any keywords (0.0-1.0)
    euphemism_patterns: List[str] = field(default_factory=list)  # Alternative phrases to use instead of keywords
    abbreviation_patterns: Dict[str, str] = field(default_factory=dict)  # keyword -> abbreviation mapping

    def is_within_date_range(self, email_date: datetime) -> bool:
        """Check if an email date falls within the request's date range."""
        if not self.date_range_start and not self.date_range_end:
            return True  # No date restrictions

        if self.date_range_start and email_date < self.date_range_start:
            return False
        if self.date_range_end and email_date > self.date_range_end:
            return False

        return True

    def get_search_query(self) -> str:
        """Generate a basic keyword search query from primary keywords."""
        return " OR ".join(self.primary_keywords)

    def get_expanded_query(self) -> List[str]:
        """Get all keywords for expanded search."""
        return self.primary_keywords + self.secondary_keywords


@dataclass
class CPRARequestSet:
    """Collection of CPRA requests for testing."""
    requests: List[CPRARequest]
    metadata: Dict = field(default_factory=dict)

    def get_request_by_id(self, request_id: str) -> Optional[CPRARequest]:
        """Get a specific request by ID."""
        return next((r for r in self.requests if r.id == request_id), None)

    def get_all_keywords(self) -> List[str]:
        """Get all unique keywords across all requests."""
        keywords = set()
        for request in self.requests:
            keywords.update(request.primary_keywords)
            keywords.update(request.secondary_keywords)
        return list(keywords)

    def get_all_concepts(self) -> List[str]:
        """Get all unique concepts across all requests."""
        concepts = set()
        for request in self.requests:
            concepts.update(request.concepts)
        return list(concepts)