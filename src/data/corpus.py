"""Corpus loading and data structures for CPRA experiments."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import json


class ChallengeType(Enum):
    """Types of challenges that make retrieval difficult."""

    AMBIGUOUS_TERMS = "ambiguous_terms"
    NEAR_MISS = "near_miss"
    INDIRECT_REFERENCE = "indirect_reference"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    PARTIAL_MATCH = "partial_match"

    @classmethod
    def from_string(cls, s: str) -> "ChallengeType":
        """Parse challenge type from string like 'ChallengeType.AMBIGUOUS_TERMS'."""
        # Handle both "ChallengeType.AMBIGUOUS_TERMS" and "ambiguous_terms" formats
        if "." in s:
            s = s.split(".")[-1].lower()
        return cls(s)


@dataclass
class Email:
    """An email from the corpus."""

    id: str
    sender: str
    recipients: list[str]
    subject: str
    body: str
    date_sent: datetime
    cc: list[str] = field(default_factory=list)
    department: str | None = None
    topics: list[str] = field(default_factory=list)
    challenge_patterns: list[ChallengeType] = field(default_factory=list)
    has_attachments: bool = False
    email_type: str = "regular"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Email":
        """Create Email from dictionary (JSON data)."""
        challenge_patterns = []
        for cp in data.get("challenge_patterns", []):
            try:
                challenge_patterns.append(ChallengeType.from_string(cp))
            except ValueError:
                pass  # Skip unknown challenge types

        return cls(
            id=data["id"],
            sender=data["sender"],
            recipients=data.get("recipients", []),
            subject=data["subject"],
            body=data["body"],
            date_sent=datetime.fromisoformat(data["date_sent"]),
            cc=data.get("cc", []),
            department=data.get("department"),
            topics=data.get("topics", []),
            challenge_patterns=challenge_patterns,
            has_attachments=data.get("has_attachments", False),
            email_type=data.get("email_type", "regular"),
        )

    @property
    def text(self) -> str:
        """Combined subject and body for search."""
        return f"{self.subject}\n\n{self.body}"


@dataclass
class CPRARequest:
    """A CPRA (California Public Records Act) request."""

    id: str
    title: str
    description: str
    request_text: str
    primary_keywords: list[str]
    secondary_keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None
    complexity: str = "moderate"
    challenge_types: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CPRARequest":
        """Create CPRARequest from dictionary (JSON data)."""
        date_start = None
        date_end = None
        if data.get("date_range_start"):
            date_start = datetime.fromisoformat(data["date_range_start"])
        if data.get("date_range_end"):
            date_end = datetime.fromisoformat(data["date_range_end"])

        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            request_text=data.get("request_text", ""),
            primary_keywords=data.get("primary_keywords", []),
            secondary_keywords=data.get("secondary_keywords", []),
            exclude_keywords=data.get("exclude_keywords", []),
            concepts=data.get("concepts", []),
            date_range_start=date_start,
            date_range_end=date_end,
            complexity=data.get("complexity", "moderate"),
            challenge_types=data.get("challenge_types", []),
        )

    @property
    def all_keywords(self) -> list[str]:
        """All keywords (primary + secondary)."""
        return self.primary_keywords + self.secondary_keywords

    @property
    def search_text(self) -> str:
        """Text to use for semantic search (title + request text)."""
        return f"{self.title}\n\n{self.request_text}"


@dataclass
class Responsiveness:
    """Responsiveness information for an email to a request."""

    email_id: str
    cpra_request_id: str
    is_responsive: bool
    confidence: float = 1.0
    reason: str = ""
    explanation: str = ""
    matching_keywords: list[str] = field(default_factory=list)
    matching_concepts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, email_id: str, data: dict[str, Any]) -> "Responsiveness":
        """Create Responsiveness from dictionary."""
        return cls(
            email_id=email_id,
            cpra_request_id=data["cpra_request_id"],
            is_responsive=data["is_responsive"],
            confidence=data.get("confidence", 1.0),
            reason=data.get("reason", ""),
            explanation=data.get("explanation", ""),
            matching_keywords=data.get("matching_keywords", []),
            matching_concepts=data.get("matching_concepts", []),
        )


class Corpus:
    """A corpus of emails with ground truth for CPRA responsiveness."""

    def __init__(
        self,
        emails: list[Email],
        requests: list[CPRARequest],
        responsiveness_map: dict[str, list[Responsiveness]],
        metadata: dict[str, Any] | None = None,
    ):
        self.emails = emails
        self.requests = requests
        self._responsiveness_map = responsiveness_map
        self.metadata = metadata or {}

        # Build indices for fast lookup
        self._email_by_id = {e.id: e for e in emails}
        self._request_by_id = {r.id: r for r in requests}

    def get_email(self, email_id: str) -> Email | None:
        """Get email by ID."""
        return self._email_by_id.get(email_id)

    def get_request(self, request_id: str) -> CPRARequest | None:
        """Get CPRA request by ID."""
        return self._request_by_id.get(request_id)

    def is_responsive(self, email_id: str, request_id: str) -> bool:
        """Check if an email is responsive to a request."""
        responses = self._responsiveness_map.get(email_id, [])
        for resp in responses:
            if resp.cpra_request_id == request_id:
                return resp.is_responsive
        return False

    def get_responsive_emails(self, request_id: str) -> set[str]:
        """Get set of email IDs that are responsive to a request."""
        responsive = set()
        for email_id, responses in self._responsiveness_map.items():
            for resp in responses:
                if resp.cpra_request_id == request_id and resp.is_responsive:
                    responsive.add(email_id)
        return responsive

    def get_challenge_types(self, email_id: str) -> list[ChallengeType]:
        """Get challenge types for an email."""
        email = self.get_email(email_id)
        if email:
            return email.challenge_patterns
        return []

    def get_emails_by_challenge(self, challenge_type: ChallengeType) -> list[Email]:
        """Get all emails with a specific challenge type."""
        return [e for e in self.emails if challenge_type in e.challenge_patterns]

    def get_responsive_by_challenge(
        self, request_id: str, challenge_type: ChallengeType
    ) -> set[str]:
        """Get responsive emails that have a specific challenge type."""
        responsive = self.get_responsive_emails(request_id)
        return {
            eid
            for eid in responsive
            if challenge_type in self.get_challenge_types(eid)
        }

    @property
    def num_emails(self) -> int:
        return len(self.emails)

    @property
    def num_requests(self) -> int:
        return len(self.requests)


def load_corpus(corpus_path: str | Path) -> Corpus:
    """Load a corpus from a generated corpus directory.

    Args:
        corpus_path: Path to corpus directory (e.g., data/generated/corpus_20251207_153555)

    Returns:
        Loaded Corpus object
    """
    corpus_path = Path(corpus_path)

    # Load ground truth (contains emails and responsiveness)
    ground_truth_path = corpus_path / "ground_truth.json"
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    # Load CPRA requests
    requests_path = corpus_path / "cpra_requests.json"
    with open(requests_path) as f:
        requests_data = json.load(f)

    # Parse emails
    emails = [Email.from_dict(e) for e in ground_truth["emails"]]

    # Parse requests
    requests = [CPRARequest.from_dict(r) for r in requests_data]

    # Parse responsiveness map
    responsiveness_map: dict[str, list[Responsiveness]] = {}
    for email_id, responses in ground_truth.get("responsiveness_map", {}).items():
        responsiveness_map[email_id] = [
            Responsiveness.from_dict(email_id, r) for r in responses
        ]

    return Corpus(
        emails=emails,
        requests=requests,
        responsiveness_map=responsiveness_map,
        metadata=ground_truth.get("metadata", {}),
    )
