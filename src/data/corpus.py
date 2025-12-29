"""Corpus loading and data structures for CPRA experiments."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import json


class ChallengeType(Enum):
    """Types of challenges that make retrieval difficult."""

    # Responsive categories
    DIRECT_MATCH = "DIRECT_MATCH"
    AMBIGUOUS_TERMS = "AMBIGUOUS_TERMS"
    INDIRECT_REFERENCE = "INDIRECT_REFERENCE"
    TECHNICAL_JARGON = "TECHNICAL_JARGON"
    TEMPORAL_REFERENCE = "TEMPORAL_REFERENCE"
    BURIED_IN_THREAD = "BURIED_IN_THREAD"

    # Non-responsive categories
    KEYWORD_FALSE_POSITIVE = "KEYWORD_FALSE_POSITIVE"
    ADJACENT_TOPIC = "ADJACENT_TOPIC"
    TRUE_NEGATIVE = "TRUE_NEGATIVE"

    @classmethod
    def from_string(cls, s: str) -> "ChallengeType":
        """Parse challenge type from string."""
        # Handle both "ChallengeType.AMBIGUOUS_TERMS" and "AMBIGUOUS_TERMS" formats
        if "." in s:
            s = s.split(".")[-1]
        return cls(s)


@dataclass
class Email:
    """An email from the corpus."""

    id: str
    from_addr: str  # 'from' in JSON, renamed to avoid Python keyword
    to: list[str]
    subject: str
    body: str
    date: datetime
    cc: list[str] = field(default_factory=list)
    thread_id: str | None = None
    thread_position: int | None = None
    thread_length: int | None = None
    has_attachment: bool = False
    attachment_names: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Email":
        """Create Email from dictionary (JSON data)."""
        return cls(
            id=data["id"],
            from_addr=data["from"],
            to=data.get("to", []),
            subject=data["subject"],
            body=data["body"],
            date=datetime.fromisoformat(data["date"]),
            cc=data.get("cc", []),
            thread_id=data.get("thread_id"),
            thread_position=data.get("thread_position"),
            thread_length=data.get("thread_length"),
            has_attachment=data.get("has_attachment", False),
            attachment_names=data.get("attachment_names", []),
        )

    @property
    def text(self) -> str:
        """Combined subject and body for search."""
        return f"{self.subject}\n\n{self.body}"

    @property
    def is_in_thread(self) -> bool:
        """Check if this email is part of a thread."""
        return self.thread_id is not None


@dataclass
class CPRARequest:
    """A CPRA (California Public Records Act) request."""

    id: str
    title: str
    request_text: str
    keywords: list[str] = field(default_factory=list)
    date_submitted: datetime | None = None
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CPRARequest":
        """Create CPRARequest from dictionary (JSON data)."""
        date_submitted = None
        date_start = None
        date_end = None

        if data.get("date_submitted"):
            date_submitted = datetime.fromisoformat(data["date_submitted"])

        # Handle date_range object with nested start/end
        date_range = data.get("date_range", {})
        if date_range.get("start"):
            date_start = datetime.fromisoformat(date_range["start"])
        if date_range.get("end"):
            date_end = datetime.fromisoformat(date_range["end"])

        return cls(
            id=data["id"],
            title=data["title"],
            request_text=data.get("request_text", ""),
            keywords=data.get("keywords", []),
            date_submitted=date_submitted,
            date_range_start=date_start,
            date_range_end=date_end,
            notes=data.get("notes", ""),
        )

    @property
    def search_text(self) -> str:
        """Text to use for semantic search (title + request text)."""
        return f"{self.title}\n\n{self.request_text}"


@dataclass
class GroundTruthLabel:
    """Ground truth annotation for an email's responsiveness."""

    email_id: str
    responsive: bool
    challenge_type: ChallengeType | None
    buried_in_thread: bool = False
    reasoning: str = ""
    keywords_present: list[str] = field(default_factory=list)
    keywords_absent: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, email_id: str, data: dict[str, Any]) -> "GroundTruthLabel":
        """Create GroundTruthLabel from dictionary."""
        challenge_type = None
        if data.get("challenge_type"):
            try:
                challenge_type = ChallengeType.from_string(data["challenge_type"])
            except ValueError:
                pass  # Unknown challenge type

        return cls(
            email_id=email_id,
            responsive=data["responsive"],
            challenge_type=challenge_type,
            buried_in_thread=data.get("buried_in_thread", False),
            reasoning=data.get("reasoning", ""),
            keywords_present=data.get("keywords_present", []),
            keywords_absent=data.get("keywords_absent", []),
        )


@dataclass
class SearchableDocument:
    """A document for search - either a single email or concatenated thread."""

    id: str  # Either email_id or thread_id
    text: str  # Concatenated text for search
    email_ids: list[str]  # Email IDs contained in this document
    is_thread: bool

    @classmethod
    def from_email(cls, email: Email) -> "SearchableDocument":
        """Create searchable document from single email."""
        return cls(
            id=email.id,
            text=email.text,
            email_ids=[email.id],
            is_thread=False,
        )

    @classmethod
    def from_thread(cls, thread_id: str, emails: list[Email]) -> "SearchableDocument":
        """Create searchable document from thread (emails sorted by position)."""
        sorted_emails = sorted(emails, key=lambda e: e.thread_position or 0)
        combined_text = "\n\n---\n\n".join(e.text for e in sorted_emails)
        return cls(
            id=thread_id,
            text=combined_text,
            email_ids=[e.id for e in sorted_emails],
            is_thread=True,
        )


class Corpus:
    """A corpus of emails with ground truth for CPRA responsiveness."""

    def __init__(
        self,
        emails: list[Email],
        request: CPRARequest,
        ground_truth: dict[str, GroundTruthLabel],
        metadata: dict[str, Any] | None = None,
    ):
        self.emails = emails
        self.request = request
        self._ground_truth = ground_truth
        self.metadata = metadata or {}

        # Build indices for fast lookup
        self._email_by_id = {e.id: e for e in emails}

        # Build thread index
        self._threads: dict[str, list[Email]] = {}
        for email in emails:
            if email.thread_id:
                if email.thread_id not in self._threads:
                    self._threads[email.thread_id] = []
                self._threads[email.thread_id].append(email)

        # Cache for searchable documents
        self._searchable_docs: list[SearchableDocument] | None = None

    def get_email(self, email_id: str) -> Email | None:
        """Get email by ID."""
        return self._email_by_id.get(email_id)

    def get_searchable_documents(self) -> list[SearchableDocument]:
        """Get documents for search - threads concatenated, standalones as-is."""
        if self._searchable_docs is not None:
            return self._searchable_docs

        docs = []
        seen_thread_ids: set[str] = set()

        for email in self.emails:
            if email.thread_id:
                if email.thread_id not in seen_thread_ids:
                    thread_emails = self._threads[email.thread_id]
                    docs.append(
                        SearchableDocument.from_thread(email.thread_id, thread_emails)
                    )
                    seen_thread_ids.add(email.thread_id)
            else:
                docs.append(SearchableDocument.from_email(email))

        self._searchable_docs = docs
        return docs

    def document_to_email_ids(self, doc_id: str) -> list[str]:
        """Map a searchable document ID back to email IDs."""
        if doc_id in self._threads:
            return [e.id for e in self._threads[doc_id]]
        return [doc_id]

    def is_responsive(self, email_id: str) -> bool:
        """Check if an email is responsive."""
        label = self._ground_truth.get(email_id)
        return label.responsive if label else False

    def get_responsive_emails(self) -> set[str]:
        """Get set of responsive email IDs."""
        return {
            eid for eid, label in self._ground_truth.items() if label.responsive
        }

    def get_challenge_type(self, email_id: str) -> ChallengeType | None:
        """Get challenge type for an email."""
        label = self._ground_truth.get(email_id)
        return label.challenge_type if label else None

    def get_emails_by_challenge(self, challenge_type: ChallengeType) -> list[Email]:
        """Get all emails with a specific challenge type."""
        return [
            self._email_by_id[eid]
            for eid, label in self._ground_truth.items()
            if label.challenge_type == challenge_type and eid in self._email_by_id
        ]

    def get_responsive_by_challenge(self, challenge_type: ChallengeType) -> set[str]:
        """Get responsive emails that have a specific challenge type."""
        return {
            eid
            for eid, label in self._ground_truth.items()
            if label.responsive and label.challenge_type == challenge_type
        }

    @property
    def num_emails(self) -> int:
        return len(self.emails)

    @property
    def num_searchable_documents(self) -> int:
        return len(self.get_searchable_documents())

    @property
    def num_threads(self) -> int:
        return len(self._threads)


def load_corpus(corpus_path: str | Path) -> Corpus:
    """Load a corpus from a corpus directory.

    Expected structure:
        corpus_path/
            request.json      # Single CPRA request
            emails.json       # All emails
            ground_truth.json # Responsiveness labels

    Args:
        corpus_path: Path to corpus directory

    Returns:
        Loaded Corpus object
    """
    corpus_path = Path(corpus_path)

    # Load single request
    with open(corpus_path / "request.json") as f:
        request = CPRARequest.from_dict(json.load(f))

    # Load emails (separate file)
    with open(corpus_path / "emails.json") as f:
        emails_data = json.load(f)
    emails = [Email.from_dict(e) for e in emails_data["emails"]]

    # Load ground truth labels
    with open(corpus_path / "ground_truth.json") as f:
        gt_data = json.load(f)
    ground_truth = {
        eid: GroundTruthLabel.from_dict(eid, label)
        for eid, label in gt_data["labels"].items()
    }

    return Corpus(
        emails=emails,
        request=request,
        ground_truth=ground_truth,
        metadata=gt_data.get("metadata", {}),
    )
