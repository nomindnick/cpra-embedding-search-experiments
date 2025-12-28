"""Data models for emails and ground truth tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Set
from enum import Enum


class EmailType(Enum):
    """Type of email for categorization."""
    REGULAR = "regular"
    REPLY = "reply"
    FORWARD = "forward"
    MEETING_REQUEST = "meeting_request"
    ANNOUNCEMENT = "announcement"
    REPORT = "report"
    MEMO = "memo"


class ResponsivenessReason(Enum):
    """Reasons why an email is responsive to a CPRA request."""
    DIRECT_MATCH = "direct_match"              # Directly discusses the requested topic
    ATTACHMENT_REFERENCE = "attachment_ref"     # References relevant attachment
    FORWARD_CONTAINS = "forward_contains"       # Forwarded content is responsive
    PARTIAL_MATCH = "partial_match"            # Contains some but not all elements
    THREAD_CONTEXT = "thread_context"          # Part of responsive email thread
    INDIRECT_REFERENCE = "indirect_ref"        # Indirectly references the topic


@dataclass
class EmailAttachment:
    """Represents an email attachment."""
    filename: str
    file_type: str
    size_kb: int
    description: Optional[str] = None


@dataclass
class Email:
    """Represents an email in the corpus."""
    # Required fields (no defaults) must come first
    id: str
    sender: str
    recipients: List[str]
    subject: str
    body: str
    date_sent: datetime

    # Optional fields with defaults
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    email_type: EmailType = EmailType.REGULAR

    # Thread information
    thread_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)

    # Attachments
    attachments: List[EmailAttachment] = field(default_factory=list)

    # Metadata for generation
    generated_for_requests: List[str] = field(default_factory=list)  # Which CPRA requests this was generated for
    challenge_patterns: List[str] = field(default_factory=list)  # Which challenge patterns were applied
    department: Optional[str] = None
    topics: List[str] = field(default_factory=list)

    def get_all_participants(self) -> Set[str]:
        """Get all email participants."""
        participants = {self.sender}
        participants.update(self.recipients)
        participants.update(self.cc)
        participants.update(self.bcc)
        return participants

    def has_attachments(self) -> bool:
        """Check if email has attachments."""
        return len(self.attachments) > 0

    def __hash__(self):
        return hash(self.id)


@dataclass
class EmailThread:
    """Represents an email thread with multiple messages."""
    thread_id: str
    emails: List[Email]  # Ordered by date (oldest first)
    responsive_email_indices: List[int] = field(default_factory=list)  # Which emails contain responsive content
    surface_email_index: int = -1  # Index of the most recent (surface) email

    def get_surface_email(self) -> Optional[Email]:
        """Get the most recent email in the thread (what a user would see first)."""
        if self.surface_email_index >= 0 and self.surface_email_index < len(self.emails):
            return self.emails[self.surface_email_index]
        return self.emails[-1] if self.emails else None

    def get_responsive_emails(self) -> List[Email]:
        """Get all emails that contain responsive content."""
        return [self.emails[i] for i in self.responsive_email_indices if i < len(self.emails)]

    def is_buried(self) -> bool:
        """Check if responsive content is buried (not in surface email)."""
        if not self.responsive_email_indices:
            return False
        surface_idx = self.surface_email_index if self.surface_email_index >= 0 else len(self.emails) - 1
        return surface_idx not in self.responsive_email_indices


@dataclass
class EmailResponsiveness:
    """Tracks whether an email is responsive to a CPRA request."""
    email_id: str
    cpra_request_id: str
    is_responsive: bool
    confidence: float  # 0.0 to 1.0
    reason: ResponsivenessReason
    explanation: str  # Human-readable explanation
    matching_keywords: List[str] = field(default_factory=list)
    matching_concepts: List[str] = field(default_factory=list)

    # New fields for keyword analysis
    contains_any_keyword: bool = True  # Whether the email contains any request keywords
    challenge_types: List[str] = field(default_factory=list)  # Which challenge types apply
    thread_id: Optional[str] = None  # If part of a thread


@dataclass
class GroundTruth:
    """Ground truth for the entire email corpus."""
    emails: List[Email]
    responsiveness_map: Dict[str, List[EmailResponsiveness]]  # email_id -> list of responsiveness entries
    cpra_requests: List[str]  # List of CPRA request IDs

    # Thread tracking
    threads: List[EmailThread] = field(default_factory=list)
    email_to_thread: Dict[str, str] = field(default_factory=dict)  # email_id -> thread_id

    # Statistics
    total_emails: int = 0
    responsive_emails: int = 0
    challenge_distribution: Dict[str, int] = field(default_factory=dict)

    # Keyword analysis statistics
    keyword_free_count: int = 0  # Responsive emails without any keywords
    keyword_containing_count: int = 0  # Responsive emails with keywords
    threaded_responsive_count: int = 0  # Responsive emails in threads
    buried_responsive_count: int = 0  # Responsive content buried in threads

    def __post_init__(self):
        """Calculate statistics after initialization."""
        self.total_emails = len(self.emails)
        self.responsive_emails = len([
            e for e in self.emails
            if any(r.is_responsive for r in self.responsiveness_map.get(e.id, []))
        ])

        # Calculate keyword analysis statistics
        for responses in self.responsiveness_map.values():
            for resp in responses:
                if resp.is_responsive:
                    if resp.contains_any_keyword:
                        self.keyword_containing_count += 1
                    else:
                        self.keyword_free_count += 1
                    if resp.thread_id:
                        self.threaded_responsive_count += 1

        # Count buried responsive content
        for thread in self.threads:
            if thread.is_buried():
                self.buried_responsive_count += len(thread.responsive_email_indices)

        # Build email_to_thread mapping
        for thread in self.threads:
            for email in thread.emails:
                self.email_to_thread[email.id] = thread.thread_id

    def get_responsive_emails(self, cpra_request_id: str) -> List[Email]:
        """Get all emails responsive to a specific CPRA request."""
        responsive_ids = set()
        for email_id, responses in self.responsiveness_map.items():
            for resp in responses:
                if resp.cpra_request_id == cpra_request_id and resp.is_responsive:
                    responsive_ids.add(email_id)

        return [e for e in self.emails if e.id in responsive_ids]

    def get_responsiveness(self, email_id: str, cpra_request_id: str) -> Optional[EmailResponsiveness]:
        """Get the responsiveness of a specific email to a specific request."""
        responses = self.responsiveness_map.get(email_id, [])
        for resp in responses:
            if resp.cpra_request_id == cpra_request_id:
                return resp
        return None

    def is_responsive(self, email_id: str, cpra_request_id: str) -> bool:
        """Check if an email is responsive to a specific request."""
        resp = self.get_responsiveness(email_id, cpra_request_id)
        return resp.is_responsive if resp else False

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the corpus."""
        stats = {
            "total_emails": self.total_emails,
            "responsive_emails": self.responsive_emails,
            "responsive_rate": self.responsive_emails / self.total_emails if self.total_emails > 0 else 0,
            "emails_per_request": {},
            "challenge_distribution": self.challenge_distribution,
            "responsiveness_reasons": {},
            # New keyword analysis stats
            "keyword_analysis": {
                "keyword_free_count": self.keyword_free_count,
                "keyword_containing_count": self.keyword_containing_count,
                "keyword_free_rate": self.keyword_free_count / self.responsive_emails if self.responsive_emails > 0 else 0,
                "expected_keyword_recall": self.keyword_containing_count / self.responsive_emails if self.responsive_emails > 0 else 0,
            },
            # Thread statistics
            "thread_analysis": {
                "total_threads": len(self.threads),
                "threaded_responsive_count": self.threaded_responsive_count,
                "buried_responsive_count": self.buried_responsive_count,
            }
        }

        # Count emails per CPRA request
        for cpra_id in self.cpra_requests:
            responsive_count = len(self.get_responsive_emails(cpra_id))
            stats["emails_per_request"][cpra_id] = responsive_count

        # Count responsiveness reasons
        reason_counts = {}
        for responses in self.responsiveness_map.values():
            for resp in responses:
                if resp.is_responsive:
                    reason = resp.reason.value
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        stats["responsiveness_reasons"] = reason_counts

        return stats