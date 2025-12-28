"""Email generation engine with LLM integration."""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from src.models.email import (
    Email, EmailType, EmailResponsiveness, ResponsivenessReason,
    EmailAttachment, GroundTruth, EmailThread
)
import re
from src.models.district import SchoolDistrict, StaffMember
from src.models.cpra import CPRARequest, ChallengeType
from src.utils.llm_client import LLMClient, LLMConfig


@dataclass
class EmailGenerationConfig:
    """Configuration for email generation."""
    total_emails: int = 2500
    responsive_rate: float = 0.15  # 15% of emails are responsive
    challenge_email_rate: float = 0.3  # 30% of responsive emails have challenges
    min_email_length: int = 50
    max_email_length: int = 500
    attachment_probability: float = 0.1
    thread_probability: float = 0.25
    cc_probability: float = 0.3
    use_llm: bool = False
    llm_provider: str = "openai"  # "openai" or "anthropic"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500

    # New settings for harder corpus generation
    keyword_free_rate_by_request: Dict[str, float] = None  # Request-specific rates
    thread_rate: float = 0.20  # Percentage of responsive emails in threads
    thread_min_length: int = 3
    thread_max_length: int = 5

    def __post_init__(self):
        if self.keyword_free_rate_by_request is None:
            self.keyword_free_rate_by_request = {}


class EmailGenerator:
    """Generates emails for the corpus with ground truth tracking."""

    def __init__(self, district: SchoolDistrict, requests: List[CPRARequest], config: EmailGenerationConfig):
        """Initialize the email generator."""
        self.district = district
        self.requests = requests
        self.config = config
        self.ground_truth = {}  # Track responsiveness
        self.email_threads = {}  # Track email threads
        self.generated_emails = []
        self.generated_thread_objects = []  # Track EmailThread objects

        # Initialize LLM client if enabled
        self.llm_client = None
        if config.use_llm:
            llm_config = LLMConfig(
                provider=config.llm_provider,
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
            self.llm_client = LLMClient(llm_config)

    def _check_contains_keywords(self, email: Email, request: CPRARequest) -> bool:
        """Check if email contains any keywords from the request."""
        text = f"{email.subject} {email.body}".lower()
        all_keywords = request.primary_keywords + request.secondary_keywords
        for keyword in all_keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text):
                return True
        return False

    def generate_corpus(self) -> GroundTruth:
        """Generate the complete email corpus with ground truth."""
        emails = []
        responsiveness_map = {}

        # Calculate how many responsive emails per request
        total_responsive = int(self.config.total_emails * self.config.responsive_rate)
        emails_per_request = total_responsive // len(self.requests)

        # Track which emails are responsive to which requests
        request_email_assignments = self._assign_responsive_emails(
            total_responsive, emails_per_request
        )

        # Generate emails
        for i in range(self.config.total_emails):
            # Determine if this email should be responsive and to which requests
            responsive_to_requests = []
            for request_id, assigned_indices in request_email_assignments.items():
                if i in assigned_indices:
                    responsive_to_requests.append(request_id)

            if responsive_to_requests:
                # Generate a responsive email
                email = self._generate_responsive_email(responsive_to_requests)
            else:
                # Generate a non-responsive email
                email = self._generate_non_responsive_email()

            emails.append(email)

            # Track responsiveness
            email_responsiveness = []
            for request in self.requests:
                is_responsive = request.id in responsive_to_requests
                if is_responsive:
                    reason, confidence = self._determine_responsiveness_details(
                        email, request
                    )
                    contains_keywords = self._check_contains_keywords(email, request)
                    resp = EmailResponsiveness(
                        email_id=email.id,
                        cpra_request_id=request.id,
                        is_responsive=True,
                        confidence=confidence,
                        reason=reason,
                        explanation=f"Email directly addresses {request.title}",
                        matching_keywords=self._find_matching_keywords(email, request),
                        matching_concepts=self._find_matching_concepts(email, request),
                        contains_any_keyword=contains_keywords,
                        challenge_types=email.challenge_patterns,
                        thread_id=email.thread_id
                    )
                else:
                    resp = EmailResponsiveness(
                        email_id=email.id,
                        cpra_request_id=request.id,
                        is_responsive=False,
                        confidence=1.0,
                        reason=ResponsivenessReason.DIRECT_MATCH,
                        explanation="Email does not address the request topic",
                        matching_keywords=[],
                        matching_concepts=[],
                        contains_any_keyword=False,
                        challenge_types=[],
                        thread_id=None
                    )
                email_responsiveness.append(resp)

            responsiveness_map[email.id] = email_responsiveness

        # Create ground truth object
        ground_truth = GroundTruth(
            emails=emails,
            responsiveness_map=responsiveness_map,
            cpra_requests=[r.id for r in self.requests],
            threads=self.generated_thread_objects
        )

        # Calculate challenge distribution
        challenge_counts = {}
        for email in emails:
            for pattern in email.challenge_patterns:
                challenge_counts[pattern] = challenge_counts.get(pattern, 0) + 1
        ground_truth.challenge_distribution = challenge_counts

        return ground_truth

    def _assign_responsive_emails(self, total_responsive: int,
                                  emails_per_request: int) -> Dict[str, List[int]]:
        """Assign which email indices will be responsive to which requests."""
        assignments = {request.id: [] for request in self.requests}
        used_indices = set()

        # Distribute responsive emails round-robin across requests
        request_ids = [r.id for r in self.requests]
        emails_assigned = 0

        while emails_assigned < total_responsive:
            # Pick a random unused index
            attempts = 0
            while attempts < 100:
                idx = random.randint(0, self.config.total_emails - 1)
                if idx not in used_indices:
                    break
                attempts += 1
            else:
                break  # Can't find more unused indices

            used_indices.add(idx)

            # Assign to the request with fewest assignments (round-robin style)
            target_request = min(request_ids, key=lambda r: len(assignments[r]))
            assignments[target_request].append(idx)
            emails_assigned += 1

        return assignments

    def _generate_responsive_email(self, request_ids: List[str]) -> Email:
        """Generate an email responsive to one or more CPRA requests."""
        # Pick primary request for content generation
        primary_request = next(r for r in self.requests if r.id == request_ids[0])

        # Select participants
        sender, recipients = self._select_participants(primary_request)

        # Determine if this should be a keyword-free email
        keyword_free_rate = self.config.keyword_free_rate_by_request.get(
            primary_request.id,
            primary_request.keyword_free_rate
        )
        is_keyword_free = random.random() < keyword_free_rate

        # Determine if this should have challenge patterns
        challenge_patterns = []
        if is_keyword_free:
            challenge_patterns.append(ChallengeType.KEYWORD_FREE)
        elif random.random() < self.config.challenge_email_rate:
            challenge_patterns = self._select_challenge_patterns(primary_request)

        # Generate email content based on challenge type
        if is_keyword_free and self.llm_client:
            subject, body = self._generate_keyword_free_content(
                primary_request, sender, recipients
            )
        elif ChallengeType.EUPHEMISM in challenge_patterns and self.llm_client:
            subject, body = self._generate_euphemism_content(
                primary_request, sender, recipients
            )
        else:
            subject, body = self._generate_responsive_content(
                primary_request, sender, recipients, challenge_patterns
            )

        # Generate date within request range
        email_date = self._generate_date_in_range(
            primary_request.date_range_start,
            primary_request.date_range_end
        )

        # Create email
        email = Email(
            id=str(uuid.uuid4()),
            sender=sender.email,
            recipients=[r.email for r in recipients],
            cc=self._maybe_add_cc(sender, recipients),
            subject=subject,
            body=body,
            date_sent=email_date,
            email_type=self._select_email_type(),
            generated_for_requests=request_ids,
            challenge_patterns=[str(cp) for cp in challenge_patterns],
            department=sender.department,
            topics=primary_request.concepts[:2]  # Use some concepts as topics
        )

        # Maybe add attachments
        if random.random() < self.config.attachment_probability:
            email.attachments = self._generate_attachments(primary_request)

        return email

    def _generate_non_responsive_email(self) -> Email:
        """Generate a routine non-responsive email."""
        # Select random participants
        sender = random.choice(self.district.staff)
        recipients = self._select_random_recipients(sender, num=random.randint(1, 3))

        # Generate routine content
        subject, body = self._generate_routine_content(sender, recipients)

        # Generate random date in the general timeframe
        email_date = self._generate_random_date()

        # Create email
        email = Email(
            id=str(uuid.uuid4()),
            sender=sender.email,
            recipients=[r.email for r in recipients],
            cc=self._maybe_add_cc(sender, recipients),
            subject=subject,
            body=body,
            date_sent=email_date,
            email_type=self._select_email_type(),
            department=sender.department,
            topics=random.sample(self.district.topics, k=random.randint(1, 2))
        )

        return email

    def _select_participants(self, request: CPRARequest) -> Tuple[StaffMember, List[StaffMember]]:
        """Select appropriate participants based on the request."""
        # Find staff in relevant departments
        relevant_staff = []
        for dept_name in request.department_targets:
            relevant_staff.extend(self.district.get_department_members(dept_name))

        if not relevant_staff:
            relevant_staff = self.district.staff

        sender = random.choice(relevant_staff)

        # Select recipients (prefer same department or supervisors)
        potential_recipients = [s for s in relevant_staff if s.email != sender.email]
        if not potential_recipients:
            potential_recipients = [s for s in self.district.staff if s.email != sender.email]

        num_recipients = random.randint(1, 3)
        recipients = random.sample(potential_recipients, min(num_recipients, len(potential_recipients)))

        return sender, recipients

    def _select_random_recipients(self, sender: StaffMember, num: int) -> List[StaffMember]:
        """Select random recipients for an email."""
        # Prefer staff from same school or department
        same_school = self.district.get_staff_by_school(sender.school)
        candidates = [s for s in same_school if s.email != sender.email]

        if len(candidates) < num:
            # Add more candidates from other schools
            candidates.extend([s for s in self.district.staff
                             if s.email != sender.email and s not in candidates])

        return random.sample(candidates, min(num, len(candidates)))

    def _generate_responsive_content(self, request: CPRARequest, sender: StaffMember,
                                    recipients: List[StaffMember],
                                    challenge_patterns: List[ChallengeType]) -> Tuple[str, str]:
        """Generate email content responsive to a CPRA request.

        Note: Uses templates for efficiency. LLM is reserved for keyword-free
        and euphemism generation where avoiding keywords is essential.
        Templates provide sufficient variety for regular responsive emails.
        """
        # Use templates based on challenge type (no LLM for regular responsive emails)
        if ChallengeType.AMBIGUOUS_TERMS in challenge_patterns:
            return self._generate_ambiguous_content(request, sender)
        elif ChallengeType.NEAR_MISS in challenge_patterns:
            return self._generate_near_miss_content(request, sender)
        elif ChallengeType.INDIRECT_REFERENCE in challenge_patterns:
            return self._generate_indirect_content(request, sender)
        else:
            return self._generate_direct_content(request, sender)

    def _generate_direct_content(self, request: CPRARequest, sender: StaffMember) -> Tuple[str, str]:
        """Generate directly responsive content."""
        # Use request-specific templates
        if request.id == "cpra_001":  # Lead testing
            subjects = [
                "Lead Testing Results - Q3 2024",
                "Water Quality Testing Schedule",
                "Re: EPA Water Standards Compliance",
                "Urgent: Lead Levels in Building C"
            ]
            bodies = [
                f"Hi Team,\\n\\nI wanted to update you on the lead testing results from last week. "
                f"The water samples from all fountains in {sender.school.name} have been tested. "
                f"Results show lead levels below 5 ppb, well within EPA standards of 15 ppb.\\n\\n"
                f"Please share with concerned parents.\\n\\nBest,\\n{sender.first_name}",

                f"Good morning,\\n\\nAs discussed, we need to schedule the quarterly water quality testing. "
                f"The lead testing contractor can come next Tuesday. All drinking fountains and kitchen "
                f"taps should be tested for lead contamination.\\n\\nPlease confirm availability.\\n\\n"
                f"{sender.full_name}"
            ]
        elif request.id == "cpra_002":  # COVID funds
            subjects = [
                "ESSER III Fund Allocation Plan",
                "Re: CARES Act Spending Priorities",
                "Federal COVID Relief - Technology Purchases",
                "ARP Funds Budget Meeting"
            ]
            bodies = [
                f"Team,\\n\\nAttached is our proposal for ESSER III fund allocation. We're recommending "
                f"40% for technology infrastructure, 30% for learning loss mitigation, and 30% for "
                f"facility improvements including HVAC upgrades.\\n\\nPlease review before tomorrow's "
                f"board meeting.\\n\\n{sender.full_name}",

                f"Hi all,\\n\\nThe federal COVID relief funds need to be obligated by September 2024. "
                f"We have $2.3M remaining in CARES Act funding that must be allocated. Priority areas "
                f"include student devices and hotspots for remote learning.\\n\\nThoughts?\\n\\n"
                f"{sender.first_name}"
            ]
        else:
            # Generic responsive content
            subjects = [f"Re: {request.title}", f"Update on {request.title}"]
            bodies = [f"Regarding {request.description}, we need to discuss the implementation details."]

        return random.choice(subjects), random.choice(bodies)

    def _generate_ambiguous_content(self, request: CPRARequest, sender: StaffMember) -> Tuple[str, str]:
        """Generate content with ambiguous terms."""
        if request.id == "cpra_001":  # Lead testing - use "lead" ambiguously
            subject = "Taking the Lead on Water Safety Initiative"
            body = (f"Hi Team,\\n\\nI'll be taking the lead on our new water safety initiative. "
                   f"As the lead coordinator, I want to ensure all our water fountains meet "
                   f"safety standards. We should lead by example in student health.\\n\\n"
                   f"The testing schedule is attached.\\n\\n{sender.full_name}")
        else:
            subject = f"Leading the {request.title} Project"
            body = f"I'll lead this initiative to ensure compliance with all requirements."

        return subject, body

    def _generate_near_miss_content(self, request: CPRARequest, sender: StaffMember) -> Tuple[str, str]:
        """Generate near-miss content (related but not quite responsive)."""
        if request.id == "cpra_001":  # Lead testing
            subject = "General Water System Maintenance"
            body = (f"Maintenance team,\\n\\nWe need to schedule routine water system maintenance "
                   f"for all buildings. This includes checking pipes, water pressure, and "
                   f"fountain functionality. Not related to any specific testing requirements.\\n\\n"
                   f"{sender.full_name}")
        elif request.id == "cpra_002":  # COVID funds
            subject = "General Budget Planning FY2024"
            body = (f"Finance team,\\n\\nLet's review our general operating budget for next year. "
                   f"We have several funding sources to consider, including state and local grants.\\n\\n"
                   f"Meeting Tuesday at 2pm.\\n\\n{sender.full_name}")
        else:
            subject = f"General Update - {sender.department or 'Operations'}"
            body = f"Team update on various initiatives and ongoing projects."

        return subject, body

    def _generate_indirect_content(self, request: CPRARequest, sender: StaffMember) -> Tuple[str, str]:
        """Generate content with indirect references."""
        if request.id == "cpra_001":  # Lead testing
            subject = "Follow-up on Yesterday's Discussion"
            body = (f"Hi,\\n\\nAs we discussed yesterday about the water situation, I think we "
                   f"should move forward with what the consultant recommended. The results "
                   f"from last month need to be addressed before parents hear about it.\\n\\n"
                   f"Let's talk more offline.\\n\\n{sender.first_name}")
        else:
            subject = "Re: That issue we discussed"
            body = (f"Following up on our conversation about the situation. We should proceed "
                   f"as planned with the recommendations from the committee.\\n\\n{sender.first_name}")

        return subject, body

    def _generate_keyword_free_content(
        self,
        request: CPRARequest,
        sender: StaffMember,
        recipients: List[StaffMember]
    ) -> Tuple[str, str]:
        """Generate email content that is responsive but contains NO keywords."""
        if not self.llm_client:
            # Fallback if LLM not available - use indirect content
            return self._generate_indirect_content(request, sender)

        cpra_dict = {
            'title': request.title,
            'description': request.description,
            'request_text': request.request_text,
            'primary_keywords': request.primary_keywords,
            'secondary_keywords': request.secondary_keywords,
        }
        sender_dict = {
            'full_name': sender.full_name,
            'first_name': sender.first_name,
            'role': sender.role,
        }
        recipient_dicts = [{'full_name': r.full_name} for r in recipients]

        subject, body, success = self.llm_client.generate_keyword_free_email(
            cpra_dict, sender_dict, recipient_dicts
        )

        if not success:
            # Fallback to indirect content if keyword-free generation failed
            print(f"Warning: Keyword-free generation failed for {request.id}, falling back to indirect")
            return self._generate_indirect_content(request, sender)

        return subject, body

    def _generate_euphemism_content(
        self,
        request: CPRARequest,
        sender: StaffMember,
        recipients: List[StaffMember]
    ) -> Tuple[str, str]:
        """Generate email content using euphemisms instead of direct keywords."""
        if not self.llm_client:
            # Fallback if LLM not available
            return self._generate_indirect_content(request, sender)

        euphemism_patterns = request.euphemism_patterns or [
            f"the situation at {sender.school.name if sender.school else 'the school'}",
            "what we discussed",
            "the issue from last month",
            "that matter",
        ]

        cpra_dict = {
            'title': request.title,
            'description': request.description,
        }
        sender_dict = {
            'full_name': sender.full_name,
            'first_name': sender.first_name,
            'role': sender.role,
        }
        recipient_dicts = [{'full_name': r.full_name} for r in recipients]

        subject, body = self.llm_client.generate_euphemism_email(
            cpra_dict, sender_dict, recipient_dicts, euphemism_patterns
        )

        return subject, body

    def _generate_routine_content(self, sender: StaffMember,
                                 recipients: List[StaffMember]) -> Tuple[str, str]:
        """Generate routine non-responsive email content.

        Note: Uses templates by default for efficiency. LLM generation for
        non-responsive emails is disabled to avoid unnecessary API costs.
        The templates provide sufficient variety for testing purposes.
        """
        # Use templates for non-responsive emails (no need for LLM)
        # LLM is reserved for keyword-free and euphemism generation where it's needed
        templates = [
            ("Staff Meeting - {date}", "Reminder: Staff meeting this {day} at 3pm in the conference room."),
            ("Schedule Change", "Due to testing, the bell schedule will be modified next week."),
            ("Parking Lot Reminder", "Please remember to display your parking permits."),
            ("Supply Order", "Please submit your supply requests by end of day Friday."),
            ("Professional Development", "Don't forget to register for the PD session next month."),
            ("Lunch Schedule", "The lunch schedule has been updated for next week."),
            ("Field Trip Permission", "Permission slips for the upcoming field trip are due Monday."),
            ("Substitute Coverage", "Looking for coverage for my 3rd period class tomorrow."),
            ("Technology Issue", "The projector in room 205 needs replacement."),
            ("Parent Conference", "Available times for parent conferences have been posted."),
        ]

        subject_template, body_template = random.choice(templates)

        # Fill in templates
        from datetime import datetime
        next_week = datetime.now() + timedelta(days=7)
        subject = subject_template.format(date=next_week.strftime("%m/%d"))
        body = f"{body_template}\\n\\nThanks,\\n{sender.first_name}"

        return subject, body

    def _select_challenge_patterns(self, request: CPRARequest) -> List[ChallengeType]:
        """Select challenge patterns to apply to an email."""
        if request.challenge_types:
            # Use request-specific challenges with some probability
            num_challenges = random.randint(1, min(2, len(request.challenge_types)))
            return random.sample(request.challenge_types, num_challenges)
        return []

    def _determine_responsiveness_details(self, email: Email,
                                         request: CPRARequest) -> Tuple[ResponsivenessReason, float]:
        """Determine the reason and confidence for responsiveness."""
        # Check challenge patterns
        if ChallengeType.INDIRECT_REFERENCE.value in email.challenge_patterns:
            return ResponsivenessReason.INDIRECT_REFERENCE, 0.7
        elif ChallengeType.AMBIGUOUS_TERMS.value in email.challenge_patterns:
            return ResponsivenessReason.DIRECT_MATCH, 0.85
        elif email.attachments:
            return ResponsivenessReason.ATTACHMENT_REFERENCE, 0.9
        else:
            return ResponsivenessReason.DIRECT_MATCH, 0.95

    def _find_matching_keywords(self, email: Email, request: CPRARequest) -> List[str]:
        """Find keywords from the request that appear in the email."""
        email_text = f"{email.subject} {email.body}".lower()
        matching = []

        for keyword in request.primary_keywords + request.secondary_keywords:
            if keyword.lower() in email_text:
                matching.append(keyword)

        return matching

    def _find_matching_concepts(self, email: Email, request: CPRARequest) -> List[str]:
        """Find conceptual matches between email and request."""
        # Simplified concept matching
        return [c for c in request.concepts if any(t in c.lower() for t in email.topics)]

    def _generate_date_in_range(self, start: Optional[datetime],
                               end: Optional[datetime]) -> datetime:
        """Generate a date within the specified range."""
        if not start:
            start = datetime(2023, 1, 1)
        if not end:
            end = datetime.now()

        # Generate random timestamp between start and end
        time_between = end - start
        days_between = time_between.days
        random_days = random.randint(0, days_between)
        random_date = start + timedelta(days=random_days)

        # Add random time
        random_date = random_date.replace(
            hour=random.randint(6, 18),
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )

        return random_date

    def _generate_random_date(self) -> datetime:
        """Generate a random date in the past 2 years."""
        days_ago = random.randint(1, 730)
        date = datetime.now() - timedelta(days=days_ago)

        # Add random time (business hours)
        date = date.replace(
            hour=random.randint(6, 18),
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )

        return date

    def _maybe_add_cc(self, sender: StaffMember,
                     recipients: List[StaffMember]) -> List[str]:
        """Maybe add CC recipients."""
        if random.random() < self.config.cc_probability:
            # Add supervisor or additional staff
            cc_candidates = [s for s in self.district.staff
                            if s.email != sender.email
                            and s not in recipients]
            if cc_candidates:
                num_cc = random.randint(1, min(2, len(cc_candidates)))
                return [s.email for s in random.sample(cc_candidates, num_cc)]
        return []

    def _select_email_type(self) -> EmailType:
        """Select the type of email."""
        weights = [0.6, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]
        types = list(EmailType)
        return random.choices(types, weights=weights)[0]

    def _generate_attachments(self, request: CPRARequest) -> List[EmailAttachment]:
        """Generate relevant attachments for an email."""
        attachment_types = [
            ("test_results.pdf", "PDF", 245, "Water quality test results"),
            ("budget_summary.xlsx", "Excel", 89, "Budget allocation spreadsheet"),
            ("meeting_notes.docx", "Word", 34, "Meeting notes and action items"),
            ("presentation.pptx", "PowerPoint", 1250, "Board presentation slides"),
            ("data_export.csv", "CSV", 156, "Exported data for analysis")
        ]

        num_attachments = random.randint(1, 2)
        selected = random.sample(attachment_types, num_attachments)

        return [EmailAttachment(filename=f, file_type=t, size_kb=s, description=d)
               for f, t, s, d in selected]