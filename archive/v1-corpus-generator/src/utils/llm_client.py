"""LLM client for generating email content."""

import os
import re
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import json

# Optional imports for LLM providers
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: str = "openai"  # "openai" or "anthropic"
    model: str = "gpt-4"  # or "claude-3-sonnet"
    temperature: float = 0.7
    max_tokens: int = 500


class LLMClient:
    """Client for generating email content using LLMs."""

    def __init__(self, config: LLMConfig):
        """Initialize the LLM client."""
        self.config = config

        if config.provider == "openai" and HAS_OPENAI:
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif config.provider == "anthropic" and HAS_ANTHROPIC:
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            self.client = None

    def generate_responsive_email(
        self,
        cpra_request: dict,
        sender: dict,
        recipients: list,
        challenge_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """Generate an email responsive to a CPRA request."""

        prompt = self._build_responsive_prompt(cpra_request, sender, recipients, challenge_type)

        if not self.client:
            # Fallback to template-based generation
            return self._fallback_generation(cpra_request, sender, challenge_type)

        try:
            if self.config.provider == "openai" and HAS_OPENAI:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are generating realistic emails for a school district."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                content = response.choices[0].message.content

            elif self.config.provider == "anthropic" and HAS_ANTHROPIC:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

            # Parse the response to extract subject and body
            return self._parse_email_content(content)

        except Exception as e:
            print(f"LLM generation failed: {e}, falling back to templates")
            return self._fallback_generation(cpra_request, sender, challenge_type)

    def generate_non_responsive_email(
        self,
        sender: dict,
        recipients: list,
        topic: str
    ) -> Tuple[str, str]:
        """Generate a routine non-responsive email."""

        prompt = self._build_routine_prompt(sender, recipients, topic)

        if not self.client:
            return self._fallback_routine_generation(sender, topic)

        try:
            if self.config.provider == "openai" and HAS_OPENAI:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are generating routine school district emails."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                content = response.choices[0].message.content

            elif self.config.provider == "anthropic" and HAS_ANTHROPIC:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

            return self._parse_email_content(content)

        except Exception as e:
            print(f"LLM generation failed: {e}, falling back to templates")
            return self._fallback_routine_generation(sender, topic)

    def _build_responsive_prompt(
        self,
        cpra_request: dict,
        sender: dict,
        recipients: list,
        challenge_type: Optional[str]
    ) -> str:
        """Build prompt for generating responsive email."""

        base_prompt = f"""Generate a realistic internal email from a school district staff member. This email should discuss the topic described below in a natural way - it's a regular work email between colleagues, NOT a response to a records request.

Topic the email should discuss: {cpra_request.get('title', '')}
Context: {cpra_request.get('request_text', '')}
Related keywords to incorporate naturally: {', '.join(cpra_request.get('primary_keywords', [])[:3])}

Sender: {sender.get('full_name', '')} ({sender.get('role', '')})
Recipients: {', '.join([r.get('full_name', '') for r in recipients])}

Write a natural workplace email - updates, questions, scheduling, sharing information, etc. Do NOT write a formal response to a public records request.
"""

        if challenge_type == "ambiguous_terms":
            base_prompt += """
Special instruction: Use ambiguous terms that could have multiple meanings. For example, if discussing lead testing in water, also use "lead" in the context of leadership (e.g., "I'll take the lead on this" or "our lead coordinator").
"""
        elif challenge_type == "near_miss":
            base_prompt += """
Special instruction: Write about a RELATED but slightly different topic. For example, if the topic is lead testing, write about general water fountain maintenance or plumbing repairs instead.
"""
        elif challenge_type == "indirect_reference":
            base_prompt += """
Special instruction: Reference the topic indirectly using pronouns or vague references like "the situation we discussed yesterday", "that issue", or "the results from last month" rather than naming it directly.
"""

        base_prompt += """
Format your response as JSON:
{
    "subject": "email subject line",
    "body": "email body text"
}
"""
        return base_prompt

    def _build_routine_prompt(self, sender: dict, recipients: list, topic: str) -> str:
        """Build prompt for routine email generation."""
        return f"""Generate a routine school district email about {topic}.

Sender: {sender.get('full_name', '')} ({sender.get('role', '')})
Recipients: {', '.join([r.get('full_name', '') for r in recipients])}

This should be a normal, non-sensitive email about daily school operations.

Format your response as JSON:
{{
    "subject": "email subject line",
    "body": "email body text"
}}
"""

    def _parse_email_content(self, content: str) -> Tuple[str, str]:
        """Parse LLM response to extract subject and body."""
        try:
            # Strip markdown code blocks if present
            cleaned = content.strip()
            if cleaned.startswith("```"):
                # Remove opening code fence (with optional language tag)
                lines = cleaned.split('\n')
                # Skip first line (```json or ```)
                lines = lines[1:]
                # Remove closing ``` if present
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = '\n'.join(lines)

            # Try to parse as JSON
            data = json.loads(cleaned)
            return data.get("subject", "No Subject"), data.get("body", "No Content")
        except:
            # Fallback parsing
            lines = content.strip().split('\n')
            if len(lines) >= 2:
                return lines[0], '\n'.join(lines[1:])
            return "No Subject", content

    def _fallback_generation(
        self,
        cpra_request: dict,
        sender: dict,
        challenge_type: Optional[str]
    ) -> Tuple[str, str]:
        """Fallback template-based generation."""
        title = cpra_request.get('title', 'Request')

        if challenge_type == "ambiguous_terms":
            subject = f"Taking the lead on {title}"
            body = f"I'll be leading this initiative. As the lead coordinator, we need to address the issues raised."
        elif challenge_type == "near_miss":
            subject = f"General update on operations"
            body = f"Various operational matters to discuss, including some facility concerns."
        else:
            subject = f"Re: {title}"
            body = f"Following up on the {title} matter. Please see attached for details."

        body += f"\n\nBest regards,\n{sender.get('first_name', 'Staff Member')}"
        return subject, body

    def _fallback_routine_generation(self, sender: dict, topic: str) -> Tuple[str, str]:
        """Fallback routine email generation."""
        subject = f"Update: {topic}"
        body = f"Team,\n\nJust a quick update on {topic}. Please let me know if you have any questions.\n\nThanks,\n{sender.get('first_name', 'Staff')}"
        return subject, body

    # ============================================================
    # NEW METHODS FOR HARDER CORPUS GENERATION
    # ============================================================

    def generate_keyword_free_email(
        self,
        cpra_request: dict,
        sender: dict,
        recipients: list,
        max_retries: int = 3
    ) -> Tuple[str, str, bool]:
        """Generate an email that is responsive but contains NO keywords.

        Returns:
            Tuple of (subject, body, success). success=False if keywords slipped through.
        """
        forbidden_keywords = (
            cpra_request.get('primary_keywords', []) +
            cpra_request.get('secondary_keywords', [])
        )

        prompt = self._build_keyword_free_prompt(cpra_request, sender, recipients, forbidden_keywords)

        if not self.client:
            # Cannot generate keyword-free without LLM
            return "Fallback subject", "Fallback body", False

        for attempt in range(max_retries):
            try:
                if self.config.provider == "openai" and HAS_OPENAI:
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": "You are generating realistic emails for a school district. You must follow keyword restrictions exactly."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    content = response.choices[0].message.content

                elif self.config.provider == "anthropic" and HAS_ANTHROPIC:
                    response = self.client.messages.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text

                subject, body = self._parse_email_content(content)

                # Validate no keywords appear
                if self._validate_no_keywords(subject, body, forbidden_keywords):
                    return subject, body, True
                else:
                    print(f"Attempt {attempt + 1}: Keywords found in output, retrying...")

            except Exception as e:
                print(f"LLM generation failed on attempt {attempt + 1}: {e}")

        return "Fallback subject", "Fallback body", False

    def generate_euphemism_email(
        self,
        cpra_request: dict,
        sender: dict,
        recipients: list,
        euphemism_patterns: List[str]
    ) -> Tuple[str, str]:
        """Generate an email using euphemisms instead of direct keywords."""

        prompt = self._build_euphemism_prompt(cpra_request, sender, recipients, euphemism_patterns)

        if not self.client:
            return self._fallback_euphemism_generation(cpra_request, sender, euphemism_patterns)

        try:
            if self.config.provider == "openai" and HAS_OPENAI:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are generating realistic emails for a school district staff avoiding direct language about sensitive topics."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                content = response.choices[0].message.content

            elif self.config.provider == "anthropic" and HAS_ANTHROPIC:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

            return self._parse_email_content(content)

        except Exception as e:
            print(f"LLM generation failed: {e}, falling back to templates")
            return self._fallback_euphemism_generation(cpra_request, sender, euphemism_patterns)

    def generate_email_thread(
        self,
        cpra_request: dict,
        participants: List[dict],
        thread_length: int = 3
    ) -> List[Dict]:
        """Generate an email thread with responsive content buried in earlier messages.

        Returns:
            List of email dicts with 'subject', 'body', 'sender', 'is_responsive' for each email in thread.
        """
        prompt = self._build_thread_prompt(cpra_request, participants, thread_length)

        if not self.client:
            return self._fallback_thread_generation(cpra_request, participants, thread_length)

        try:
            if self.config.provider == "openai" and HAS_OPENAI:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are generating realistic email threads for a school district."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens * 2  # Threads need more tokens
                )
                content = response.choices[0].message.content

            elif self.config.provider == "anthropic" and HAS_ANTHROPIC:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens * 2,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

            return self._parse_thread_content(content, participants)

        except Exception as e:
            print(f"LLM thread generation failed: {e}, falling back to templates")
            return self._fallback_thread_generation(cpra_request, participants, thread_length)

    def _build_keyword_free_prompt(
        self,
        cpra_request: dict,
        sender: dict,
        recipients: list,
        forbidden_keywords: List[str]
    ) -> str:
        """Build prompt for keyword-free email generation."""
        return f"""Generate a realistic internal email that discusses the topic below WITHOUT using ANY of the forbidden words.

TOPIC: {cpra_request.get('title', '')}
CONTEXT: {cpra_request.get('description', '')}

CRITICAL - FORBIDDEN WORDS (do NOT use ANY of these):
{', '.join(forbidden_keywords)}

Instead of the forbidden words, use:
- Descriptive phrases ("the samples from the drinking fountains" instead of "water quality")
- References to previous conversations ("what we discussed", "the results from last month")
- Euphemisms and indirect language ("the situation at the elementary school")
- Pronouns and context ("it", "this issue", "that matter")

The email should be clearly about {cpra_request.get('title', 'the topic')} to a human reader, but impossible to find with a keyword search.

Sender: {sender.get('full_name', '')} ({sender.get('role', '')})
Recipients: {', '.join([r.get('full_name', '') for r in recipients])}

Format your response as JSON:
{{
    "subject": "email subject line (also must avoid forbidden words)",
    "body": "email body text"
}}
"""

    def _build_euphemism_prompt(
        self,
        cpra_request: dict,
        sender: dict,
        recipients: list,
        euphemism_patterns: List[str]
    ) -> str:
        """Build prompt for euphemism-based email generation."""
        return f"""Generate a realistic internal email discussing {cpra_request.get('title', '')} using indirect language and euphemisms.

Staff often avoid direct language about sensitive topics due to liability concerns, parent communication worries, or just workplace norms.

USE THESE EUPHEMISM PATTERNS:
{chr(10).join('- ' + p for p in euphemism_patterns)}

Example euphemisms:
- "the situation at Jefferson" instead of "lead contamination at Jefferson Elementary"
- "what the inspector found" instead of "failed water quality test"
- "those concerns parents raised" instead of "complaints about water safety"

Sender: {sender.get('full_name', '')} ({sender.get('role', '')})
Recipients: {', '.join([r.get('full_name', '') for r in recipients])}

Format your response as JSON:
{{
    "subject": "email subject line",
    "body": "email body text using euphemisms"
}}
"""

    def _build_thread_prompt(
        self,
        cpra_request: dict,
        participants: List[dict],
        thread_length: int
    ) -> str:
        """Build prompt for email thread generation."""
        participant_names = [p.get('full_name', f"Person {i}") for i, p in enumerate(participants)]

        return f"""Generate an email thread of {thread_length} messages about {cpra_request.get('title', 'a topic')}.

IMPORTANT STRUCTURE:
1. The FIRST (oldest) email should contain the actual substantive content about {cpra_request.get('title', '')}
2. Later emails should be about logistics, scheduling, or acknowledgments
3. The LAST (most recent) email should have a benign, generic subject that hides the topic

This simulates how responsive content gets "buried" in email threads - the surface email looks routine but the thread contains important information.

Participants: {', '.join(participant_names)}

Format your response as JSON:
{{
    "emails": [
        {{
            "sender_index": 0,
            "subject": "subject for email 1 (original, contains topic)",
            "body": "email body with actual responsive content about {cpra_request.get('title', '')}",
            "is_responsive": true
        }},
        {{
            "sender_index": 1,
            "subject": "Re: [previous subject]",
            "body": "acknowledgment or scheduling reply",
            "is_responsive": false
        }},
        {{
            "sender_index": 0,
            "subject": "Re: Quick follow-up",
            "body": "generic logistics message that hides the real topic",
            "is_responsive": false
        }}
    ]
}}
"""

    def _validate_no_keywords(self, subject: str, body: str, keywords: List[str]) -> bool:
        """Check that no forbidden keywords appear in the email."""
        text = f"{subject} {body}".lower()
        for keyword in keywords:
            # Use word boundary matching
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text):
                return False
        return True

    def _parse_thread_content(self, content: str, participants: List[dict]) -> List[Dict]:
        """Parse LLM response to extract email thread."""
        try:
            # Strip markdown code blocks if present
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split('\n')
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = '\n'.join(lines)

            data = json.loads(cleaned)
            emails = data.get("emails", [])

            result = []
            for email in emails:
                sender_idx = email.get("sender_index", 0) % len(participants)
                result.append({
                    "subject": email.get("subject", "No Subject"),
                    "body": email.get("body", ""),
                    "sender": participants[sender_idx],
                    "is_responsive": email.get("is_responsive", False)
                })
            return result

        except Exception as e:
            print(f"Failed to parse thread content: {e}")
            return self._fallback_thread_generation({}, participants, 3)

    def _fallback_euphemism_generation(
        self,
        cpra_request: dict,
        sender: dict,
        euphemism_patterns: List[str]
    ) -> Tuple[str, str]:
        """Fallback euphemism email generation."""
        pattern = euphemism_patterns[0] if euphemism_patterns else "the situation"
        subject = f"Follow-up on {pattern}"
        body = f"Hi,\n\nI wanted to touch base about {pattern} we discussed. Let me know when you have a moment to talk through this.\n\nThanks,\n{sender.get('first_name', 'Staff')}"
        return subject, body

    def _fallback_thread_generation(
        self,
        cpra_request: dict,
        participants: List[dict],
        thread_length: int
    ) -> List[Dict]:
        """Fallback thread generation."""
        title = cpra_request.get('title', 'the matter')
        emails = []

        # First email - responsive
        emails.append({
            "subject": f"Update on {title}",
            "body": f"Team,\n\nHere's the latest on {title}. We need to discuss next steps.\n\nRegards,\n{participants[0].get('first_name', 'Staff') if participants else 'Staff'}",
            "sender": participants[0] if participants else {},
            "is_responsive": True
        })

        # Middle emails - non-responsive
        for i in range(1, thread_length - 1):
            sender = participants[i % len(participants)] if participants else {}
            emails.append({
                "subject": f"Re: Update on {title}",
                "body": f"Thanks for the update. Let me know if you need anything from my end.\n\n{sender.get('first_name', 'Staff')}",
                "sender": sender,
                "is_responsive": False
            })

        # Last email - benign surface
        if thread_length > 1:
            sender = participants[(thread_length - 1) % len(participants)] if participants else {}
            emails.append({
                "subject": "Re: Quick question",
                "body": f"Can we meet tomorrow to discuss? I'm free after 2pm.\n\n{sender.get('first_name', 'Staff')}",
                "sender": sender,
                "is_responsive": False
            })

        return emails