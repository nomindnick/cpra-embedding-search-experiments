"""Task definitions and prompts for LLM evaluation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskType(Enum):
    """Types of tasks to evaluate."""

    CLASSIFICATION_BINARY = "classification_binary"
    CLASSIFICATION_TERNARY = "classification_ternary"
    JSON_OUTPUT = "json_output"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    PARAPHRASE_GENERATION = "paraphrase_generation"
    EXAMPLE_GENERATION = "example_generation"
    KEYWORD_EXTRACTION = "keyword_extraction"


@dataclass
class Task:
    """A task for LLM evaluation."""

    task_type: TaskType
    name: str
    description: str
    prompt_template: str
    system_prompt: str | None = None
    evaluator_fn: str | None = None  # Name of function to evaluate response

    def format_prompt(self, **kwargs: Any) -> str:
        """Format the prompt template with given arguments."""
        return self.prompt_template.format(**kwargs)


# Classification task: Binary (YES/NO)
CLASSIFICATION_BINARY = Task(
    task_type=TaskType.CLASSIFICATION_BINARY,
    name="Binary Classification",
    description="Classify document as responsive YES or NO",
    system_prompt=(
        "You are a legal document reviewer evaluating documents for a California "
        "Public Records Act (CPRA) request. Your task is to determine if documents "
        "are responsive (should be disclosed) or not. Be precise and follow "
        "instructions exactly."
    ),
    prompt_template="""CPRA REQUEST:
{request_text}

DOCUMENT TO EVALUATE:
{document_text}

Is this document RESPONSIVE to the CPRA request?
A document is responsive if it contains information that would need to be disclosed.

Answer with exactly one word: YES or NO""",
    evaluator_fn="evaluate_binary_classification",
)

# Classification task: Ternary with confidence
CLASSIFICATION_TERNARY = Task(
    task_type=TaskType.CLASSIFICATION_TERNARY,
    name="Ternary Classification with Confidence",
    description="Classify as yes/no/maybe with confidence score",
    system_prompt=(
        "You are a legal document reviewer evaluating documents for a California "
        "Public Records Act (CPRA) request. Your task is to determine if documents "
        "are responsive (should be disclosed) or not. Be precise and follow "
        "instructions exactly."
    ),
    prompt_template="""CPRA REQUEST:
{request_text}

DOCUMENT TO EVALUATE:
{document_text}

Determine if this document is RESPONSIVE to the CPRA request.
A document is responsive if it contains information that would need to be disclosed.

Respond with exactly two lines:
Line 1: Your classification (exactly one of: YES, NO, or MAYBE)
Line 2: Your confidence as a number from 0 to 100

Example response:
YES
85""",
    evaluator_fn="evaluate_ternary_classification",
)

# JSON output task
JSON_OUTPUT = Task(
    task_type=TaskType.JSON_OUTPUT,
    name="JSON Structured Output",
    description="Output classification in valid JSON format",
    system_prompt=(
        "You are a legal document reviewer. Always respond with valid JSON only, "
        "no other text before or after the JSON."
    ),
    prompt_template="""CPRA REQUEST:
{request_text}

DOCUMENT TO EVALUATE:
{document_text}

Analyze this document and respond with a JSON object containing:
- "responsive": "yes", "no", or "maybe"
- "confidence": number from 0 to 100
- "category": one of "direct_match", "indirect_reference", "keyword_false_positive", "adjacent_topic", "unrelated"
- "reasoning": brief explanation (1-2 sentences)

Respond with only valid JSON, no other text.""",
    evaluator_fn="evaluate_json_output",
)

# Evidence extraction task
EVIDENCE_EXTRACTION = Task(
    task_type=TaskType.EVIDENCE_EXTRACTION,
    name="Evidence Extraction",
    description="Extract verbatim quotes supporting responsiveness",
    system_prompt=(
        "You are a legal document reviewer. Your task is to identify specific "
        "evidence in documents that supports their relevance to a records request. "
        "Always quote exactly from the source document."
    ),
    prompt_template="""CPRA REQUEST:
{request_text}

DOCUMENT TO EVALUATE:
{document_text}

If this document contains information relevant to the CPRA request, extract 1-3
verbatim quotes from the document that demonstrate relevance. Each quote must
appear exactly in the document above.

If the document is not relevant, respond with: NO RELEVANT CONTENT

Format your response as:
QUOTE 1: "exact quote from document"
QUOTE 2: "exact quote from document"
(etc.)""",
    evaluator_fn="evaluate_evidence_extraction",
)

# Paraphrase generation task
PARAPHRASE_GENERATION = Task(
    task_type=TaskType.PARAPHRASE_GENERATION,
    name="Paraphrase Generation",
    description="Generate diverse paraphrases of a request",
    system_prompt=(
        "You are an expert at reformulating search queries and document requests. "
        "Generate diverse paraphrases that capture the same information need."
    ),
    prompt_template="""Original CPRA request:
{request_text}

Generate 5 semantically different paraphrases of this request. Each paraphrase should:
- Capture the same information need
- Use different vocabulary and phrasing
- Be a complete, standalone request

Output exactly 5 paraphrases, numbered 1-5, one per line.""",
    evaluator_fn="evaluate_paraphrase_generation",
)

# Example generation task (positive examples)
EXAMPLE_GENERATION_POSITIVE = Task(
    task_type=TaskType.EXAMPLE_GENERATION,
    name="Positive Example Generation",
    description="Generate realistic responsive document examples",
    system_prompt=(
        "You are an expert at generating realistic email content for document "
        "retrieval testing. Generate plausible emails that would appear in a "
        "government agency's email system."
    ),
    prompt_template="""CPRA REQUEST:
{request_text}

Generate a realistic email that WOULD be responsive to this CPRA request.

Requirements:
- Include realistic From, To, Subject fields
- The email should clearly contain information relevant to the request
- Make it sound like a real internal government email
- Length: 100-250 words for the body

Format:
From: [email]
To: [email]
Subject: [subject]

[body]""",
    evaluator_fn="evaluate_example_generation",
)

# Example generation task (negative examples)
EXAMPLE_GENERATION_NEGATIVE = Task(
    task_type=TaskType.EXAMPLE_GENERATION,
    name="Negative Example Generation",
    description="Generate realistic non-responsive document examples",
    system_prompt=(
        "You are an expert at generating realistic email content for document "
        "retrieval testing. Generate plausible emails that would appear in a "
        "government agency's email system."
    ),
    prompt_template="""CPRA REQUEST:
{request_text}

Generate a realistic email that is RELATED TO but NOT responsive to this CPRA request.
This should be a plausible false positive - something that might seem relevant but isn't.

Requirements:
- Include realistic From, To, Subject fields
- The email should be about a related topic but NOT actually responsive to the request
- Examples of non-responsive but confusing content:
  - Keywords used in different contexts
  - Adjacent topics in the same domain
  - Administrative content tangentially related
- Length: 100-250 words for the body

Format:
From: [email]
To: [email]
Subject: [subject]

[body]""",
    evaluator_fn="evaluate_example_generation",
)

# Keyword extraction task
KEYWORD_EXTRACTION = Task(
    task_type=TaskType.KEYWORD_EXTRACTION,
    name="Keyword Extraction",
    description="Extract relevant keywords and entities from text",
    system_prompt=(
        "You are an expert at identifying key terms, entities, and concepts "
        "in documents. Extract precise, relevant terms."
    ),
    prompt_template="""CPRA REQUEST:
{request_text}

DOCUMENT:
{document_text}

Extract keywords and entities from this document that are relevant to the CPRA request.

Organize your extraction as:
KEYWORDS: [comma-separated list of important terms]
ENTITIES: [comma-separated list of people, organizations, places, projects]
ACRONYMS: [comma-separated list of abbreviations with their meanings]

Only include terms that actually appear in or are directly implied by the document.""",
    evaluator_fn="evaluate_keyword_extraction",
)


# All tasks for evaluation
TASKS = {
    "classification_binary": CLASSIFICATION_BINARY,
    "classification_ternary": CLASSIFICATION_TERNARY,
    "json_output": JSON_OUTPUT,
    "evidence_extraction": EVIDENCE_EXTRACTION,
    "paraphrase_generation": PARAPHRASE_GENERATION,
    "example_generation_positive": EXAMPLE_GENERATION_POSITIVE,
    "example_generation_negative": EXAMPLE_GENERATION_NEGATIVE,
    "keyword_extraction": KEYWORD_EXTRACTION,
}


# Task categories for reporting
TASK_CATEGORIES = {
    "classification": [
        "classification_binary",
        "classification_ternary",
        "json_output",
    ],
    "extraction": [
        "evidence_extraction",
        "keyword_extraction",
    ],
    "generation": [
        "paraphrase_generation",
        "example_generation_positive",
        "example_generation_negative",
    ],
}
