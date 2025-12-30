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
        "You are evaluating whether a document must be disclosed under the "
        "California Public Records Act (CPRA). Be precise and follow instructions exactly."
    ),
    prompt_template="""You are evaluating whether a document must be disclosed under the California Public Records Act (CPRA).

A document is RESPONSIVE if it contains information reasonably related to what the requester asked for. It does not need to match the request exactly—if the document's content falls within the scope of the request, it is responsive.

A document is NON-RESPONSIVE if it contains no information related to the request.

Do not consider exemptions or privileges—only whether the document relates to the request.

REQUEST:
{request_text}

DOCUMENT:
{document_text}

Is this document RESPONSIVE to the request? Answer YES or NO.""",
    evaluator_fn="evaluate_binary_classification",
)

# Classification task: Binary with few-shot examples
CLASSIFICATION_FEW_SHOT = Task(
    task_type=TaskType.CLASSIFICATION_BINARY,
    name="Few-Shot Binary Classification",
    description="Classify document as responsive YES or NO using few-shot examples",
    system_prompt=None,  # Self-contained prompt with examples
    prompt_template="""You are classifying documents under the California Public Records Act (CPRA).

A document is RESPONSIVE if it contains information related to what the requester asked for.
A document is NON-RESPONSIVE if it has no information related to the request.

===
EXAMPLE 1

REQUEST: All emails about the Smith construction project.

DOCUMENT: Email from John to Mary, dated 3/15/24. Subject: "Smith project delay." Body: "The contractor said materials won't arrive until next week."

ANSWER: YES
===
EXAMPLE 2

REQUEST: All emails about the Smith construction project.

DOCUMENT: Email from John to Mary, dated 3/18/24. Subject: "Lunch Friday?" Body: "Want to grab lunch at the new Thai place?"

ANSWER: NO
===
EXAMPLE 3

REQUEST: All emails about the Smith construction project.

DOCUMENT: Email from Susan to John, dated 3/20/24. Subject: "Budget meeting." Body: "Reminder about tomorrow's budget meeting. We'll cover Smith project overruns and the new HVAC contract."

ANSWER: YES
===
NOW CLASSIFY THIS DOCUMENT

REQUEST:
{request_text}

DOCUMENT:
{document_text}

ANSWER:""",
    evaluator_fn="evaluate_binary_classification",
)

# Classification task: Multi-shot with 6 examples covering challenge types
CLASSIFICATION_MULTI_SHOT = Task(
    task_type=TaskType.CLASSIFICATION_BINARY,
    name="Multi-Shot Binary Classification",
    description="Classify document as responsive YES or NO using 6 examples covering challenge types",
    system_prompt=None,  # Self-contained prompt with examples
    prompt_template="""You are classifying documents under the California Public Records Act (CPRA).

A document is RESPONSIVE if it contains information related to what the requester asked for.
A document is NON-RESPONSIVE if it has no information related to the request.

Important:
- A document can be RESPONSIVE even if it does not use the exact words from the request. Look for contextual clues and related concepts.
- A document is NON-RESPONSIVE if it merely shares vocabulary or general topic area but is about a different specific subject.
- Words can have multiple meanings. Focus on the meaning relevant to the request.

===
EXAMPLE 1

REQUEST: All records about the Smith construction project.

DOCUMENT: Email from John to Mary, 3/15/24. Subject: "Smith project delay." Body: "The contractor said materials won't arrive until next week."

ANSWER: YES
===
EXAMPLE 2

REQUEST: All records about the Smith construction project.

DOCUMENT: Email from John to Mary, 3/18/24. Subject: "Lunch Friday?" Body: "Want to grab lunch at the new Thai place?"

ANSWER: NO
===
EXAMPLE 3 (indirect reference — no keywords, but contextually related)

REQUEST: All records about the Smith construction project.

DOCUMENT: Email from Susan to Contractor, 3/19/24. Subject: "Oak Street timeline." Body: "The owner is asking when the framing will be done at 445 Oak. Can you send an updated schedule?"

ANSWER: YES
===
EXAMPLE 4 (adjacent topic — similar domain, different subject)

REQUEST: All records about the Smith construction project.

DOCUMENT: Email from Facilities to Board, 3/20/24. Subject: "Capital Projects Update." Body: "The Johnson Elementary roof replacement is on schedule. HVAC upgrade bids are due next week."

ANSWER: NO
===
EXAMPLE 5 (word with multiple meanings — wrong meaning)

REQUEST: All records about the Smith construction project.

DOCUMENT: Email from HR, 3/21/24. Subject: "Smith promoted." Body: "Please congratulate Jane Smith on her promotion to lead project manager for IT services."

ANSWER: NO
===
EXAMPLE 6 (partial relevance — mentioned among other topics)

REQUEST: All records about the Smith construction project.

DOCUMENT: Email from Susan to John, 3/20/24. Subject: "Budget meeting." Body: "Reminder about tomorrow's meeting. We'll cover Smith project overruns and the new HVAC contract."

ANSWER: YES
===
NOW CLASSIFY THIS DOCUMENT

REQUEST:
{request_text}

DOCUMENT:
{document_text}

ANSWER:""",
    evaluator_fn="evaluate_binary_classification",
)

# Classification task: Ternary with confidence
CLASSIFICATION_TERNARY = Task(
    task_type=TaskType.CLASSIFICATION_TERNARY,
    name="Ternary Classification with Confidence",
    description="Classify as yes/no/maybe with confidence score",
    system_prompt=(
        "You are evaluating whether a document must be disclosed under the "
        "California Public Records Act (CPRA). Be precise and follow instructions exactly."
    ),
    prompt_template="""You are evaluating whether a document must be disclosed under the California Public Records Act (CPRA).

A document is RESPONSIVE if it contains information reasonably related to what the requester asked for. It does not need to match the request exactly—if the document's content falls within the scope of the request, it is responsive.

A document is NON-RESPONSIVE if it contains no information related to the request.

Use MAYBE if the document is borderline or you are uncertain.

Do not consider exemptions or privileges—only whether the document relates to the request.

REQUEST:
{request_text}

DOCUMENT:
{document_text}

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
        "You are evaluating whether a document must be disclosed under the "
        "California Public Records Act (CPRA). Always respond with valid JSON only, "
        "no other text before or after the JSON."
    ),
    prompt_template="""You are evaluating whether a document must be disclosed under the California Public Records Act (CPRA).

A document is RESPONSIVE if it contains information reasonably related to what the requester asked for. It does not need to match the request exactly—if the document's content falls within the scope of the request, it is responsive.

A document is NON-RESPONSIVE if it contains no information related to the request.

Do not consider exemptions or privileges—only whether the document relates to the request.

REQUEST:
{request_text}

DOCUMENT:
{document_text}

Respond with a JSON object containing:
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
    system_prompt=None,  # Self-contained prompt with example
    prompt_template="""Extract passages from this document that relate to the public records request. Copy text EXACTLY as written—do not paraphrase or summarize.

===
EXAMPLE

REQUEST: Records about water main breaks on Elm Street.

DOCUMENT: From: Mike. To: Repairs. Date: 3/15/24. Subject: Elm St update. Body: "The 6-inch main at Elm and 3rd failed overnight. Crew arrived 6am, isolated the break by 8am. Approximately 12 customers affected. Unrelated: the Oak Street paving project starts Monday."

EVIDENCE:
- "The 6-inch main at Elm and 3rd failed overnight"
- "Crew arrived 6am, isolated the break by 8am"
- "Approximately 12 customers affected"

===
NOW EXTRACT FROM THIS DOCUMENT

REQUEST: {request_text}

DOCUMENT: {document_text}

If this document has no relevant content, respond with only: NONE

EVIDENCE:""",
    evaluator_fn="evaluate_evidence_extraction",
)

# Paraphrase generation task
PARAPHRASE_GENERATION = Task(
    task_type=TaskType.PARAPHRASE_GENERATION,
    name="Paraphrase Generation",
    description="Generate diverse paraphrases of a request",
    system_prompt=None,  # Self-contained prompt with example
    prompt_template="""Rewrite this public records request 5 different ways. Each version should ask for the same records but use different words.

===
EXAMPLE

ORIGINAL: All emails about the Smith construction project.

PARAPHRASES:
1. Correspondence regarding the Smith building project
2. Email communications related to Smith construction work
3. Messages discussing the construction project for Smith
4. Electronic mail concerning Smith's construction activities
5. Any emails mentioning the Smith construction job

===
NOW PARAPHRASE THIS REQUEST

ORIGINAL: {request_text}

PARAPHRASES:""",
    evaluator_fn="evaluate_paraphrase_generation",
)

# Example generation task (positive examples)
EXAMPLE_GENERATION_POSITIVE = Task(
    task_type=TaskType.EXAMPLE_GENERATION,
    name="Positive Example Generation",
    description="Generate realistic responsive document examples",
    system_prompt=None,  # Self-contained prompt with example
    prompt_template="""Write a realistic work email that would be responsive to this public records request. Do not use obvious keywords—show relevance through context.

===
EXAMPLE

REQUEST: All records about lead contamination in the water system.

EMAIL:
From: Sarah Chen <schen@citywater.gov>
To: Operations Team <ops@citywater.gov>
Date: March 15, 2024
Subject: Pre-1950 service line replacements - Q2 schedule

Team,

We need to prioritize the Oakwood neighborhood replacements this quarter. Most homes there were built in the 1940s and still have original plumbing. Three residents have already requested testing after the news coverage last month.

I've attached the prioritization matrix based on construction date and complaint history. Let's discuss at Thursday's meeting.

Sarah

===
NOW GENERATE AN EMAIL

REQUEST: {request_text}

EMAIL:""",
    evaluator_fn="evaluate_example_generation",
)

# Example generation task (negative examples)
EXAMPLE_GENERATION_NEGATIVE = Task(
    task_type=TaskType.EXAMPLE_GENERATION,
    name="Negative Example Generation",
    description="Generate realistic non-responsive document examples",
    system_prompt=None,  # Self-contained prompt with example
    prompt_template="""Write a realistic work email that is NOT responsive to this public records request, but might be confused for responsive because it shares some vocabulary or general topic.

===
EXAMPLE

REQUEST: All records about lead contamination in the water system.

EMAIL:
From: James Wu <jwu@citywater.gov>
To: HR Department <hr@citywater.gov>
Date: March 18, 2024
Subject: Lead Engineer Position - Interview Panel

Hi HR,

I'd like to recommend Maria Torres and Kevin Park for the interview panel for our open Lead Engineer position. Both have experience evaluating technical candidates.

Can we target next Wednesday for first-round interviews? The water infrastructure team is eager to fill this role before the summer project season.

Thanks,
James

===
NOW GENERATE AN EMAIL

REQUEST: {request_text}

EMAIL:""",
    evaluator_fn="evaluate_example_generation",
)

# Search term extraction task (from request, not document)
KEYWORD_EXTRACTION = Task(
    task_type=TaskType.KEYWORD_EXTRACTION,
    name="Search Term Extraction",
    description="Extract search terms from a public records request",
    system_prompt=None,  # Self-contained prompt with examples
    prompt_template="""Extract search terms from this public records request that could help find relevant documents.

===
EXAMPLE

REQUEST: All emails from 2023 about the Smith Elementary roof replacement project, including communications with ABC Roofing contractors.

SEARCH TERMS:
- Smith Elementary (school name)
- roof replacement (project type)
- ABC Roofing (contractor)
- 2023 (time period)
- roofing, contractor, construction, repair (related words)

===
EXAMPLE

REQUEST: Records of complaints about water quality in the Oakwood neighborhood.

SEARCH TERMS:
- Oakwood (neighborhood)
- water quality (subject)
- complaint, concern, issue, problem (related words)
- contamination, testing, results, sample (related words)

===
NOW EXTRACT SEARCH TERMS

REQUEST: {request_text}

SEARCH TERMS:""",
    evaluator_fn="evaluate_keyword_extraction",
)


# All tasks for evaluation
TASKS = {
    "classification_binary": CLASSIFICATION_BINARY,
    "classification_few_shot": CLASSIFICATION_FEW_SHOT,
    "classification_multi_shot": CLASSIFICATION_MULTI_SHOT,
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
        "classification_few_shot",
        "classification_multi_shot",
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
