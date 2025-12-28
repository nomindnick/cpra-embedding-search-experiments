"""Generator for CPRA requests with built-in challenge patterns."""

from datetime import datetime, timedelta
from typing import List

from src.models.cpra import (
    CPRARequest, CPRARequestSet, RequestComplexity, ChallengeType
)


class CPRARequestGenerator:
    """Generates CPRA requests for testing purposes."""

    def generate_requests(self) -> CPRARequestSet:
        """Generate the standard set of CPRA requests for testing."""
        requests = []

        # Request 1: Lead Testing - High ambiguity potential (HARDEST - 60% keyword-free)
        requests.append(CPRARequest(
            id="cpra_001",
            title="Lead Testing in Water Systems",
            description="Request for communications about lead testing in school water systems",
            request_text=(
                "Please provide all emails, memos, and communications regarding lead testing, "
                "water quality testing, or drinking water safety in school facilities from "
                "January 1, 2023 to present. This includes communications about test results, "
                "remediation efforts, and communications with parents or regulatory agencies."
            ),
            date_submitted=datetime(2024, 10, 1),
            date_range_start=datetime(2023, 1, 1),
            date_range_end=datetime(2024, 10, 1),
            primary_keywords=["lead testing", "water quality", "drinking water", "lead levels"],
            secondary_keywords=["water safety", "contamination", "EPA", "testing results", "water fountains"],
            exclude_keywords=["lead teacher", "leadership", "leading"],
            concepts=[
                "water quality testing",
                "environmental safety",
                "lead contamination",
                "student health",
                "regulatory compliance"
            ],
            complexity=RequestComplexity.COMPLEX,
            challenge_types=[ChallengeType.AMBIGUOUS_TERMS, ChallengeType.NEAR_MISS],
            responsive_email_patterns=[
                "discussing lead test results",
                "water fountain testing schedule",
                "EPA water quality standards",
                "lead remediation in buildings"
            ],
            non_responsive_patterns=[
                "lead teacher assignments",
                "taking the lead on projects",
                "leadership team meetings"
            ],
            department_targets=["Facilities", "Safety"],
            # Difficulty settings - 60% keyword-free (hardest request)
            keyword_free_rate=0.60,
            euphemism_patterns=[
                "the situation at [school]",
                "what the inspector found",
                "the samples we discussed",
                "those results from the lab",
                "the issue with building C",
                "what we found in the fountains"
            ],
            abbreviation_patterns={
                "lead": "Pb",
                "water": "H2O",
                "parts per billion": "ppb"
            }
        ))

        # Request 2: Budget Allocation - Moderate complexity (40% keyword-free)
        requests.append(CPRARequest(
            id="cpra_002",
            title="COVID Relief Fund Allocation",
            description="Request for communications about federal COVID relief fund allocation",
            request_text=(
                "All communications, including emails and attachments, related to the allocation, "
                "distribution, or use of federal COVID-19 relief funds (ESSER, CARES Act, ARP) "
                "received by the district. Include discussions of spending priorities, vendor "
                "selections, and board presentations from March 2020 to present."
            ),
            date_submitted=datetime(2024, 10, 1),
            date_range_start=datetime(2020, 3, 1),
            date_range_end=datetime(2024, 10, 1),
            primary_keywords=["ESSER", "CARES Act", "ARP funds", "COVID relief", "federal funds"],
            secondary_keywords=["budget allocation", "pandemic funding", "spending plan", "relief funds"],
            exclude_keywords=["personal relief", "stress relief"],
            concepts=[
                "federal funding",
                "pandemic response",
                "budget planning",
                "educational technology",
                "facility improvements"
            ],
            complexity=RequestComplexity.MODERATE,
            challenge_types=[ChallengeType.INDIRECT_REFERENCE],
            responsive_email_patterns=[
                "ESSER fund allocation",
                "COVID relief spending priorities",
                "federal grant applications",
                "pandemic fund distribution"
            ],
            non_responsive_patterns=[
                "general budget discussions",
                "non-COVID grants",
                "regular operational funds"
            ],
            department_targets=["Finance", "District Office"],
            # Difficulty settings - 40% keyword-free
            keyword_free_rate=0.40,
            euphemism_patterns=[
                "those federal dollars",
                "the pandemic money",
                "what we got from Washington",
                "the special allocation",
                "that funding we discussed"
            ],
            abbreviation_patterns={}
        ))

        # Request 3: Special Education Changes (30% keyword-free)
        requests.append(CPRARequest(
            id="cpra_003",
            title="Special Education Program Changes",
            description="Request for communications about special education program modifications",
            request_text=(
                "Please provide all emails and documents discussing changes, modifications, or "
                "updates to special education programs, IEP processes, or special education "
                "staffing from August 2023 to present. Include communications with parents, "
                "staff discussions, and policy updates."
            ),
            date_submitted=datetime(2024, 10, 1),
            date_range_start=datetime(2023, 8, 1),
            date_range_end=datetime(2024, 10, 1),
            primary_keywords=["special education", "IEP", "SPED", "504 plan"],
            secondary_keywords=["inclusion", "mainstreaming", "accommodations", "modifications"],
            exclude_keywords=["special events", "special recognition"],
            concepts=[
                "special education services",
                "individualized education",
                "disability accommodations",
                "inclusive education",
                "parent communication"
            ],
            complexity=RequestComplexity.MODERATE,
            challenge_types=[ChallengeType.NEAR_MISS, ChallengeType.PARTIAL_MATCH],
            responsive_email_patterns=[
                "IEP meeting schedules",
                "special education staffing changes",
                "SPED program updates",
                "parent concerns about services"
            ],
            non_responsive_patterns=[
                "general education curriculum",
                "special events planning",
                "regular parent communication"
            ],
            department_targets=["Special Education", "Curriculum"],
            # Difficulty settings - 30% keyword-free
            keyword_free_rate=0.30,
            euphemism_patterns=[
                "services for that student",
                "the meeting with the family",
                "those students who need extra help",
                "the program we discussed",
                "the support services"
            ],
            abbreviation_patterns={}
        ))

        # Request 4: Technology Contracts (EASIEST - 20% keyword-free)
        requests.append(CPRARequest(
            id="cpra_004",
            title="EdTech Vendor Contracts",
            description="Request for communications about educational technology vendor selection",
            request_text=(
                "All emails and communications regarding the selection, evaluation, or contracting "
                "with educational technology vendors, including but not limited to learning management "
                "systems, student information systems, and digital curriculum platforms from "
                "January 2024 to present."
            ),
            date_submitted=datetime(2024, 10, 1),
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 10, 1),
            primary_keywords=["vendor", "contract", "EdTech", "procurement", "RFP"],
            secondary_keywords=["software", "platform", "subscription", "licensing", "SaaS"],
            exclude_keywords=["food vendor", "maintenance vendor"],
            concepts=[
                "technology procurement",
                "vendor evaluation",
                "digital learning tools",
                "software contracts",
                "data privacy"
            ],
            complexity=RequestComplexity.SIMPLE,
            challenge_types=[ChallengeType.TEMPORAL_MISMATCH],
            responsive_email_patterns=[
                "EdTech vendor proposals",
                "software evaluation committee",
                "technology contract negotiations",
                "student data privacy concerns"
            ],
            non_responsive_patterns=[
                "facility vendor contracts",
                "food service vendors",
                "transportation contracts"
            ],
            department_targets=["Technology", "Finance"],
            # Difficulty settings - 20% keyword-free (easiest request)
            keyword_free_rate=0.20,
            euphemism_patterns=[
                "the company we're evaluating",
                "that learning tool",
                "the proposal we received"
            ],
            abbreviation_patterns={
                "learning management system": "LMS",
                "student information system": "SIS"
            }
        ))

        # Request 5: Safety Incidents (50% keyword-free - sensitive topic)
        requests.append(CPRARequest(
            id="cpra_005",
            title="Student Safety Incidents",
            description="Request for communications about student safety incidents and responses",
            request_text=(
                "Please provide all emails and reports regarding student safety incidents, "
                "including but not limited to bullying reports, playground injuries requiring "
                "medical attention, and security concerns from the current school year "
                "(August 2024 to present). Exclude routine minor incidents."
            ),
            date_submitted=datetime(2024, 10, 1),
            date_range_start=datetime(2024, 8, 1),
            date_range_end=datetime(2024, 10, 1),
            primary_keywords=["safety incident", "injury report", "bullying", "security"],
            secondary_keywords=["medical attention", "parent notification", "incident response"],
            exclude_keywords=["safety drill", "safety training"],
            concepts=[
                "student safety",
                "incident management",
                "school security",
                "parent communication",
                "risk management"
            ],
            complexity=RequestComplexity.MODERATE,
            challenge_types=[ChallengeType.INDIRECT_REFERENCE, ChallengeType.NEAR_MISS],
            responsive_email_patterns=[
                "injury requiring nurse visit",
                "bullying investigation",
                "security concern reported",
                "parent complaint about safety"
            ],
            non_responsive_patterns=[
                "routine safety drills",
                "general safety reminders",
                "safety committee meetings"
            ],
            department_targets=["Safety", "Principal"],
            # Difficulty settings - 50% keyword-free (sensitive topics use euphemisms)
            keyword_free_rate=0.50,
            euphemism_patterns=[
                "what happened at recess",
                "the situation with those students",
                "the incident we discussed",
                "what we need to document",
                "the matter at [school]",
                "the issue between those kids"
            ],
            abbreviation_patterns={}
        ))

        return CPRARequestSet(
            requests=requests,
            metadata={
                "generated_date": datetime.now().isoformat(),
                "total_requests": len(requests),
                "complexity_distribution": {
                    "simple": sum(1 for r in requests if r.complexity == RequestComplexity.SIMPLE),
                    "moderate": sum(1 for r in requests if r.complexity == RequestComplexity.MODERATE),
                    "complex": sum(1 for r in requests if r.complexity == RequestComplexity.COMPLEX)
                }
            }
        )