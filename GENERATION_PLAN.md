# CPRA Embedding Search: Test Corpus Generation Plan

This document guides the generation of a new synthetic test corpus for evaluating embedding-based semantic search against keyword search for California Public Records Act (CPRA) document discovery.

---

## 1. Project Overview

### The Problem
Public agencies responding to CPRA requests must identify all responsive documents from large email archives. Traditional keyword search has a fundamental problem: **ambiguous terms create massive false positive rates while missing relevant documents that don't use exact keywords.**

Example: A CPRA request for documents about "lead contamination" in the water supply:
- **False positives**: Thousands of emails about "leadership", "leading the project", "take the lead"
- **False negatives**: Emails discussing "the Flint situation", "elevated metal levels", "infrastructure concerns" that never use the word "lead"

### The Goal
Determine whether embedding-based semantic search can achieve:
- **≥94% recall** (legal/ethical requirement - missing responsive documents is unacceptable)
- **Significantly improved precision** over keyword search (reducing manual review burden)

### Why This Matters
Public agencies spend enormous resources on CPRA compliance. A better search approach could:
- Reduce attorney review time by orders of magnitude
- Decrease risk of missing responsive documents (legal liability)
- Make records requests more manageable for understaffed agencies

---

## 2. Background: Previous Approach and Lessons Learned

### What We Tried
We built a test corpus generator (`cpra-golden-emails/`) that:
1. Defined 5 different CPRA requests
2. Used Claude API calls to generate synthetic emails
3. Categorized emails by challenge type (AMBIGUOUS_TERMS, NEAR_MISS, etc.)
4. Tested 13 different embedding models

### What We Learned

**Model Performance:**
- Best model: Snowflake Arctic Embed L v2.0 (95.20% recall, 86.02% precision)
- Embedding search significantly outperformed keyword search on most challenge types
- Larger models generally performed better, but diminishing returns above ~500M parameters

**Corpus Quality Issues:**
- API-generated emails lacked full experimental context
- Some emails didn't properly represent their challenge type
- Difficult to verify quality at scale with API generation
- Five different CPRA requests added complexity without proportional insight

### Why We're Starting Fresh
The core insight: **when Claude Code generates each email with full project context, it understands exactly what we're testing and why each email needs specific characteristics.** API calls only see a prompt, not the experimental design.

This approach trades generation speed for corpus quality - a worthwhile tradeoff for a test dataset we'll use repeatedly.

---

## 3. Goals for New Corpus

### Primary Goals
1. **High-quality challenge representation**: Each email genuinely tests its designated challenge type
2. **Realistic content**: Emails read like actual government communications
3. **Verifiable ground truth**: Clear reasoning for why each email is/isn't responsive
4. **Controlled complexity**: One primary CPRA request, deeply tested

### Secondary Goals
1. **Validation generalization**: Small secondary request to check we're not overfitting
2. **Thread realism**: Include email threads where relevant content may be buried
3. **Variety within categories**: Multiple approaches to each challenge type

### Non-Goals
- Massive scale (quality over quantity)
- Perfect realism (synthetic is fine if it tests the right things)
- Covering every possible CPRA request type

---

## 4. CPRA Requests

### 4.1 Primary Request: Lead Contamination

**Request Text:**
> "Pursuant to the California Public Records Act (Government Code Section 6250 et seq.), I request copies of all documents, emails, and communications related to lead contamination, lead testing, lead levels, or lead remediation in the city's water supply system from January 1, 2020 to present. This includes but is not limited to: test results, internal communications, communications with state or federal agencies, contractor communications, public notices, and budget documents related to lead issues."

**Why This Request:**
- "Lead" is maximally ambiguous (metal vs. leadership/leading)
- Water contamination is a real, common CPRA topic
- Rich in indirect references (Flint, infrastructure, EPA, public health)
- Natural temporal aspects (historical events, remediation timelines)
- Technical jargon available (PPB, action levels, LSL, corrosion control)

### 4.2 Validation Request: PFAS Contamination

**Request Text:**
> "Pursuant to the California Public Records Act, I request all documents, emails, and communications related to PFAS, PFOA, PFOS, or 'forever chemicals' contamination in the city's water supply or groundwater from January 1, 2020 to present."

**Why This Request:**
- Same domain (water contamination) for consistency
- Different terminology challenges (acronyms vs. common words)
- Currently relevant topic in environmental law
- Tests whether our findings generalize beyond "lead" ambiguity

**Note:** The validation corpus will be smaller (~50-75 emails) and generated after the primary corpus is complete and initial experiments run.

---

## 5. Corpus Structure

### 5.1 Directory Structure
```
corpus/
├── primary/
│   ├── request.json           # The CPRA request definition
│   ├── emails.json            # All emails with content and metadata
│   └── ground_truth.json      # Responsiveness labels and annotations
├── validation/
│   ├── request.json
│   ├── emails.json
│   └── ground_truth.json
└── README.md                  # Corpus documentation
```

### 5.2 Email Schema

Each email document:
```json
{
  "id": "email_001",
  "thread_id": "thread_001",      // null if standalone
  "thread_position": 1,           // position in thread (1-indexed)
  "thread_length": 3,             // total messages in thread
  "date": "2023-06-15T14:32:00",
  "from": "john.smith@citywater.gov",
  "to": ["jane.doe@citywater.gov"],
  "cc": ["water-team@citywater.gov"],
  "subject": "Re: Q2 Infrastructure Update",
  "body": "Email body text...",
  "has_attachment": false,
  "attachment_names": []
}
```

### 5.3 Ground Truth Schema

```json
{
  "email_001": {
    "responsive": true,
    "challenge_type": "INDIRECT_REFERENCE",
    "buried_in_thread": false,
    "reasoning": "Discusses water infrastructure concerns and references 'the situation in Michigan' without using 'lead' - tests whether embedding captures topical relevance without keywords.",
    "keywords_present": ["water", "infrastructure", "Michigan"],
    "keywords_absent": ["lead", "contamination", "testing"]
  }
}
```

### 5.4 Thread Handling

Threads are stored as separate email documents sharing a `thread_id`. For search purposes:
- The entire thread is concatenated into one searchable document
- Individual message metadata preserved for analysis
- `thread_position` indicates where in thread (relevant for "buried" analysis)

**Example thread structure:**
```
Thread 001:
  - email_001 (position 1): Original message about budget meeting
  - email_002 (position 2): Reply mentioning infrastructure concerns including lead pipes
  - email_003 (position 3): Reply about scheduling follow-up
```
The thread is responsive because of email_002, but the relevant content is "buried" in the middle.

---

## 6. Challenge Types

### 6.1 Responsive Email Categories

#### DIRECT_MATCH (Baseline)
**Definition:** Explicitly discusses lead contamination using clear, direct language.
**Purpose:** Establishes baseline - both keyword and embedding should find these.
**Example signals:** "lead levels", "lead contamination", "lead testing results"
**Target count:** 25-30 emails

#### AMBIGUOUS_TERMS
**Definition:** Uses "lead" in contexts where surrounding text clarifies it's about the metal, but keyword search would also match leadership uses.
**Purpose:** Tests whether embeddings can disambiguate based on context.
**Example:** "The lead levels in samples from the north district..." (clearly metal, but "lead" keyword fires)
**Note:** We also need non-responsive emails using "lead" for leadership to test false positive rejection.
**Target count:** 25-30 emails

#### INDIRECT_REFERENCE
**Definition:** Discusses lead contamination topics without using "lead" - references events, health effects, regulations, or related concepts.
**Purpose:** Tests semantic understanding beyond keyword matching.
**Example signals:** "the Flint crisis", "elevated metal levels", "infrastructure from the 1950s", "EPA action level exceedances"
**Target count:** 30-35 emails (critical category)

#### TECHNICAL_JARGON
**Definition:** Uses technical terminology that experts would recognize as lead-related but doesn't use common keywords.
**Purpose:** Tests whether embeddings capture domain-specific language.
**Example signals:** "LSL replacement program", "corrosion control optimization", "15 ppb threshold", "LCR compliance"
**Target count:** 20-25 emails

#### TEMPORAL_REFERENCE
**Definition:** Discusses historical lead events or future remediation plans with time as a key element.
**Purpose:** Tests handling of temporal context and historical references.
**Example:** "Similar to what we dealt with in 2019", "the remediation timeline extends through 2025"
**Target count:** 20-25 emails

#### BURIED_IN_THREAD
**Definition:** Responsive content exists but is surrounded by unrelated thread messages.
**Purpose:** Tests whether search can find relevant content in noisy contexts.
**Implementation:** Apply to other categories - e.g., INDIRECT_REFERENCE buried in thread
**Target count:** 30-40 emails (across multiple underlying types)

### 6.2 Non-Responsive Email Categories

#### KEYWORD_FALSE_POSITIVE
**Definition:** Uses "lead" but clearly about leadership, leading projects, or other non-metal meanings.
**Purpose:** Tests false positive rejection - critical for precision.
**Example:** "Sarah will lead the committee", "taking the lead on this initiative"
**Target count:** 50-60 emails (important for precision testing)

#### ADJACENT_TOPIC
**Definition:** Related to water/infrastructure but not about lead specifically.
**Purpose:** Tests specificity - should not match just because it's about water.
**Example:** Copper pipe replacement, general water quality, other contaminants (arsenic, bacteria)
**Target count:** 40-50 emails

#### TRUE_NEGATIVE
**Definition:** Clearly unrelated to water, contamination, or infrastructure.
**Purpose:** Baseline negative - neither approach should match these.
**Example:** HR matters, IT updates, general budget discussions, meeting scheduling
**Target count:** 50-60 emails

---

## 7. Generation Guidelines

### 7.1 Email Quality Standards

**Realism Requirements:**
- Use realistic government email conventions (formal but not stiff)
- Include typical email artifacts (signatures, disclaimers, reply quotes)
- Vary writing styles across different "senders"
- Include realistic metadata (plausible dates, department names, titles)

**Content Requirements:**
- Each email should be self-contained and coherent
- Challenge type should be genuinely represented, not forced
- Avoid obviously synthetic patterns (too perfect, too uniform)
- Include natural variations (typos occasionally acceptable, varying lengths)

**What to Avoid:**
- Cookie-cutter templates with word substitutions
- Unrealistic formality or informality
- Perfect grammar in every email (real emails have minor errors)
- Identical email structures

### 7.2 Variety Requirements

Within each challenge type, vary:
- **Sender roles:** Directors, managers, analysts, external contractors, state agencies
- **Email types:** Updates, requests, replies, forwards, meeting notes
- **Tone:** Formal reports, casual updates, urgent alerts
- **Length:** Short acknowledgments to detailed reports
- **Complexity:** Simple single-topic to multi-topic emails

### 7.3 Thread Requirements

For thread-based emails:
- 3-5 messages per thread typically
- Clear thread progression (original → replies)
- Realistic subject line evolution ("Re: Re: Fwd:")
- Mixed relevance within threads (not all messages responsive)
- Bury responsive content at various thread positions (beginning, middle, end)

---

## 8. Generation Batches

### Phase 1: Responsive Emails (Primary Request)

Each batch should be generated, reviewed, and verified before proceeding to the next.

#### Batch 1: DIRECT_MATCH
- [ ] Generate 25-30 emails with explicit lead contamination discussion
- [ ] Verify: Clear lead references, varied contexts
- [ ] Status: NOT STARTED
- [ ] Count: 0/30

#### Batch 2: AMBIGUOUS_TERMS
- [ ] Generate 25-30 emails using "lead" (metal) with disambiguating context
- [ ] Verify: "Lead" appears, context clearly indicates metal
- [ ] Status: NOT STARTED
- [ ] Count: 0/30

#### Batch 3: INDIRECT_REFERENCE
- [ ] Generate 30-35 emails about lead topics without using "lead"
- [ ] Verify: No "lead" keyword, clearly about lead contamination
- [ ] Status: NOT STARTED
- [ ] Count: 0/35

#### Batch 4: TECHNICAL_JARGON
- [ ] Generate 20-25 emails using technical/regulatory terminology
- [ ] Verify: Technical terms present, accessible to domain experts
- [ ] Status: NOT STARTED
- [ ] Count: 0/25

#### Batch 5: TEMPORAL_REFERENCE
- [ ] Generate 20-25 emails with historical/future temporal framing
- [ ] Verify: Temporal element central to relevance
- [ ] Status: NOT STARTED
- [ ] Count: 0/25

#### Batch 6: BURIED_IN_THREAD (Responsive)
- [ ] Generate 30-40 thread emails with buried responsive content
- [ ] Verify: Responsive content exists but not prominent
- [ ] Status: NOT STARTED
- [ ] Count: 0/40

### Phase 2: Non-Responsive Emails (Primary Request)

#### Batch 7: KEYWORD_FALSE_POSITIVE
- [ ] Generate 50-60 emails using "lead" for leadership/leading
- [ ] Verify: "Lead" present, clearly not about metal
- [ ] Status: NOT STARTED
- [ ] Count: 0/60

#### Batch 8: ADJACENT_TOPIC
- [ ] Generate 40-50 emails about related but non-lead topics
- [ ] Verify: Water/infrastructure related, not lead-specific
- [ ] Status: NOT STARTED
- [ ] Count: 0/50

#### Batch 9: TRUE_NEGATIVE
- [ ] Generate 50-60 clearly unrelated emails
- [ ] Verify: No reasonable connection to lead/water/contamination
- [ ] Status: NOT STARTED
- [ ] Count: 0/60

### Phase 3: Validation Corpus (PFAS Request)

#### Batch 10: PFAS Mixed Set
- [ ] Generate 50-75 emails for PFAS request (mix of responsive/non-responsive)
- [ ] Verify: Appropriate challenge type distribution
- [ ] Status: NOT STARTED
- [ ] Count: 0/75

---

## 9. Verification Protocol

### Per-Batch Verification

After generating each batch:

1. **Challenge Type Accuracy**
   - Does each email genuinely represent its designated challenge type?
   - Would a human reviewer agree with the classification?

2. **Keyword Verification**
   - For INDIRECT_REFERENCE/TECHNICAL: Confirm "lead" does NOT appear
   - For KEYWORD_FALSE_POSITIVE: Confirm "lead" DOES appear
   - For AMBIGUOUS_TERMS: Confirm "lead" appears in metal context

3. **Variety Check**
   - Are sender roles varied?
   - Are email types/tones varied?
   - Any obvious patterns or repetition?

4. **Realism Check**
   - Do emails read like actual government communications?
   - Are dates/metadata plausible?
   - Any obviously synthetic artifacts?

### Cross-Batch Verification

After all batches complete:

1. **Distribution Check**
   - Appropriate balance across challenge types
   - Sufficient thread vs standalone emails
   - Reasonable responsive/non-responsive ratio (~45% responsive)

2. **Duplicate/Similarity Check**
   - No duplicate or near-duplicate emails
   - Sufficient variation across similar topics

3. **Ground Truth Consistency**
   - All annotations complete
   - Reasoning documented for edge cases

---

## 10. Post-Generation: Infrastructure Updates

### Required Code Changes

1. **Corpus Loader** (`src/data/corpus.py`)
   - Update to handle new corpus structure
   - Add thread concatenation logic
   - Simplify for single-request focus

2. **Experiment Runner** (`src/run_experiment.py`)
   - Update corpus path handling
   - Simplify request iteration (single request)
   - Add challenge-type breakdown reporting

3. **Evaluator** (`src/evaluation/evaluator.py`)
   - Add per-challenge-type metrics
   - Add thread-position analysis (do buried emails have lower scores?)
   - Update reporting format

4. **Configs**
   - New experiment configs for single-request setup
   - Threshold sweep configs

### Experiments to Run

After corpus generation:

1. **Baseline comparison**: Keyword vs top embedding models on new corpus
2. **Challenge type analysis**: Which types are hardest for each approach?
3. **Thread analysis**: Does burial depth affect retrieval?
4. **Threshold optimization**: What threshold balances recall/precision?
5. **Model comparison**: Re-run top models from previous experiments

---

## 11. Progress Tracking

### Overall Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Responsive | NOT STARTED | 0/185 |
| Phase 2: Non-Responsive | NOT STARTED | 0/170 |
| Phase 3: Validation | NOT STARTED | 0/75 |
| Infrastructure Updates | NOT STARTED | - |
| **Total** | **NOT STARTED** | **0/430** |

### Current Batch
**Next batch to generate:** Batch 1 (DIRECT_MATCH)

### Blockers/Notes
- None yet

---

## Appendix A: Example Emails by Challenge Type

### DIRECT_MATCH Example
```
From: maria.gonzalez@citywater.gov
To: infrastructure-team@citywater.gov
Date: March 15, 2023
Subject: Q1 Lead Testing Results - North District

Team,

Attached are the Q1 lead testing results for the North District service area.
We collected 127 samples from residential taps per LCR requirements.

Key findings:
- 3 samples exceeded the 15 ppb action level
- 90th percentile: 8.2 ppb (below action level)
- All exceedances were from pre-1950 homes with known lead service lines

We'll be scheduling LSL replacements for the affected addresses. Please
coordinate with Operations on the timeline.

Maria Gonzalez
Water Quality Manager
City Water Department
```

### INDIRECT_REFERENCE Example
```
From: david.chen@citywater.gov
To: city-manager@cityofexample.gov
Date: April 3, 2023
Subject: Infrastructure Assessment - Older Neighborhoods

Following up on our conversation about the aging pipe infrastructure in
the downtown historic district.

As you know, much of this infrastructure dates to the 1940s. Given what
happened in Flint and Newark, we've been prioritizing assessment of service
lines in this area. The materials used during that era are a known concern
for water quality.

I'd recommend we discuss budget allocation for accelerated replacement at
next month's council meeting. Several other California cities have faced
significant liability after similar situations came to light.

Happy to prepare a briefing document if helpful.

David
```

### KEYWORD_FALSE_POSITIVE Example
```
From: sarah.johnson@cityofexample.gov
To: department-heads@cityofexample.gov
Date: February 28, 2023
Subject: Committee Leadership Assignments

All,

Following last week's reorganization discussion, here are the updated
committee leadership assignments:

- Budget Committee: James will lead (previously Maria)
- Public Safety: Diana continues to lead
- Infrastructure: We need someone to take the lead - any volunteers?

Please confirm these work for your schedules. We want strong leadership
on each committee heading into the budget cycle.

Sarah Johnson
City Manager's Office
```

---

## Appendix B: Technical Terms Reference

For TECHNICAL_JARGON emails, use terms like:
- **LCR**: Lead and Copper Rule (EPA regulation)
- **LSL**: Lead Service Line
- **PPB**: Parts per billion (measurement unit)
- **Action Level**: 15 ppb for lead (EPA threshold)
- **90th percentile**: Statistical measure used in LCR compliance
- **Corrosion control**: Treatment to prevent pipe corrosion
- **First-draw sample**: Water sample taken after 6+ hours stagnation
- **CCT**: Corrosion Control Treatment
- **LCRR**: Lead and Copper Rule Revisions (2021 update)
- **Service line inventory**: Required catalog of pipe materials

---

*Last updated: [Date of last batch completion]*
*Generated by: Claude Code with human verification*
