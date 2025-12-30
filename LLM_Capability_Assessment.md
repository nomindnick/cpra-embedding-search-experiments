# EXP-000: Local LLM Capability Assessment

> Last updated: 2025-12-29 (qwen3:0.6b, deepseek-r1:1.5b complete; few-shot approach validated)

**Goal:** Identify which local LLMs to use for which tasks in subsequent experiments. Different models excel at different tasks (classification vs generation vs extraction), and latency matters when processing 339+ documents.

**Environment:** CPU-only (Ollama), testing models one at a time.

---

## Key Finding: Optimal Prompt Strategy is Model-Dependent

**Different models prefer different prompt approaches.** Testing revealed:

| Model | Best Approach | Accuracy | Latency |
|-------|---------------|----------|---------|
| qwen3:0.6b | Few-shot | **85%** | 4.23s |
| qwen3:1.7b | Zero-shot binary | **85%** | 13.33s |
| deepseek-r1:1.5b | Few-shot | 65% | 10.79s |

**Key insight:** Few-shot helps smaller models but can hurt larger ones. qwen3:1.7b had 25% parse errors with few-shot, suggesting it "overthinks" the examples.

**Recommendations:**
- **qwen3:0.6b:** Use `classification_few_shot` — best accuracy AND fastest
- **qwen3:1.7b:** Use `classification_binary` — few-shot causes parse errors
- **Test both approaches** when evaluating new models

---

## Summary Matrix

Quick reference for model recommendations by task type. Updated as testing progresses.

### Classification Tasks

| Model | Few-Shot | Binary | Ternary | JSON | Latency (s) | Notes |
|-------|----------|--------|---------|------|-------------|-------|
| qwen3:0.6b | **85%** | 70% | 50% | 85% | 4.23s | Few-shot is best |
| qwen3:1.7b | 60% | **85%** | — | 80% | 16.17s | Binary is best; few-shot hurts! |
| qwen3:8b | — | — | — | — | — | Pending |
| gemma3:4b | — | — | — | — | — | Pending |
| gemma3:12b | — | — | — | — | — | Pending |
| gemma2:2b | — | — | — | — | — | Pending |
| phi4-mini:3.8b | — | — | — | — | — | Pending |
| phi4-mini-reasoning:3.8b | — | — | — | — | — | Pending |
| phi3:mini | — | — | — | — | — | Pending |
| granite3.3:2b | — | — | — | — | — | Pending |
| granite3.3:8b | — | — | — | — | — | Pending |
| deepseek-r1:1.5b | 65% | 50% | — | 55% | 13.11s | NOT REC: Few-shot helps but still poor |
| deepseek-r1:8b | — | — | — | — | — | Pending |
| ministral-3:3b | — | — | — | — | — | Pending |
| ministral-3:8b | — | — | — | — | — | Pending |
| ministral-3:14b | — | — | — | — | — | Pending |
| llama3:8b-instruct-q5_K_M | — | — | — | — | — | Pending |
| gpt-oss:20b | — | — | — | — | — | Pending |
| olmo-3:7b | — | — | — | — | — | Pending |
| functiongemma:270m | — | — | — | — | — | Pending |

### Generation Tasks

| Model | Paraphrases | Examples | Diversity | Latency (s) | Notes |
|-------|-------------|----------|-----------|-------------|-------|
| qwen3:0.6b | 5/5 | 100% | 4.4% | 9.34s | Low diversity but correct structure |
| qwen3:1.7b | — | — | — | — | Pending |
| qwen3:8b | — | — | — | — | Pending |
| gemma3:4b | — | — | — | — | Pending |
| gemma3:12b | — | — | — | — | Pending |
| gemma2:2b | — | — | — | — | Pending |
| phi4-mini:3.8b | — | — | — | — | Pending |
| phi4-mini-reasoning:3.8b | — | — | — | — | Pending |
| phi3:mini | — | — | — | — | Pending |
| granite3.3:2b | — | — | — | — | Pending |
| granite3.3:8b | — | — | — | — | Pending |
| deepseek-r1:1.5b | 0/5 | 50% | — | 22.26s | Failed paraphrase; poor structure |
| deepseek-r1:8b | — | — | — | — | Pending |
| ministral-3:3b | — | — | — | — | Pending |
| ministral-3:8b | — | — | — | — | Pending |
| ministral-3:14b | — | — | — | — | Pending |
| llama3:8b-instruct-q5_K_M | — | — | — | — | Pending |
| gpt-oss:20b | — | — | — | — | Pending |
| olmo-3:7b | — | — | — | — | Pending |
| functiongemma:270m | — | — | — | — | Pending |

### Extraction Tasks

| Model | Evidence Quotes | Keywords | Quote Accuracy | Latency (s) | Notes |
|-------|-----------------|----------|----------------|-------------|-------|
| qwen3:0.6b | 1.1 avg | 100% | 65% | 6.75s | Good format, some hallucinated quotes |
| qwen3:1.7b | — | — | — | — | Pending |
| qwen3:8b | — | — | — | — | Pending |
| gemma3:4b | — | — | — | — | Pending |
| gemma3:12b | — | — | — | — | Pending |
| gemma2:2b | — | — | — | — | Pending |
| phi4-mini:3.8b | — | — | — | — | Pending |
| phi4-mini-reasoning:3.8b | — | — | — | — | Pending |
| phi3:mini | — | — | — | — | Pending |
| granite3.3:2b | — | — | — | — | Pending |
| granite3.3:8b | — | — | — | — | Pending |
| deepseek-r1:1.5b | 1.2 avg | 90% | 22% | 19.66s | High hallucination; poor quote accuracy |
| deepseek-r1:8b | — | — | — | — | Pending |
| ministral-3:3b | — | — | — | — | Pending |
| ministral-3:8b | — | — | — | — | Pending |
| ministral-3:14b | — | — | — | — | Pending |
| llama3:8b-instruct-q5_K_M | — | — | — | — | Pending |
| gpt-oss:20b | — | — | — | — | Pending |
| olmo-3:7b | — | — | — | — | Pending |
| functiongemma:270m | — | — | — | — | Pending |

---

## Recommendations (Updated as Testing Progresses)

### Best for Classification
- **Primary:** TBD
- **Fast alternative:** TBD

### Best for Generation
- **Primary:** TBD
- **Fast alternative:** TBD

### Best for Extraction
- **Primary:** TBD
- **Fast alternative:** TBD

### Speed vs Quality Tradeoffs
- **Fastest usable:** TBD
- **Best quality (if time allows):** TBD
- **Sweet spot:** TBD

---

## Test Dataset

20 documents selected from primary corpus to cover the full difficulty spectrum:

### Clear Responsive (5 docs) - DIRECT_MATCH
| Doc ID | Subject | Notes |
|--------|---------|-------|
| email_001 | Q1 Lead Testing Results - North District | Lead testing results with ppb values |
| email_002 | Urgent: Elevated Lead Levels at Jefferson Elementary | School lead contamination alert |
| email_003 | Lead and Copper Rule Compliance Review - Scheduled Site Visit | EPA LCR compliance review |
| email_004 | FY24 Budget Request: Lead Service Line Replacement Program | Budget for LSL replacement |
| email_009 | Fall Lead Sampling Campaign - Schedule and Assignments | Lead sampling logistics |

### Tricky Responsive (5 docs) - INDIRECT_REFERENCE, TECHNICAL_JARGON, BURIED_IN_THREAD
| Doc ID | Subject | Challenge Type | Notes |
|--------|---------|----------------|-------|
| email_065 | Verification dig at 892 Maple - you need to see this | INDIRECT_REFERENCE | Describes lead pipe as "gray, soft when scratched" without saying "lead" |
| email_067 | Fwd: Concerned resident - house built in 1938 | INDIRECT_REFERENCE | References Newark crisis without mentioning lead |
| email_096 | LCRR Implementation Timeline Reminder | TECHNICAL_JARGON | Uses LCRR, LSL, GRR, CCT, AL terminology |
| email_099 | CCT Optimization Study - Preliminary Findings | TECHNICAL_JARGON | Pb solubility, passivation, orthophosphate dosage |
| email_147 | Re: New Employee Orientation - March 20th Cohort | BURIED_IN_THREAD | Lead program mentioned in reply to HR orientation email |

### Tricky Non-Responsive (5 docs) - KEYWORD_FALSE_POSITIVE, ADJACENT_TOPIC
| Doc ID | Subject | Challenge Type | Notes |
|--------|---------|----------------|-------|
| email_185 | Committee Leadership Assignments for 2023 | KEYWORD_FALSE_POSITIVE | "lead" = leadership/leading |
| email_186 | Leadership Development Program - Applications Open | KEYWORD_FALSE_POSITIVE | "lead" = leadership roles |
| email_189 | Taking the Lead on Permit System Upgrade | KEYWORD_FALSE_POSITIVE | "lead" = taking charge |
| email_240 | Copper Action Level Exceedance - Oak Street | ADJACENT_TOPIC | Water quality but copper, not lead |
| email_245 | URGENT: Total Coliform Positive - Zone 4 | ADJACENT_TOPIC | Water quality but bacteria, not lead |

### Clear Non-Responsive (5 docs) - TRUE_NEGATIVE
| Doc ID | Subject | Notes |
|--------|---------|-------|
| email_146 | New Employee Orientation - March 20th Cohort | HR orientation scheduling |
| email_288 | Retirement Celebration - Chief Williams | HR retirement announcement |
| email_290 | Holiday Schedule - Independence Day | HR holiday schedule |
| email_291 | New Telework Policy Effective August 1 | HR telework policy |
| email_295 | URGENT: Password Reset Required by January 31 | IT security announcement |

---

## Task Definitions & Prompts

### Task 1: Binary Classification

**Goal:** Given document + request, output YES/NO

**Prompt:**
```
You are evaluating a document for a California Public Records Act (CPRA) request.

CPRA REQUEST:
{request_text}

DOCUMENT:
{document_text}

Is this document responsive to the CPRA request? Answer only YES or NO.
```

**Metrics:**
- Accuracy (% correct)
- Latency (seconds)

---

### Task 2: Ternary Classification with Confidence

**Goal:** Output yes/no/maybe with confidence 0-100

**Prompt:**
```
You are evaluating a document for a California Public Records Act (CPRA) request.

CPRA REQUEST:
{request_text}

DOCUMENT:
{document_text}

Determine if this document is responsive to the CPRA request.

Output your answer in this exact format:
DECISION: [yes/no/maybe]
CONFIDENCE: [0-100]
```

**Metrics:**
- Accuracy (% correct decisions)
- Confidence calibration (are high-confidence answers more accurate?)
- Format compliance (% valid format)
- Latency (seconds)

---

### Task 3: JSON Format Compliance

**Goal:** Can it output valid, parseable JSON?

**Prompt:**
```
You are evaluating a document for a California Public Records Act (CPRA) request.

CPRA REQUEST:
{request_text}

DOCUMENT:
{document_text}

Analyze this document and provide your response as valid JSON with this structure:
{
  "responsive": "yes" | "no" | "maybe",
  "confidence": 0-100,
  "reasoning": "brief explanation"
}

Output ONLY the JSON, no other text.
```

**Metrics:**
- JSON parse success rate
- Schema compliance rate
- Accuracy of decisions
- Latency (seconds)

---

### Task 4: Evidence Extraction

**Goal:** Can it quote verbatim from the document?

**Prompt:**
```
You are evaluating a document for a California Public Records Act (CPRA) request.

CPRA REQUEST:
{request_text}

DOCUMENT:
{document_text}

If this document contains information relevant to the CPRA request, extract the specific passages that are relevant. Quote them EXACTLY as they appear in the document.

Output format:
RESPONSIVE: [yes/no]
EVIDENCE:
- "[exact quote 1]"
- "[exact quote 2]"
```

**Metrics:**
- Quote accuracy (do quotes appear verbatim in source?)
- Relevance of quoted passages
- Latency (seconds)

---

### Task 5: Paraphrase Generation

**Goal:** Generate 5 diverse paraphrases of a request

**Prompt:**
```
Given this CPRA request:
{request_text}

Generate 5 semantically different paraphrases of this request.
Each paraphrase should:
- Capture the same information need
- Use different vocabulary and phrasing
- Emphasize different aspects of what's being requested

Output only the paraphrases, one per line, numbered 1-5.
```

**Metrics:**
- Number of unique paraphrases generated
- Semantic similarity to original (should be high)
- Diversity between paraphrases (should be moderate)
- Latency (seconds)

---

### Task 6: Example Generation

**Goal:** Generate realistic responsive/non-responsive emails

**Prompt (Responsive):**
```
Given this CPRA request:
{request_text}

Generate a realistic email that WOULD be responsive to this request.
The email should:
- Discuss the subject matter substantively
- NOT use these exact keywords: {keywords}
- Include realistic email metadata (From, To, Subject, Date)
- Be 100-200 words

Output only the email.
```

**Prompt (Non-Responsive):**
```
Given this CPRA request:
{request_text}

Generate a realistic email that is RELATED TO but NOT responsive to this request.
It should be a plausible false positive that might fool a keyword search.

The email should:
- Mention related topics but not the actual subject matter
- Include realistic email metadata (From, To, Subject, Date)
- Be 100-200 words

Output only the email.
```

**Metrics:**
- Realism (subjective)
- Correct responsiveness classification
- Follows constraints (no forbidden keywords, appropriate length)
- Latency (seconds)

---

### Task 7: Keyword/Entity Extraction

**Goal:** Extract relevant keywords/entities from text

**Prompt:**
```
Given this CPRA request:
{request_text}

Extract the following to help find relevant documents:

KEYWORDS: Important single words and multi-word phrases
ENTITIES: Organizations, facilities, people, or projects mentioned
CONCEPTS: Abstract topics or themes being requested

Output in this format:
KEYWORDS: term1, term2, term3
ENTITIES: entity1, entity2
CONCEPTS: concept1, concept2
```

**Metrics:**
- Relevance of extracted terms
- Coverage of key concepts
- Format compliance
- Latency (seconds)

---

## Detailed Model Results

### qwen3:0.6b

**Status:** Complete

**Test Date:** 2025-12-29

**Model Info:**
- Parameters: 0.6B
- Size: 522 MB
- Notes: Ultra-fast baseline, smallest Qwen model

#### Classification Results

**Summary (after prompt iteration):**
- **Few-shot accuracy: 17/20 (85%)** — best approach, nearly unbiased (0.55 predicted)
- Binary accuracy: 14/20 (70%) — improved with new prompt, slight YES bias
- Ternary accuracy: 10/20 (50%) — regressed with new prompt
- JSON accuracy: 17/20 (85%) — tied with few-shot
- Avg latency: 4.23s (few-shot), 4.88s (binary), 6.15s (JSON)

**Observations:**
- Few-shot examples dramatically improve classification accuracy
- Few-shot has best calibration (predicted 0.55 vs expected 0.50)
- JSON produces same accuracy but slower and more complex to parse
- Ternary (yes/no/maybe) format confuses the model — avoid

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | 5/5 generated | 100% | 12.04 |
| Responsive example | Structure complete | 100% (145 words avg) | 6.86 |
| Non-responsive example | Structure complete | 100% (167 words avg) | 9.12 |

**Observations:**
- Produces correct number of paraphrases every time
- Very low diversity (4.4%) — paraphrases too similar to original
- Email examples have correct From/To/Subject/Body structure
- Good word count in target range (100-250)

#### Extraction Results

**Summary:**
- Evidence quotes found: 1.1 average per document
- Quote accuracy: 65% (quotes actually appear in source)
- "No relevant content" responses: 55% of documents
- Keyword extraction: 100% format compliance
- Keywords per doc: 9.4 average

**Observations:**
- Hallucination issue: ~35% of quotes don't appear verbatim in source
- Good at recognizing non-responsive documents (55% correct "no content")
- Keyword extraction format is reliable (KEYWORDS/ENTITIES/ACRONYMS)

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| Fast (4-5s avg) | Quote hallucination (35%) |
| **85% classification with few-shot** | Low paraphrase diversity |
| Good structure following | Ternary format unreliable |
| Reliable format output | |

**Recommendation:** USE FEW-SHOT PROMPTING. With few-shot examples, this tiny model achieves 85% classification accuracy with nearly perfect calibration. Best balance of speed and accuracy for classification tasks. Still has hallucination issues on extraction tasks.

---

### qwen3:1.7b

**Status:** Complete (classification only)

**Test Date:** 2025-12-29

**Model Info:**
- Parameters: 1.7B
- Size: 1.4 GB
- Notes: Larger Qwen model — interesting reversal from 0.6b

#### Classification Results

**Summary:**
- Few-shot accuracy: 12/20 (60%) — WORSE than 0.6b! 25% parse errors, severe NO bias
- **Binary accuracy: 17/20 (85%)** — BEST for this model, matches 0.6b few-shot
- JSON accuracy: 16/20 (80%) — good, 100% valid JSON
- Avg latency: 16.17s (3-4x slower than qwen3:0.6b)

**Observations:**
- **Few-shot hurts this model** — parse errors and NO bias (predicted 0.13 vs expected 0.33)
- Zero-shot binary is the optimal approach for qwen3:1.7b
- Model may be "overthinking" the few-shot examples
- Much slower than 0.6b without accuracy gains

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

*(Not tested — classification focus)*

#### Extraction Results

*(Not tested — classification focus)*

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| 85% binary accuracy | 3-4x slower than 0.6b |
| 100% JSON compliance | Few-shot causes problems |
| Good calibration on binary | No accuracy gain over 0.6b |

**Recommendation:** MIXED. Achieves same 85% as qwen3:0.6b few-shot, but via zero-shot binary and at 3-4x the latency. For speed-sensitive applications, stick with qwen3:0.6b + few-shot. For batch processing where latency matters less, either works.

---

### qwen3:8b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 8B
- Notes: Flagship Qwen

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### gemma3:4b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 4B
- Notes: Google's latest, efficient

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### gemma3:12b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 12B
- Notes: Google's latest, larger

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### gemma2:2b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 2B
- Notes: Previous gen, fast

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### phi4-mini:3.8b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 3.8B
- Notes: Microsoft's efficient model

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### phi4-mini-reasoning:3.8b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 3.8B
- Notes: Reasoning-focused variant

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### phi3:mini

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 3B
- Notes: Previous gen, stable

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### granite3.3:2b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 2B
- Notes: IBM, fast

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### granite3.3:8b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 8B
- Notes: IBM, good for structured output

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### deepseek-r1:1.5b

**Status:** Complete

**Test Date:** 2025-12-29

**Model Info:**
- Parameters: 1.5B
- Size: 1.1 GB
- Notes: Reasoning-focused model (R1 series)

#### Classification Results

**Summary (with updated prompts including few-shot):**
- **Few-shot accuracy: 13/20 (65%)** — best for this model, but still 20pts below qwen3:0.6b
- Binary accuracy: 10/20 (50%) — coin flip, regressed with new prompts
- JSON accuracy: 11/20 (55%) — improved slightly
- Avg latency: 13.11 seconds (3x slower than qwen3:0.6b)

**Observations:**
- Few-shot helps this model (+20pts over original binary) but not enough
- High parse error rates: 5% (few-shot), 25% (binary), 20% (JSON)
- Still biased toward NO (predicted 0.26 vs expected 0.47)
- Reasoning overhead makes it slower without accuracy gains

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | 0/5 generated | 0% | 32.60 |
| Responsive example | Missing From field | 0% | 21.06 |
| Non-responsive example | Structure complete | 100% | 13.12 |

**Observations:**
- **Complete failure on paraphrase task** — generated 0 paraphrases
- Positive example generation missing required From: field
- Only negative example generation worked correctly
- Very slow on generation tasks (22s+ average)

#### Extraction Results

**Summary:**
- Evidence quotes found: 1.2 average per document
- Quote accuracy: 22% (quotes actually appear in source)
- "No relevant content" responses: 55% of documents
- Keyword extraction: 90% format compliance
- Keywords per doc: 3.45 average (less than qwen3:0.6b)

**Observations:**
- **Severe hallucination problem** — 78% of quotes are fabricated
- Worse than qwen3:0.6b on every extraction metric
- Lower keyword count despite larger model

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| Few-shot approach helps (+20pts) | 3x slower than qwen3:0.6b |
| | Still 20pts below qwen3:0.6b with few-shot |
| | High parse error rates (5-25%) |
| | Failed paraphrase generation |
| | 78% quote hallucination rate |

**Recommendation:** NOT RECOMMENDED. Even with few-shot prompting (which helps), this model only reaches 65% vs qwen3:0.6b's 85%. The reasoning-focused architecture creates format-following issues and slowness without compensating accuracy gains. Skip deepseek-r1 family for CPRA tasks.

---

### deepseek-r1:8b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 8B
- Notes: Reasoning focused, larger

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### ministral-3:3b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 3B
- Notes: Mistral small/fast

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### ministral-3:8b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 8B
- Notes: Mistral medium

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### ministral-3:14b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 14B
- Notes: Mistral large

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### llama3:8b-instruct-q5_K_M

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 8B
- Notes: Meta, instruction-tuned

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### gpt-oss:20b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 20B
- Notes: Largest available

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### olmo-3:7b

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 7B
- Notes: Allen AI open model

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

### functiongemma:270m

**Status:** Pending

**Test Date:** —

**Model Info:**
- Parameters: 270M
- Notes: Function calling specialist

#### Classification Results

| Doc ID | Expected | Binary | Ternary | Confidence | JSON Valid |
|--------|----------|--------|---------|------------|------------|
| — | — | — | — | — | — |

**Summary:**
- Binary accuracy: —/20 (—%)
- Ternary accuracy: —/20 (—%)
- JSON compliance: —/20 (—%)
- Avg latency: — seconds

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

#### Extraction Results

| Doc ID | Quotes Accurate | Keywords Relevant | Latency (s) |
|--------|-----------------|-------------------|-------------|
| — | — | — | — |

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

#### Notes

—

---

## Testing Log

Record of test runs for reproducibility.

| Date | Model | Tasks Run | Duration | Notes |
|------|-------|-----------|----------|-------|
| 2025-12-29 | qwen3:0.6b | All 8 | ~2 min | Baseline complete. 67% classification avg, 100% JSON compliance |
| 2025-12-29 | deepseek-r1:1.5b | All 8 | ~5 min | NOT RECOMMENDED. Slower & worse than qwen3:0.6b. 78% quote hallucination |
| 2025-12-29 | qwen3:0.6b | Classification | ~2 min | **FEW-SHOT = 85%** — best approach discovered. Balanced predictions. |
| 2025-12-29 | deepseek-r1:1.5b | Classification | ~4 min | Few-shot = 65% (best for this model). Still 20pts below qwen3:0.6b. |
| 2025-12-29 | qwen3:1.7b | Classification | ~5 min | Binary = 85% best. Few-shot hurts (60%)! Prompt approach is model-dependent. |

---

## Conclusions

*To be completed after testing.*
