# EXP-000: Local LLM Capability Assessment

> Last updated: 2026-01-08 (added olmo-3:7b partial — catastrophic failure on few-shot/multi-shot)

**Goal:** Identify which local LLMs to use for which tasks in subsequent experiments. Different models excel at different tasks (classification vs generation vs extraction), and latency matters when processing 339+ documents.

**Environment:** CPU-only (Ollama), testing models one at a time.

---

## Key Finding: Optimal Prompt Strategy is Model-Dependent

**Different models prefer different prompt approaches.** Testing revealed:

| Model | Size | Best Approach | Accuracy | Latency |
|-------|------|---------------|----------|---------|
| gemma3:4b | 3.3 GB | Few-shot | **100%** ⭐⭐ | **3.3s** |
| gemma3:12b | 8.1 GB | Binary | 96% | 10.5s |
| qwen3:8b | 5.2 GB | Zero-shot | 95% | 45.6s |
| phi4-mini:3.8b | 2.5 GB | Multi-shot | 90% | 6.7s |
| gemma2:2b | 1.6 GB | Ternary | 90% | 2.2s |
| qwen3:0.6b | 522 MB | Few-shot | 85% | 4.2s |
| qwen3:1.7b | 1.4 GB | Zero-shot | 85% | 13.3s |
| deepseek-r1:1.5b | 1.1 GB | Few-shot | 65% | 10.8s |

**Key insight:** Optimal prompt strategy varies dramatically by model family:
- **gemma3:4b:** Few-shot achieves **100% accuracy** — best for classification!
- **gemma3:12b:** Binary best (96%); few-shot doesn't help; **96% quote extraction** ⭐
- **phi4-mini:** Multi-shot (6 examples) works best — 90% @ 6.7s; **76% extraction**
- **gemma2:** Ternary format works best — 90% @ 2.2s (different from gemma3!)
- **qwen:** Larger models prefer zero-shot; smaller prefer few-shot
- Reasoning models (phi4-mini-reasoning, deepseek-r1) don't work — parsing issues

**Recommendations:**
- **gemma3:4b:** Use `classification_few_shot` — **100% @ 3.3s** ⭐⭐ FASTEST PERFECT
- **ministral-3:3b:** Use `classification_ternary` — **100% @ 11.3s** ⭐⭐ NEW! Also **96% extraction**
- **gemma3:12b:** Use for extraction — **96% quote accuracy** ⭐ (tied with ministral-3:3b)
- **phi4-mini:3.8b:** Use `classification_multi_shot` — 90% @ 6.7s; also **76% extraction**, **100% paraphrase**
- **gemma2:2b:** Use `classification_ternary` — 90% @ 2.2s (fastest)
- **qwen3:8b:** Use `classification_binary` — 95% @ 45.6s (if latency acceptable)
- **AVOID reasoning models** (phi4-mini-reasoning, deepseek-r1) — parsing issues

---

## Summary Matrix

Quick reference for model recommendations by task type. Updated as testing progresses.

### Classification Tasks

| Model | Few-Shot | Multi-Shot | Binary | Ternary | JSON | Latency (s) | Notes |
|-------|----------|------------|--------|---------|------|-------------|-------|
| qwen3:0.6b | **85%** | 75% | 70% | 50% | 85% | 4.2s | Few-shot is best |
| qwen3:1.7b | 60% | — | **85%** | — | 80% | 16.2s | Binary is best; few-shot hurts! |
| qwen3:8b | 60% | 60% | **95%** | **95%** | **95%** | 45.6s | Zero-shot is best; few-shot hurts! ⭐ |
| gemma3:4b | **100%** ⭐ | 95% | 95% | 70% | 85% | 3.3s | Few-shot is best! Perfect score |
| gemma3:12b | 95% | 95% | **96%** | 95% | 92% | 10.5s | Binary best; examples don't help |
| gemma2:2b | 65% | 80% | 60% | **90%** ⭐ | 50% | 2.2s | Ternary is best! Multi-shot helps |
| phi4-mini:3.8b | 84% | **90%** | 82% | 89% | 89% | 6.7s | Multi-shot best; well-rounded |
| phi4-mini-reasoning:3.8b | — | — | 46% | — | — | 46.5s | NOT REC: reasoning breaks parsing |
| phi3:mini | 61% | **71%** | 60% | — | — | 22.9s | NOT REC: Multi-shot best but weak (71%) |
| granite3.3:2b | 60% | 77% | **85%** | 70% | **85%** | 11.8s | Binary/JSON best; 100% JSON compliance ⭐ |
| granite3.3:8b | — | — | — | — | — | — | Pending |
| deepseek-r1:1.5b | 65% | — | 50% | — | 55% | 13.1s | NOT REC: Few-shot helps but still poor |
| deepseek-r1:8b | — | — | — | — | — | — | Pending |
| ministral-3:3b | 80% | 55% | 90% | **100%** ⭐ | 95% | 11.3s | Ternary is best! Zero-shot wins |
| ministral-3:8b | 60% | Timeout | 90% | **100%** ⭐ | Timeout | 26.2s | NOT REC: 3b is better (faster, better extraction) |
| ministral-3:14b | — | — | — | — | — | — | Pending |
| llama3:8b-instruct-q5_K_M | — | — | — | — | — | — | Pending |
| gpt-oss:20b | 85% | 80% | 85% | **87%** | 80% | ~55s | NOT REC: 15pts below gemma3:4b, extremely slow |
| olmo-3:7b | **5%** | 10% | **72%** | — | — | ~22s | NOT REC: Catastrophic few-shot failure |
| functiongemma:270m | 50% | 50% | 57% | 50% | 37% | 1.25s | **NOT REC**: Function calling model, random accuracy |

### Generation Tasks

| Model | Paraphrases | Examples | Diversity | Latency (s) | Notes |
|-------|-------------|----------|-----------|-------------|-------|
| qwen3:0.6b | 5/5 | 100% | 8.8% | 8.6s | Good structure |
| qwen3:1.7b | — | — | — | — | Pending |
| qwen3:8b | 5/5 | 100%* | 15.9% | 106s | Higher diversity; *timeouts on neg examples |
| gemma3:4b | 0/5 | **100%** | — | 24s | Failed paraphrase; excellent email gen |
| gemma3:12b | 0/5 | 50%* | — | 114s | Timeouts; *only neg example works |
| gemma2:2b | 0/5 | 50%* | — | 17s | Failed paraphrase; *neg only works |
| phi4-mini:3.8b | **5/5** | 50%* | — | 22s | **100% paraphrase**; *pos example fails |
| phi4-mini-reasoning:3.8b | — | — | — | — | NOT REC: too slow |
| phi3:mini | — | — | — | — | Pending |
| granite3.3:2b | **5/5** | **100%** | 9.2% | 32.5s | Excellent generation; all tasks complete |
| granite3.3:8b | — | — | — | — | Pending |
| deepseek-r1:1.5b | 0/5 | 50% | — | 22.26s | Failed paraphrase; poor structure |
| deepseek-r1:8b | — | — | — | — | Pending |
| ministral-3:3b | **5/5** | **100%** | — | 43.7s | All 3 tasks work! Slow but reliable |
| ministral-3:8b | Timeout | Timeout | — | — | NOT REC: Too slow, all generation tasks timeout |
| ministral-3:14b | — | — | — | — | Pending |
| llama3:8b-instruct-q5_K_M | — | — | — | — | Pending |
| gpt-oss:20b | — | — | — | — | Pending |
| olmo-3:7b | — | — | — | — | Pending |
| functiongemma:270m | 0/5 | 0% | — | 0.8s | **NOT REC**: Refuses generation tasks |

### Extraction Tasks

| Model | Evidence Quotes | Search Terms | Quote Accuracy | Latency (s) | Notes |
|-------|-----------------|--------------|----------------|-------------|-------|
| qwen3:0.6b | 1.1 avg | 14 terms | 36% | 6.9s | Good format, some hallucinated quotes |
| qwen3:1.7b | — | — | — | — | Pending |
| qwen3:8b | 2.0 avg | timeout | 29% | 56.7s | More quotes but lower accuracy; timeout on search terms |
| gemma3:4b | ~0.6 avg | ✓ | 16% | 10.8s | Low quote accuracy; search terms work |
| gemma3:12b | High | ✓ | **96%** ⭐ | 19.2s | BEST quote accuracy! Slow but excellent |
| gemma2:2b | 0.6 avg | Failed | 41% | 10.6s | Conservative; search term format failed |
| phi4-mini:3.8b | Good | ✓ | **76%** | 4.4s | Excellent extraction; fast |
| phi4-mini-reasoning:3.8b | — | — | — | — | NOT REC: too slow |
| phi3:mini | — | — | — | — | Pending |
| granite3.3:2b | 0.17 avg | 13 terms | 25% | 16.6s | Conservative (70% "no content"); good format |
| granite3.3:8b | — | — | — | — | Pending |
| deepseek-r1:1.5b | 1.2 avg | 90% | 22% | 19.66s | High hallucination; poor quote accuracy |
| deepseek-r1:8b | — | — | — | — | Pending |
| ministral-3:3b | ✓ | ✓ | **96%** ⭐ | 15.2s | Ties gemma3:12b for best quote accuracy! |
| ministral-3:8b | ✓ | Timeout | 87% | 29.9s | NOT REC: Worse than 3b (87% vs 96%), slower |
| ministral-3:14b | — | — | — | — | Pending |
| llama3:8b-instruct-q5_K_M | — | — | — | — | Pending |
| gpt-oss:20b | — | — | — | — | Pending |
| olmo-3:7b | — | — | — | — | Pending |
| functiongemma:270m | 0 avg | 0 terms | 0% | 0.9s | **NOT REC**: Complete failure on extraction |

---

## Recommendations (Updated as Testing Progresses)

### Best for Classification
- **Primary (fastest):** gemma3:4b (**100% accuracy @ 3.3s**) — use `classification_few_shot` ⭐⭐ FASTEST PERFECT
- **Primary (alternative):** ministral-3:3b (**100% accuracy @ 11.3s**) — use `classification_ternary` ⭐⭐ NEW
- **Fast alternative:** gemma2:2b (90% accuracy @ 2.2s) — use `classification_ternary`

### Best for Generation
- **Primary:** phi4-mini:3.8b — **100% paraphrase**, good structure @ 22s avg
- **Email generation:** gemma3:4b — 100% structure compliance for both positive and negative
- **Paraphrase + emails:** qwen3:0.6b — 100% paraphrase, 100% emails, fastest (8.6s)
- **Paraphrase diversity:** qwen3:8b — 15.9% diversity (but slow and timeouts)

### Best for Extraction
- **Primary (tied):** gemma3:12b — **96% quote accuracy** @ 19.2s ⭐
- **Primary (tied):** ministral-3:3b — **96% quote accuracy** @ 15.2s ⭐ (faster!)
- **Fast + accurate:** phi4-mini:3.8b — **76% quote accuracy** @ 4.4s (best speed/accuracy ratio)
- **Budget:** gemma2:2b — 41% quote accuracy, fast (10.6s)

### Speed vs Quality Tradeoffs
- **Best quality:** gemma3:4b + few-shot (**100% @ 3.3s/doc**) ⭐⭐ PERFECT
- **Fastest usable:** gemma2:2b + ternary (90% @ 2.2s/doc)
- **Sweet spot:** gemma3:4b + few-shot — perfect accuracy, fast enough

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

**Status:** Complete

**Test Date:** 2025-12-30

**Model Info:**
- Parameters: 8B
- Size: 5.2 GB
- Notes: Flagship Qwen — **best classification accuracy (95%)** but slow on CPU

#### Classification Results

**Summary:**
- **Binary accuracy: 19/20 (95%)** — BEST overall, perfectly calibrated (0.45 predicted vs 0.50 expected)
- **Ternary accuracy: 19/20 (95%)** — excellent, high confidence (95.5 avg)
- **JSON accuracy: 19/20 (95%)** — 100% valid JSON
- Few-shot accuracy: 12/20 (60%) — 25% parse errors, severe NO bias (0.13 predicted)
- Multi-shot accuracy: 12/20 (60%) — 20% parse errors, severe NO bias (0.12 predicted)
- Avg latency: 45.6 seconds (binary), 55.0s (few-shot), 65.7s (multi-shot)

**Observations:**
- **Zero-shot dramatically outperforms few-shot** — examples confuse this model
- Parse errors with few-shot (25%) and multi-shot (20%) suggest model "overthinks" examples
- Severe NO bias with examples (predicted ~0.12 vs expected ~0.35)
- All zero-shot approaches (binary, ternary, JSON) achieve same 95% accuracy
- ~10x slower than qwen3:0.6b on CPU

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | 5/5 generated | 100% | 116.0 |
| Responsive example | Structure complete | 100% (118 words) | 95.6 |
| Non-responsive example | **TIMEOUT** | — | — |

**Observations:**
- Higher paraphrase diversity (15.9%) than qwen3:0.6b (8.8%)
- Negative example generation hit 120s timeout — model too slow for long outputs on CPU
- Good structure compliance when it completes

#### Extraction Results

**Summary:**
- Evidence quotes found: 2.0 average per document (more than 0.6b)
- Quote accuracy: 29% (worse than 0.6b's 36%)
- "No relevant content" responses: 5% of documents
- Search term extraction: **TIMEOUT**

**Observations:**
- Finds more quotes but lower accuracy — more hallucination
- Too slow for search term extraction (120s timeout)
- 1 error in evidence extraction task

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| **95% classification accuracy** ⭐ | ~10x slower than 0.6b (~45s vs ~4s) |
| Perfect calibration on zero-shot | Timeouts on generation/extraction |
| 100% JSON compliance | Few-shot causes parse errors & bias |
| High confidence (95.5 avg) | Not practical for batch processing on CPU |

**Recommendation:** BEST FOR ACCURACY, use zero-shot prompts. If latency is acceptable (~45s/doc), qwen3:8b achieves the highest classification accuracy at 95%. Avoid few-shot/multi-shot prompts — they confuse the model. For speed-sensitive applications, use qwen3:0.6b with few-shot (85% accuracy, 10x faster).

---

### gemma3:4b

**Status:** Complete ⭐⭐

**Test Date:** 2025-12-30

**Model Info:**
- Parameters: 4B
- Size: 3.3 GB
- Notes: **BEST CLASSIFICATION MODEL — 100% accuracy with few-shot!**

#### Classification Results

**Summary:**
- Binary accuracy: 19/20 (95%) — well-calibrated (0.55 pred vs 0.50 expected)
- **Few-shot accuracy: 20/20 (100%)** ⭐⭐ — PERFECT, perfectly calibrated (0.50 pred)
- Multi-shot accuracy: 19/20 (95%) — no improvement over binary
- Ternary accuracy: 14/20 (70%) — drops significantly, avoid
- JSON accuracy: 17/20 (85%) — 100% valid JSON, slight YES bias (0.65 pred)
- Avg latency: 3.3-3.5s (classification), 9.2s (JSON)

**Observations:**
- **Few-shot achieves PERFECT 100% accuracy** — first model to do so!
- Perfectly calibrated predictions (0.50 predicted vs 0.50 expected)
- Ternary format hurts this model (-30% vs few-shot) — opposite of gemma2!
- Very consistent ~3.3s latency across most tasks
- JSON works but slower and less accurate

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | 0/5 generated | 0% | 34.8 |
| Responsive example | Structure complete | **100%** | 24.3 |
| Non-responsive example | Structure complete | **100%** | 24.0 |

**Observations:**
- Failed paraphrase generation — didn't produce numbered format
- Excellent email generation — 100% structure compliance for both positive and negative
- Fast for generation tasks compared to larger models

#### Extraction Results

**Summary:**
- Evidence quotes found: Low (~0.6 avg)
- Quote accuracy: **16%** (worse than other models)
- Search term extraction: Completed @ 15.4s
- Very conservative on evidence extraction

**Observations:**
- Struggles with verbatim quote extraction — low accuracy
- Works for search term extraction but not evidence
- Classification is this model's strength, not extraction

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| **100% classification accuracy** ⭐⭐ | Ternary format hurts (-30%) |
| Perfect calibration | Failed paraphrase generation |
| Fast (3.3s/doc) | Low extraction accuracy (16%) |
| No parse errors | |
| 100% email generation | |

**Recommendation:** BEST FOR CLASSIFICATION. Use `classification_few_shot` for perfect 100% accuracy at 3.3s/doc. Avoid ternary (drops to 70%). Excellent at email generation (100%) but struggles with extraction tasks. This is the gold standard for CPRA classification.

---

### gemma3:12b

**Status:** Complete

**Test Date:** 2025-12-30

**Model Info:**
- Parameters: 12B
- Size: 8.1 GB
- Notes: Larger gemma3 — **96% classification with zero-shot binary**

#### Classification Results

**Summary:**
- **Binary accuracy: 77/80 (96%)** — BEST for this model, well-calibrated
- Few-shot accuracy: 76/80 (95%) — examples don't help
- Multi-shot accuracy: 76/80 (95%) — same as few-shot
- Ternary accuracy: 76/80 (95%) — same as few-shot
- JSON accuracy: 74/80 (92%) — slowest approach
- Avg latency: **10.5s (binary)**, 11.3s (few-shot), 16.6s (multi-shot), 13.0s (ternary), 27.0s (JSON)

**Observations:**
- **Zero-shot binary is best** — opposite of gemma3:4b where few-shot won
- All example-based approaches (few-shot, multi-shot) perform identically at 95%
- Larger model doesn't benefit from examples — already "understands" task well
- ~3x slower than gemma3:4b

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | Failed | 0% | timeout |
| Responsive example | Failed | 0% | timeout |
| Non-responsive example | Structure complete | **100%** | 114.5 |

**Observations:**
- Very slow on generation (114s for negative example)
- Paraphrase and positive example likely timed out
- Only negative example generation completed successfully
- Not recommended for generation tasks due to speed

#### Extraction Results

**Summary:**
- Evidence quote accuracy: **96%** ⭐ (BEST of all models!)
- Search term extraction: ✓ @ 57s
- Much better extraction than gemma3:4b (16%)

**Observations:**
- Excellent at verbatim quote extraction — 96% accuracy!
- Best extraction model tested so far
- Slow but accurate

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| **96% classification accuracy** | 3x slower than gemma3:4b |
| **96% quote extraction** ⭐ | Generation tasks timeout |
| Zero-shot works best | Too slow for batch generation |

**Recommendation:** BEST FOR EXTRACTION. Use `classification_binary` for 96% classification at 10.5s/doc. Outstanding 96% quote extraction accuracy — best of all models tested. Too slow for generation tasks. For classification-only workloads where latency matters less, this is a strong choice.

---

### gemma2:2b

**Status:** Complete ⭐

**Test Date:** 2025-12-30

**Model Info:**
- Parameters: 2B
- Size: 1.6 GB
- Notes: **Best speed/accuracy tradeoff discovered!** 90% @ 2.2s with ternary

#### Classification Results

**Summary:**
- Binary accuracy: 12/20 (60%) — severe NO bias (0.10 predicted)
- Few-shot accuracy: 13/20 (65%) — slight improvement, 10% parse errors
- Multi-shot accuracy: 16/20 (80%) — big jump! Annotated examples help
- **Ternary accuracy: 18/20 (90%)** ⭐ — BEST, fast (2.2s), well-calibrated
- JSON accuracy: 10/20 (50%) — 100% valid JSON but extreme YES bias (1.0 predicted)
- Avg latency: 1.9s (binary), 6.5s (few-shot), 5.9s (multi-shot), **2.2s (ternary)**, 6.0s (JSON)

**Observations:**
- **Ternary format unlocks this model** — YES/NO/MAYBE + confidence works perfectly
- Multi-shot (6 annotated examples) helps significantly (+15% over few-shot)
- Binary has extreme NO bias; JSON has extreme YES bias — avoid both
- Very fast across all tasks

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | 0/5 generated | 0% | 24.6 |
| Responsive example | Missing From/To | 0% (partial) | 12.4 |
| Non-responsive example | Structure complete | **100%** | 13.5 |

**Observations:**
- Failed paraphrase generation — didn't produce numbered format
- Positive examples missing From/To fields (has Subject + body)
- Negative examples work perfectly — 100% structure compliance
- Not recommended for generation tasks

#### Extraction Results

**Summary:**
- Evidence quotes found: 0.6 average per document
- Quote accuracy: **41%** (best so far!)
- "No relevant content" responses: 90% (very conservative)
- Search term extraction: Failed (0 terms, wrong format)

**Observations:**
- Very conservative on evidence extraction — says "no content" 90% of time
- But when it does extract, quotes are more accurate than other models (41%)
- Search term extraction failed format compliance

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| **90% classification @ 2.2s** ⭐ | Failed paraphrase generation |
| Best speed/accuracy tradeoff | Partial example generation |
| 41% quote accuracy (best) | Search term format failed |
| Multi-shot helps (+15%) | Extreme bias on binary/JSON |

**Recommendation:** BEST FOR SPEED+ACCURACY on classification. Use `classification_ternary` to achieve 90% accuracy at just 2.2s/doc — 20x faster than qwen3:8b with only 5pt accuracy drop. Avoid binary (NO bias) and JSON (YES bias). Not suitable for generation tasks.

---

### phi4-mini:3.8b

**Status:** Complete

**Test Date:** 2025-12-30

**Model Info:**
- Parameters: 3.8B
- Size: 2.5 GB
- Notes: Microsoft's efficient model — **well-rounded performer**

#### Classification Results

**Summary:**
- Binary accuracy: 66/80 (82%) — baseline
- Few-shot accuracy: 67/80 (84%) — slight improvement
- **Multi-shot accuracy: 72/80 (90%)** — BEST, big jump with examples
- Ternary accuracy: 71/80 (89%) — good
- JSON accuracy: 71/80 (89%) — good
- Avg latency: 9.0s (binary), 7.5s (few-shot), **6.7s (multi-shot)**, 5.2s (ternary), 7.3s (JSON)

**Observations:**
- **Multi-shot (6 examples) is best** — +8% over binary/few-shot
- Model benefits from more examples unlike qwen/gemma3
- Ternary and JSON both work well (89%)
- Consistent, well-calibrated across approaches

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | **5/5 generated** | **100%** | 27.1 |
| Responsive example | Missing structure | 0% | 24.2 |
| Non-responsive example | Structure complete | **100%** | 16.8 |

**Observations:**
- **Excellent paraphrase generation** — 100% success, all 5 paraphrases
- Positive example generation fails (missing email structure)
- Negative example generation works perfectly

#### Extraction Results

**Summary:**
- Evidence quote accuracy: **76%** (second best after gemma3:12b!)
- Search term extraction: ✓ @ 9.4s
- Fast extraction (4.4s avg)

**Observations:**
- Excellent extraction accuracy — 76% quotes verbatim
- Second only to gemma3:12b (96%) but much faster
- Good balance of speed and accuracy

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| **90% classification with multi-shot** | Positive example generation fails |
| **76% quote extraction** (2nd best) | Not as good as gemma3:4b for classification |
| **100% paraphrase generation** | |
| Well-rounded across tasks | |
| Fast (5-9s per task) | |

**Recommendation:** EXCELLENT WELL-ROUNDED MODEL. Use `classification_multi_shot` for 90% accuracy. Outstanding for extraction (76% — second best) and paraphrase generation (100%). A strong alternative if gemma3:4b is unavailable.

---

### phi4-mini-reasoning:3.8b

**Status:** In Progress (NOT RECOMMENDED)

**Test Date:** 2025-12-30

**Model Info:**
- Parameters: 3.8B
- Size: 3.2 GB
- Notes: Reasoning-focused variant — **NOT suitable for this task**

#### Classification Results

**Summary:**
- Binary accuracy: 37/80 (**46%**) — worse than coin flip!
- Avg latency: **46.47s** (5x slower than regular phi4-mini)
- Other tasks: Not tested due to poor performance

**Observations:**
- **Reasoning models don't work well** for simple classification
- Extended "thinking" output interferes with YES/NO parsing
- 5x slower than regular phi4-mini for worse results
- Same problem as deepseek-r1 family

#### Generation Results

*(Not tested — model unsuitable)*

#### Extraction Results

*(Not tested — model unsuitable)*

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| (none for this task) | 46% accuracy (worse than random) |
| | 5x slower than regular phi4-mini |
| | Reasoning tokens cause parse errors |

**Recommendation:** NOT RECOMMENDED. Reasoning models produce extended "thinking" output that interferes with structured output tasks. Use regular phi4-mini:3.8b instead — it's faster and far more accurate.

---

### phi3:mini

**Status:** In Progress (classification partial)

**Test Date:** 2025-12-30

**Model Info:**
- Parameters: 3B
- Size: 2.2 GB
- Notes: Previous gen — **weaker than phi4-mini**

#### Classification Results

**Summary (partial):**
- Binary accuracy: 48/80 (60%) — poor baseline
- Few-shot accuracy: 49/80 (61%) — minimal improvement
- **Multi-shot accuracy: 57/80 (71%)** — BEST so far, +11% over binary
- Ternary accuracy: — (pending)
- JSON accuracy: — (pending)
- Avg latency: 15.0s (binary), 14.2s (few-shot), 22.9s (multi-shot)

**Observations:**
- Much weaker than phi4-mini across all approaches
- Multi-shot helps (+11%) but still only reaches 71%
- Slower than phi4-mini despite being smaller
- Not competitive with other models tested

#### Generation Results

*(Testing pending)*

#### Extraction Results

*(Testing pending)*

#### Overall Assessment (Partial)

| Strength | Weakness |
|----------|----------|
| Multi-shot helps (+11%) | 71% max accuracy (poor) |
| | Slower than phi4-mini |
| | Far behind gemma3/phi4-mini |

**Recommendation (preliminary):** NOT RECOMMENDED. Even with multi-shot, only reaches 71% — far below phi4-mini (90%) and gemma3:4b (100%). Use phi4-mini:3.8b instead.

---

### granite3.3:2b

**Status:** Complete

**Test Date:** 2026-01-06

**Model Info:**
- Parameters: 2B
- Size: 1.5 GB
- Notes: IBM model — **solid performer with perfect JSON compliance**

#### Classification Results

**Summary:**
- **Binary accuracy: 34/40 (85%)** — BEST for this model, well-calibrated (0.45 pred)
- Few-shot accuracy: 24/40 (60%) — examples hurt! Severe NO bias (0.10 pred)
- Multi-shot accuracy: 31/40 (77.5%) — better than few-shot but below binary
- Ternary accuracy: 28/40 (70%) — low confidence scores (20.4 avg)
- **JSON accuracy: 34/40 (85%)** — **100% valid JSON!** Tied with binary
- Avg latency: 11.8s (binary), 11.6s (few-shot), 15.5s (multi-shot), 10.9s (ternary), 16.4s (JSON)

**Observations:**
- **Zero-shot binary is best** — same pattern as qwen3:8b, ministral-3:3b
- Few-shot causes severe NO bias (predicted 0.10 vs expected 0.50)
- **Perfect JSON compliance** — 100% valid JSON with all required fields
- Low ternary confidence (20.4 avg) suggests model is uncertain
- Moderate latency (~12s) — slower than gemma models but reasonable

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | **5/5 generated** | **100%** | 43.4 |
| Responsive example | Structure complete | **100%** | 28.9 |
| Non-responsive example | Structure complete | **100%** | 25.3 |

**Observations:**
- **Excellent generation across all tasks** — 100% success rate
- All 5 paraphrases generated (9.2% diversity)
- Both positive and negative examples have complete email structure
- Slower on generation than classification (~25-43s vs ~12s)

#### Extraction Results

**Summary:**
- Evidence quote accuracy: **25%** @ 13.4s
- Search term extraction: 13 terms, 100% format compliance @ 19.9s
- "No relevant content" responses: 70% (very conservative)

**Observations:**
- Very conservative on evidence extraction — says "no content" 70% of time
- When it does extract, moderate accuracy (25%)
- Good search term extraction format compliance
- Not as strong as gemma3:12b (96%) or ministral-3:3b (96%) for extraction

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| **85% classification (binary/JSON)** | Below top performers (100%) |
| **100% JSON compliance** ⭐ | Few-shot causes severe bias |
| **100% generation success** | Conservative extraction (25%) |
| All tasks complete (no timeouts) | Moderate latency (~12s) |

**Recommendation:** SOLID ALTERNATIVE. Use `classification_binary` or `json_output` for 85% accuracy. Outstanding JSON compliance (100% valid) makes it ideal for structured output pipelines. Excellent generation (100% on all tasks). However, classification accuracy (85%) doesn't match top performers like gemma3:4b (100%) or ministral-3:3b (100%). Choose granite if you need perfect JSON compliance.

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

**Status:** Complete ⭐⭐

**Test Date:** 2026-01-05

**Model Info:**
- Parameters: 3B
- Size: 3.0 GB
- Notes: **EXCELLENT all-rounder — 100% ternary classification + 96% extraction!**

#### Classification Results

**Summary:**
- Binary accuracy: 18/20 (90%) — good baseline, well-calibrated
- Few-shot accuracy: 16/20 (80%) — examples hurt (-10%)
- Multi-shot accuracy: 11/20 (55%) — examples hurt severely (-35%)
- **Ternary accuracy: 20/20 (100%)** ⭐⭐ — PERFECT, best approach for this model
- JSON accuracy: 19/20 (95%) — excellent structured output
- Avg latency: 9.6s (binary), 21s (few-shot), 26s (multi-shot), **11.3s (ternary)**, 18.9s (JSON)

**Observations:**
- **Ternary format unlocks perfect classification** — YES/NO/MAYBE + confidence
- Zero-shot dramatically outperforms few-shot/multi-shot (same pattern as qwen3:8b)
- Examples cause confusion and degradation (-10% few-shot, -35% multi-shot)
- Second model to achieve 100% (after gemma3:4b)
- Ternary is both fastest and most accurate

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | **5/5 generated** | **100%** | 43.7 |
| Responsive example | Structure complete | **100%** | 81.9 |
| Non-responsive example | Structure complete | **100%** | 29.6 |

**Observations:**
- Excellent generation across all tasks — 100% success rate
- All 5 paraphrases generated successfully
- Both positive and negative email examples have complete structure
- Faster on negative examples (30s vs 82s for positive)

#### Extraction Results

**Summary:**
- Evidence quote accuracy: **96%** ⭐ (ties gemma3:12b for BEST!)
- Search term extraction: 100% format compliance @ 50.2s
- Keywords per doc: 23 average (comprehensive)
- All terms annotated (100%)

**Observations:**
- Outstanding quote extraction — 96% verbatim accuracy!
- Ties gemma3:12b as best extraction model
- Comprehensive keyword extraction (23 terms vs typical 9-14)
- All extracted terms have annotations (explanations)

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| **100% classification (ternary)** ⭐⭐ | Examples hurt performance (-35% multi-shot) |
| **96% quote extraction** ⭐ | Slower than gemma3:4b (~11s vs 3.3s) |
| 100% generation success | |
| Well-rounded excellence | |
| Comprehensive keyword extraction | |

**Recommendation:** EXCELLENT ALL-ROUNDER. Use `classification_ternary` for perfect 100% classification at 11.3s/doc. Tied for best extraction (96% with gemma3:12b). Perfect generation success (100% on all tasks). This model excels at everything — a true generalist. Comparable to gemma3:4b but 3x slower; choose based on latency requirements.

---

### ministral-3:8b

**Status:** Complete (partial - generation timeouts)

**Test Date:** 2026-01-06

**Model Info:**
- Parameters: 8B
- Size: 6.0 GB
- Notes: **NOT RECOMMENDED - 3b is better (faster, better extraction, completes all tasks)**

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

**Status:** Partial (classification only — interrupted)

**Test Date:** 2026-01-06

**Model Info:**
- Parameters: 20B
- Size: 13 GB
- Notes: Largest model tested — **underperforms smaller models despite size**

#### Classification Results

**Summary (PARTIAL — test interrupted after classification tasks):**
- Binary accuracy: 34/40 (85%) — same as granite3.3:2b (1.5GB!)
- Few-shot accuracy: 34/40 (85%) — examples don't help
- Multi-shot accuracy: 32/40 (80%) — examples actually hurt
- **Ternary accuracy: 35/40 (87.5%)** — BEST for this model
- JSON accuracy: 32/40 (80%) — below binary/ternary
- Avg latency: **~50-60s per document** (extremely slow on CPU)

**Observations:**
- **Ternary is the best approach** — 87.5% accuracy
- Examples don't help (few-shot) or hurt (multi-shot) — same pattern as qwen3:8b
- **Dramatically underperforms smaller models**:
  - gemma3:4b (3.3GB): 100% — 15pts better, 15x faster
  - ministral-3:3b (3.0GB): 100% — 15pts better, 5x faster
  - qwen3:8b (5.2GB): 95% — 10pts better, similar speed
- At ~1 min/doc, processing 339 documents would take ~6 hours
- Test interrupted before generation/extraction tasks

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

*(Not tested — evaluation interrupted)*

#### Extraction Results

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

*(Not tested — evaluation interrupted)*

#### Overall Assessment (Partial)

| Strength | Weakness |
|----------|----------|
| 87.5% ternary accuracy | **15pts below gemma3:4b (100%)** |
| Large model capacity | **Extremely slow (~60s/doc)** |
| | Examples hurt performance |
| | Not practical for batch processing |

**Recommendation:** **NOT RECOMMENDED for CPU deployment**. Despite being the largest model (20B params, 13GB), gpt-oss:20b achieves only 87.5% accuracy — significantly below gemma3:4b (100%) and ministral-3:3b (100%) which are 4-6x smaller and 5-15x faster. This is a clear case of "bigger ≠ better" for this task. The model may perform better on GPU, but on CPU it's impractical. Use gemma3:4b with few-shot for better accuracy and speed.

**Key Insight:** Model size does not correlate with CPRA classification accuracy. Smaller, well-tuned models (gemma3:4b, ministral-3:3b) dramatically outperform this 20B parameter model. The task benefits more from prompt-following ability than raw model capacity.

---

### olmo-3:7b

**Status:** Partial (classification only — interrupted due to catastrophic failure)

**Test Date:** 2026-01-08

**Model Info:**
- Parameters: 7B
- Size: 4.9 GB
- Notes: Allen AI open model — **CATASTROPHIC FAILURE on few-shot/multi-shot prompts**

#### Classification Results

**Summary (PARTIAL — test interrupted after observing catastrophic failure):**
- **Binary accuracy: 29/40 (72.5%)** — best for this model, but still 28pts below leaders
- **Few-shot accuracy: 2/40 (5%)** — CATASTROPHIC FAILURE
- **Multi-shot accuracy: 4/40 (10%)** — also catastrophic
- Ternary accuracy: — (not tested)
- JSON accuracy: — (not tested)
- Avg latency: ~22s per document

**Observations:**
- **Few-shot examples completely break this model** — drops from 72% to 5%
- Worst few-shot result of any model tested (5%)
- Multi-shot (10%) almost as bad as few-shot (5%)
- Even binary (72.5%) is well below acceptable threshold
- This is the opposite pattern of gemma3:4b where few-shot achieves 100%
- Model may be fundamentally incompatible with in-context learning for this task

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | — | — | — |
| Responsive example | — | — | — |
| Non-responsive example | — | — | — |

*(Not tested — evaluation interrupted)*

#### Extraction Results

**Summary:**
- Quote accuracy: —%
- Keyword relevance: —/5

*(Not tested — evaluation interrupted)*

#### Overall Assessment (Partial)

| Strength | Weakness |
|----------|----------|
| (none identified) | **5% few-shot accuracy** (worst of all models) |
| | **10% multi-shot accuracy** |
| | 72.5% binary (28pts below leaders) |
| | Examples cause catastrophic degradation |

**Recommendation:** **NOT RECOMMENDED**. olmo-3:7b exhibits catastrophic failure when given few-shot examples (5% accuracy) — the worst result of any model tested. Even zero-shot binary achieves only 72.5%, far below gemma3:4b (100%) and ministral-3:3b (100%). This model appears fundamentally incompatible with in-context learning for CPRA classification. Test was interrupted as there was no point continuing after observing these results.

---

### functiongemma:270m

**Status:** Complete ❌

**Test Date:** 2026-01-06

**Model Info:**
- Parameters: 270M
- Size: 300 MB
- Notes: **NOT SUITABLE** — Function calling specialist, not designed for text tasks

#### Classification Results

**Summary:**
- Binary accuracy: 23/40 (57.5%) — barely above random (50%)
- Few-shot accuracy: 20/40 (50%) — exactly random
- Multi-shot accuracy: 20/40 (50%) — exactly random
- Ternary accuracy: 20/40 (50%) — random guessing
- JSON accuracy: 15/40 (37.5%) — can produce JSON structure but wrong answers
- Avg latency: 1.25s (fastest of any model tested)

**Observations:**
- **Predicts NO for almost everything** — avg_predicted: 0.00-0.12 vs avg_expected: 0.50
- Model actively **refuses many requests** with "I cannot assist with..."
- Specialized for function calling (extracting structured parameters), not text classification
- Very fast but useless for this task

**Sample Responses:**

Binary Classification (expected YES):
```
"NO"
"I cannot assist with drafting or evaluating the suitability of documents..."
```

JSON Output (malformed):
```json
{"category": "no", "reasoning": "...", "reasoning": "..."}}$$
```

#### Generation Results

| Task | Output Quality | Constraints Met | Latency (s) |
|------|----------------|-----------------|-------------|
| Paraphrases | 0/5 generated | 0% | 0.70 |
| Responsive example | Missing structure | 0% | 0.79 |
| Non-responsive example | Missing structure | 0% | 0.83 |

**Observations:**
- **Complete failure on all generation tasks**
- Model refuses: "I am sorry, but I cannot fulfill this request. My current capabilities are limited to assisting with administrative tasks..."
- Produces ~43-48 words but no proper email structure (no From/To/Subject)
- Function calling models are not designed for content generation

#### Extraction Results

**Summary:**
- Evidence quotes found: 0 average
- Quote accuracy: 0%
- Search term extraction: 0/5 (0 terms extracted)
- Avg latency: ~0.9s

**Observations:**
- Complete failure on extraction tasks
- Model doesn't understand the task — produces empty or irrelevant output
- Not designed for text analysis

#### Overall Assessment

| Strength | Weakness |
|----------|----------|
| Very fast (1.25s avg) | 50% accuracy (random chance) |
| | Refuses most requests |
| | Malformed JSON output |
| | 0% generation/extraction success |
| | Wrong model type for this task |

**Recommendation:** **NOT SUITABLE**. functiongemma is a specialized model for **function/tool calling** (extracting structured parameters from natural language, like Claude's tool use). It actively refuses text classification and generation tasks. Despite being the fastest model tested, its accuracy is at or below random chance. Do not use for CPRA document classification or any text analysis tasks.

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
| 2025-12-30 | qwen3:0.6b | All 10 | ~3 min | Retested with revised prompts. Few-shot still best (85%). Generation improved. |
| 2025-12-30 | qwen3:8b | All 10 | ~30 min | **95% classification** ⭐ with zero-shot. Few-shot hurts (60%). Timeouts on gen/extraction. |
| 2025-12-30 | gemma2:2b | All 10 | ~15 min | **90% @ 2.2s with ternary** ⭐ Best speed/accuracy! Multi-shot helps (+15%). Generation weak. |
| 2025-12-30 | gemma3:4b | All 10 | ~25 min | **100% classification with few-shot** ⭐⭐ PERFECT! 100% email gen. Poor extraction (16%). |
| 2025-12-30 | gemma3:12b | All 10 | ~40 min | 96% classification (binary best). **96% quote extraction** ⭐ BEST! Generation timeouts. |
| 2025-12-30 | phi4-mini:3.8b | All 10 | ~20 min | 90% classification (multi-shot). **76% extraction**, **100% paraphrase**. Well-rounded! |
| 2025-12-30 | phi4-mini-reasoning:3.8b | Binary only | ~60 min | **NOT REC**: 46% accuracy (worse than random), 5x slower. Reasoning breaks parsing. |
| 2025-12-30 | phi3:mini | Classification (partial) | ~15 min | **NOT REC**: 71% max (multi-shot). Weaker and slower than phi4-mini. |
| 2026-01-06 | functiongemma:270m | All 10 | ~2 min | **NOT REC**: 50% accuracy (random), refuses tasks. Function calling model, not for text. |
| 2026-01-06 | granite3.3:2b | All 10 | ~15 min | 85% classification (binary/JSON); **100% JSON compliance** ⭐; 100% generation; 25% extraction. |
| 2026-01-06 | gpt-oss:20b | Classification only | ~2 hrs (interrupted) | **87.5% ternary** (best); extremely slow (~55s/doc); **underperforms smaller models**. |
| 2026-01-08 | olmo-3:7b | Classification partial | ~30 min (interrupted) | **5% few-shot** (CATASTROPHIC); 72.5% binary; worst few-shot result of all models. |

---

## Conclusions

### Testing Summary

**16 models tested** across classification, generation, and extraction tasks. Two clear winners emerged for CPRA document classification.

### Key Findings

#### 1. Optimal Models Identified

| Use Case | Model | Accuracy | Latency | Prompt Style |
|----------|-------|----------|---------|--------------|
| **Classification (fastest)** | gemma3:4b | **100%** | 3.3s | Few-shot |
| **Classification (best all-rounder)** | ministral-3:3b | **100%** | 11.3s | Ternary |
| **Extraction (quote accuracy)** | gemma3:12b / ministral-3:3b | **96%** | 15-19s | Zero-shot |
| **Generation (paraphrases)** | phi4-mini:3.8b | **100%** | 22s | — |
| **Speed priority** | gemma2:2b | 90% | **2.2s** | Ternary |

#### 2. Bigger ≠ Better

Model size does **not** correlate with CPRA classification accuracy:

| Model | Size | Best Accuracy |
|-------|------|---------------|
| gemma3:4b | 3.3 GB | **100%** ⭐ |
| ministral-3:3b | 3.0 GB | **100%** ⭐ |
| gpt-oss:20b | 13 GB | 87.5% |
| olmo-3:7b | 4.9 GB | 72.5% |

The largest model tested (gpt-oss:20b) scored 12.5 points **below** models 4x smaller.

#### 3. Prompt Strategy is Model-Dependent

Different models respond dramatically differently to the same prompting approach:

| Prompt Style | Best Model | Worst Model |
|--------------|------------|-------------|
| Few-shot | gemma3:4b (100%) | olmo-3:7b (**5%**) |
| Multi-shot | phi4-mini (90%) | ministral-3:3b (55%) |
| Zero-shot binary | qwen3:8b (95%) | gemma2:2b (60%) |
| Ternary | ministral-3:3b (100%) | gemma3:4b (70%) |

**Critical insight:** Always test multiple prompt strategies per model. A model that excels with one approach may fail catastrophically with another.

#### 4. Models to Avoid

| Model | Issue |
|-------|-------|
| Reasoning models (phi4-mini-reasoning, deepseek-r1) | Extended thinking breaks output parsing |
| Function calling models (functiongemma) | Wrong task type — refuses text classification |
| olmo-3:7b | Catastrophic few-shot failure (5%) |
| gpt-oss:20b | Too slow for CPU, underperforms smaller models |

### Final Recommendations

**For CPRA document classification on CPU:**

1. **Primary choice:** `gemma3:4b` with `classification_few_shot`
   - 100% accuracy, 3.3s/doc, perfect calibration
   - Process 339 docs in ~19 minutes

2. **Alternative (better extraction):** `ministral-3:3b` with `classification_ternary`
   - 100% accuracy, 11.3s/doc
   - Also 96% quote extraction accuracy
   - Process 339 docs in ~64 minutes

3. **Budget/speed priority:** `gemma2:2b` with `classification_ternary`
   - 90% accuracy, 2.2s/doc (fastest)
   - Process 339 docs in ~12 minutes

### Remaining Untested Models

| Model | Priority | Rationale |
|-------|----------|-----------|
| granite3.3:8b | Low | 2b version only reached 85% |
| llama3:8b-instruct-q5_K_M | Low | Have 100% models already |
| ministral-3:14b | Low | 3b already achieves 100% |
| deepseek-r1:8b | Skip | Reasoning models don't work |

**Recommendation:** Testing is effectively complete. We have identified optimal models (gemma3:4b, ministral-3:3b) achieving 100% accuracy. Additional testing unlikely to yield better results.
