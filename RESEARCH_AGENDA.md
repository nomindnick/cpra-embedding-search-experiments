# Research Agenda: CPRA Semantic Search (v2 corpus)

> Last updated: 2025-12-29

This is the "what do we try next?" plan for pushing **precision up** while keeping **recall ≥ 94%** (legal requirement).

---

## Where We Are (Baseline)

**Best single-model baseline:** `all-mpnet-base-v2` (Exp 006) hits **98.71% recall** at threshold 0.30 with **57.74% precision**.

**Cross-encoders didn't help** on the keyword-free v2 corpus (Exp 011–019). The issue appears to be *training mismatch*:
- MS-MARCO/BGE models over-rely on lexical overlap
- NLI models saturate (score everything as "entails")
- STS models expect similar-length inputs
- Paraphrase models fail on query-document pairs

**The takeaway:** The next gains are likely to come from **pipeline-level strategies**, not swapping another off-the-shelf model.

---

## North-Star Metrics

**Primary (compliance-oriented):**

1. **Precision at fixed recall ≥ 94%** — choose the threshold that *just* meets 94% recall, then report precision
2. **False negatives count** — absolute FN matters operationally

**Secondary (review-burden oriented):**

3. **Precision@K / Recall@K** for realistic reviewer cutoffs (K ∈ {50, 100, 200})
4. **By challenge-type recall** — must not regress on: INDIRECT_REFERENCE, TECHNICAL_JARGON, BURIED_IN_THREAD

**Generalization:**

5. **Primary → Validation transfer** — tune on Lead corpus, then report on PFAS corpus with minimal retuning

---

## Design Principles

**Generalization is critical**: All experiments must work for ANY CPRA request, not just "lead contamination." Approaches should:
- Use `request.request_text` and `request.keywords` from corpus `request.json`
- Generate examples/discriminators dynamically based on the request
- Avoid hardcoded keywords, phrases, or domain-specific rules

**Validation strategy**: Test on both corpora:
- Primary corpus (lead contamination) — development
- Validation corpus (PFAS) — confirms generalization

**Two-stage pipeline pattern**: Many approaches follow this structure:
- Stage 1: High-recall candidate generation (embeddings at low threshold)
- Stage 2: Precision filtering/reranking (LLM, ensemble, classifier)

---

## Priority 0: Foundation & Sanity Checks

### EXP-000 — Local LLM Capability Assessment

**Goal:** Identify which local LLMs to use for which tasks in subsequent experiments. Different models excel at different tasks (classification vs generation vs extraction), and latency matters when processing 339+ documents.

**Models to evaluate (available via Ollama):**

| Model | Size | Notes |
|-------|------|-------|
| qwen3:0.6b | 0.6B | Ultra-fast baseline |
| qwen3:1.7b | 1.7B | Small but capable |
| qwen3:8b | 8B | Flagship qwen |
| gemma3:4b | 4B | Google's latest |
| gemma3:12b | 12B | Larger gemma |
| gemma2:2b | 2B | Previous gen |
| phi4-mini:3.8b | 3.8B | Microsoft |
| phi4-mini-reasoning:3.8b | 3.8B | Reasoning variant |
| phi3:mini | 3B | Previous gen |
| granite3.3:2b | 2B | IBM |
| granite3.3:8b | 8B | IBM larger |
| deepseek-r1:1.5b | 1.5B | Reasoning focused |
| deepseek-r1:8b | 8B | Reasoning larger |
| ministral-3:3b | 3B | Mistral small |
| ministral-3:8b | 8B | Mistral medium |
| ministral-3:14b | 14B | Mistral large |
| llama3:8b-instruct-q5_K_M | 8B | Meta |
| gpt-oss:20b | 20B | Largest available |
| olmo-3:7b | 7B | Allen AI open model |
| functiongemma:270m | 270M | Function calling specialist |

**Tasks to evaluate:**

1. **Classification (few-shot)** — ⭐ BEST APPROACH: Few-shot examples + YES/NO output
2. **Classification (binary)** — Zero-shot YES/NO (baseline comparison)
3. **Classification (ternary + confidence)** — Output yes/no/maybe with confidence 0-100
4. **JSON format compliance** — Can it output valid, parseable JSON?
5. **Evidence extraction** — Can it quote verbatim from the document?
6. **Generation (paraphrases)** — Generate 5 diverse paraphrases of a request
7. **Generation (examples)** — Generate realistic responsive/non-responsive emails
8. **Extraction (keywords)** — Extract relevant keywords/entities from text
9. **Latency** — Time per task (critical for corpus-scale processing)

**Early finding:** Few-shot prompting dramatically improves classification (85% vs 60-70% zero-shot on qwen3:0.6b). Use `classification_few_shot` task for all future classification work.

**Evaluation approach:**

1. Select 20 documents from corpus:
   - 5 clear responsive (DIRECT_MATCH)
   - 5 tricky responsive (INDIRECT_REFERENCE, TECHNICAL_JARGON, BURIED_IN_THREAD)
   - 5 tricky non-responsive (KEYWORD_FALSE_POSITIVE, ADJACENT_TOPIC)
   - 5 clear non-responsive (TRUE_NEGATIVE)

2. Run each model on each task type, measuring:
   - **Accuracy** — correct classification rate
   - **Format compliance** — valid JSON rate
   - **Extraction quality** — quotes actually appear in source
   - **Generation diversity** — unique paraphrases, realistic examples
   - **Latency** — seconds per request

3. Produce recommendation matrix:
   - Best models for classification tasks
   - Best models for generation tasks
   - Best models for extraction tasks
   - Speed vs accuracy tradeoffs

**Expected outcome:** Identify 3-5 models to focus on for subsequent experiments, with clear guidance on which to use for which task type.

**Status:** In Progress — qwen3:0.6b (85% few-shot), deepseek-r1:1.5b (not recommended) complete. See `LLM_Capability_Assessment.md` for detailed results.

---

### EXP-020 — Validation Corpus Sanity Check

**Hypothesis:** The "winner" on Lead remains strong on PFAS; if not, we may be overfitting.

**Method:**
1. Run experiments 003–010 (top bi-encoders) on `corpus/validation` with the same thresholds used on primary
2. Produce:
   - Precision/recall/F1/MAP
   - Recall@K curves (K=25/50; validation set is small)
   - Challenge-type breakdown (responsive vs non-responsive)

**Decision rule:**
- If `all-mpnet-base-v2` is still top-1 or top-2 on *precision at ≥94% recall*, keep it as the default embedder
- If a different model generalizes better, promote that model

**Status:** Pending

---

## Priority 1: Query Expansion (LLM-Assisted Multi-Query Retrieval)

**Goal:** Increase recall and/or create a better ranking distribution so we can raise the threshold without losing recall.

### EXP-021 — LLM Paraphrase Expansion (Multi-Query "OR")

**Hypothesis:** Multiple semantically-different phrasings reduce vocabulary mismatch and boost recall at higher thresholds.

**Method:**
1. Use local LLM to generate N paraphrases of the CPRA request (N ∈ {3, 5, 10})
2. Embed: original request + paraphrases
3. Score each email by aggregation across queries

**Variants:**
- 021a: max cosine across all query embeddings
- 021b: average cosine
- 021c: RRF merge of per-query ranked lists

**Prompt:**
```
Given this CPRA request:
{request.request_text}

Generate {N} semantically different paraphrases of this request.
Each paraphrase should:
- Capture the same information need
- Use different vocabulary and phrasing
- Emphasize different aspects of what's being requested

Output only the paraphrases, one per line.
```

**Risk:** Query drift — paraphrases that broaden scope could increase false positives.

**What to measure:**
- Does precision at ≥94% recall increase vs baseline?
- Per-type recall: does this help INDIRECT_REFERENCE and TECHNICAL_JARGON without inflating ADJACENT_TOPIC?

**Status:** Pending

---

### EXP-022 — Facet Decomposition (Sub-Questions)

**Hypothesis:** Breaking a request into facets captures "buried" and indirect references better than a single embedding.

**Method:**
1. LLM converts request into 5–10 facet queries
2. Retrieve per facet; merge with RRF
3. Optionally weight facets

**Prompt:**
```
Given this CPRA request:
{request.request_text}

Break this down into 5-10 specific facets or sub-questions that together cover all aspects of what's being requested. Consider:
- Different types of documents that might be responsive
- Different activities or events that might be relevant
- Different terminology that might be used
- Different timeframes or stages

Output each facet as a search query, one per line.
```

**Variants:**
- 022a: equal weights for all facets
- 022b: LLM outputs weights (0–1) per facet based on importance

**Status:** Pending

---

### EXP-023 — LLM Keyword/Entity Expansion for Hybrid Search

**Hypothesis:** A *smart* lexical query boosts precision on obvious matches while embeddings preserve recall.

**Method:**
1. Local LLM extracts from the request:
   - Keywords/phrases (including multi-word phrases)
   - Acronyms and jargon relevant to the domain
   - Entity types (agencies, facilities, projects, etc.)
2. Run lexical retrieval (BM25 or simple OR keyword matching) using the expanded set
3. Combine lexical + embedding rankings with RRF

**Prompt:**
```
Given this CPRA request:
{request.request_text}

Extract the following to help find relevant documents:

KEYWORDS: Important single words and phrases
ACRONYMS: Abbreviations and technical acronyms that might appear
ENTITIES: Types of organizations, facilities, people, or projects mentioned
MUST_HAVE: Terms that MUST appear for a document to be relevant
NICE_TO_HAVE: Terms that suggest relevance but aren't required

Output in this format:
KEYWORDS: term1, term2, ...
ACRONYMS: ABC, XYZ, ...
ENTITIES: [entity types]
MUST_HAVE: [required terms]
NICE_TO_HAVE: [optional terms]
```

**Variants:**
- 023a: embeddings-only baseline
- 023b: BM25 + embeddings (RRF)
- 023c: BM25 filter then embeddings rerank (careful: may harm recall)

**Status:** Pending

---

### EXP-024 — Pseudo-Relevance Feedback (Top Results → Expansion)

**Hypothesis:** The top-ranked documents contain "missing vocabulary" (jargon, project names, facility names) that the request doesn't include. Feeding that back improves both recall and ranking.

**Method:**
1. Run baseline retrieval (mpnet) and take top K documents (K ∈ {10, 25, 50})
2. Ask local LLM to extract from those documents:
   - Key terms / acronyms
   - Named entities (people, facilities, vendors, labs)
   - Concrete artifacts ("sample results", "lab report", "fixture replacement")
3. Turn that into:
   - Expanded keyword query for BM25, and/or
   - 3–10 expansion queries to embed (multi-query retrieval)
4. Re-run retrieval and merge with the baseline list (RRF)

**Prompt:**
```
Here are {K} documents that appear relevant to this CPRA request:
{request.request_text}

---
DOCUMENTS:
{top_k_documents}
---

Extract vocabulary from these documents that would help find MORE relevant documents:
- Key terms and phrases used to discuss this topic
- Acronyms and abbreviations
- Names of specific facilities, projects, programs, or people
- Types of artifacts mentioned (reports, results, notices, etc.)

Only include terms that appear in 2+ documents (to avoid noise).
Mark each term as MUST_HAVE or NICE_TO_HAVE.
```

**Variants:**
- 024a: LLM keyword extraction only → BM25 + embeddings fusion
- 024b: LLM generates 5 expansion queries → embeddings-only fusion
- 024c: both (BM25 + multi-query embeddings)

**Risk:** Query drift (top docs may reflect adjacent topic). Mitigation: only extract terms appearing in ≥2 of top K docs.

**Status:** Pending

---

## Priority 2: Positive/Negative Prototypes (Contrastive Scoring)

This is directly aligned with the original design in SPEC: generate **positive candidates** and **negative candidates** and use them as "anchors."

### EXP-025 — LLM-Generated Positive & Negative Pseudo-Emails

**Hypothesis:** Modeling "what responsive looks like" *and* "what a red herring looks like" will reduce false positives from polysemy/adjacent topics.

**Method:**
1. LLM generates:
   - P positive examples (P ∈ {3, 5, 10})
   - N negative examples (N ∈ {3, 5, 10})
2. Embed them with the same embedding model as documents
3. Score each email with contrastive formula:
   - **Score A:** `max_sim(email, positives)`
   - **Score B:** `max_sim(email, positives) - λ * max_sim(email, negatives)`
   - **Score C:** `avg_sim(email, positives) - λ * avg_sim(email, negatives)`
4. Tune λ on primary corpus; evaluate on validation

**Positive generation prompt:**
```
Given this CPRA request:
{request.request_text}

Generate a realistic email that WOULD be responsive to this request.
The email should:
- Discuss the subject matter substantively
- NOT use these exact keywords: {request.keywords}
- Include realistic email metadata (sender, recipient, subject)
- Be 100-300 words

Generate {P} different examples covering different types of responsive content.
```

**Negative generation prompt:**
```
Given this CPRA request:
{request.request_text}

Generate a realistic email that is RELATED TO but NOT responsive to this request.
It should be a plausible false positive - discussing adjacent topics that might
seem relevant but don't actually address the request subject matter.

Specifically target these false positive patterns:
- Keywords used in unrelated contexts (e.g., "lead" as verb meaning "to guide")
- Same domain but different specific topic
- Administrative/procedural content tangentially related

Generate {N} different examples of non-responsive but plausibly-confused content.
```

**Expected win:** Precision bump without sacrificing recall.

**Status:** Pending

---

### EXP-026 — Prototype Centroids from Labeled Data (Upper Bound)

**Hypothesis:** If prototypes work, using *real* positives/negatives should work even better than LLM-generated text.

**Method:**
1. Build centroid vectors from ground truth:
   - `c_pos = mean(embeddings of known responsive docs)`
   - `c_neg = mean(embeddings of known non-responsive docs)` (or of specific negative classes like KEYWORD_FALSE_POSITIVE)
2. Score: `cos(d, c_pos) - λ * cos(d, c_neg)`

**Why do this:** Gives an "upper bound" on what prototype scoring could achieve without LLM variability.

**Variants:**
- 026a: All non-responsive as negatives
- 026b: Only KEYWORD_FALSE_POSITIVE as negatives
- 026c: Only ADJACENT_TOPIC as negatives

**Status:** Pending

---

## Priority 3: Ensembles & Disagreement Strategies

Ensembles can help in two ways:
- **RRF improves ranking** by smoothing model quirks
- **Disagreement identifies "hard" docs** where expensive verification is worth it

### EXP-027 — Reciprocal Rank Fusion Across Top Bi-Encoders

**Hypothesis:** Combining diverse embedders increases precision at the compliance threshold by stabilizing ranking.

**Candidates:**
- `all-mpnet-base-v2` (best precision@high recall)
- `embeddinggemma` (strong MAP, 100% recall at low threshold)
- `bge-large-en-v1.5`
- `jina-embeddings-v3`
- `qwen3-embedding-0.6b` (precision specialist)

**Method:**
1. Run each model, get ranked list (top_k large, e.g., 250–339)
2. Fuse with RRF: `score(d) = Σ 1/(k + rank_m(d))` where k=60 (standard)

**Variants:**
- 027a: 2-model fusion (mpnet + embeddinggemma)
- 027b: 3–4 model fusion
- 027c: add BM25 as another "model" into RRF

**Status:** Pending

---

### EXP-028 — Precision Specialist Rerank (mpnet → qwen3)

**Hypothesis:** Qwen3's high precision (but lower recall) makes it a good *second-stage* scorer if mpnet supplies the high-recall candidate set.

**Method:**
1. Stage 1: retrieve candidates with mpnet (top_k large, e.g., 250)
2. Stage 2: compute qwen3 similarity for the same candidates
3. Final score:
   - 028a: `score = α * mpnet_sim + (1-α) * qwen3_sim`
   - 028b: RRF over the two candidate rankings
4. Tune α on primary; evaluate on validation

**What to watch:** If recall drops below 94%, the reranker is too aggressive; expand candidate set or reduce qwen3 weight.

**Status:** Pending

---

### EXP-029 — Disagreement Gating ("Easy vs Hard")

**Hypothesis:** If two models agree a document is responsive, it's probably responsive; if they disagree, that's where precision losses live.

**Method:**
1. Pick two complementary models (mpnet + qwen3, or mpnet + embeddinggemma)
2. Define at threshold that gives each model ≥94% recall:
   - **easy-positive:** rank ≤ K in both models
   - **easy-negative:** rank > K in both models
   - **hard:** everything else (disagreement)
3. Only run expensive verifier (LLM) on the hard set

**Measure:**
- Precision/recall overall
- % of corpus requiring LLM calls (cost proxy)

**Status:** Pending

---

### EXP-030 — Stacked Classifier Over Scores

**Hypothesis:** A tiny supervised model can learn a better decision boundary than any single threshold.

**Method:**
1. Features per email:
   - Cosine scores from 2–4 embedding models
   - BM25 score(s)
   - Basic lexical features (#keyword hits, presence of key phrases)
2. Train logistic regression / linear SVM on primary corpus
3. Evaluate with cross-validation
4. Test on PFAS validation corpus

**Caution:** With only two requests (Lead + PFAS), may overfit. Treat as exploratory.

**Status:** Pending

---

## Priority 4: Local LLM as Verifier

Cross-encoders failed because they were trained for different relevance signals. A local LLM "judge" can be prompted with **CPRA-specific definitions** and required to provide **extractive evidence**.

### EXP-031 — LLM Verifier with Evidence Requirement

**Hypothesis:** LLM verification on a candidate set increases precision while preserving recall if we bias toward "include."

**Note from EXP-000:** Few-shot prompting dramatically improves classification accuracy (85% vs 60-70% zero-shot). Consider using few-shot approach for the verifier prompt instead of zero-shot structured output.

**Method:**

**Stage 1 (high recall):** Retrieve candidate set using mpnet (threshold tuned for ≥98% recall, or top_k=250)

**Stage 2 (LLM judge):** For each candidate email, ask for structured output:

**Prompt:**
```
You are evaluating documents for a California Public Records Act (CPRA) request.

CPRA REQUEST:
{request.request_text}

DOCUMENT TO EVALUATE:
{document.text}

Determine if this document is RESPONSIVE to the CPRA request.
A document is responsive if it contains information that would need to be disclosed.

You MUST provide your analysis in this exact JSON format:
{
  "responsive": "yes" | "no" | "maybe",
  "confidence": 0-100,
  "evidence_quotes": ["quote1", "quote2"],
  "reasoning": "1-3 bullet points explaining why",
  "category": "testing|remediation|communication|infrastructure|unrelated|keyword_false_positive|adjacent_topic"
}

IMPORTANT: If you answer "yes" or "maybe", you MUST include at least one verbatim quote from the document that supports responsiveness. If you cannot quote supporting evidence, answer "no".
```

**Decision rule (recall-biased):**
- 031a: treat **yes OR maybe** as responsive
- 031b: include if confidence ≥ X (tune X)
- 031c: include if evidence_quotes non-empty (strong anti-hallucination guardrail)

**What to measure:**
- Does LLM reduce KEYWORD_FALSE_POSITIVE and ADJACENT_TOPIC false positives?
- Which challenge types does it incorrectly reject (risk to recall)?
- Tokens/time per document

**Models to test:** qwen3:8b, gemma3:4b, ministral-3:8b, phi4-mini:3.8b, granite3.3:8b

**Status:** Pending

---

### EXP-032 — LLM Verifier with Few-Shot Examples

**Concept:** Same as 031 but with examples in the prompt.

**Implementation options:**

a) **Use corpus examples:** Sample 3-5 responsive and 3-5 non-responsive documents from ground truth

b) **LLM-generated examples:** Before verification, ask LLM to generate hypothetical examples based on `request.request_text`

c) **Challenge-type coverage:** Select examples covering edge cases:
   - Responsive: indirect references, technical jargon, buried context
   - Non-responsive: keyword false positives, adjacent topics

**Hypothesis:** Few-shot examples improve LLM accuracy on edge cases.

**Status:** Pending

---

### EXP-033 — LLM Query-Aware Compression → Embedding Rerank

**Hypothesis:** Many false positives are driven by noise (signatures, scheduling, boilerplate). Compressing to "only relevant spans" improves embedding scoring.

**Method:**
1. For each email, LLM produces a short "request-relevant extract" (≤ 150–250 tokens)
2. Embed the extract (not the full email) and re-score against the request
3. Compare:
   - Baseline embedding score on raw email
   - Embedding score on compressed extract
   - Hybrid (max of both)

**Prompt:**
```
Given this CPRA request:
{request.request_text}

And this document:
{document.text}

Extract ONLY the portions of this document that are potentially relevant to the CPRA request. If nothing is relevant, output "NOTHING RELEVANT".

Keep your extract to 150-250 tokens maximum. Quote directly from the document.
```

**Note:** This can also help with BURIED_IN_THREAD by extracting the truly relevant earlier content.

**Status:** Pending

---

## Priority 5: Better Document Representations (Threading + Chunking)

### EXP-034 — Chunk-Level Embeddings (Multi-Vector Max Pooling)

**Hypothesis:** Long emails and threads bury relevance; chunking improves recall without broadening too much.

**Method:**
1. Split emails into chunks:
   - Subject line
   - First 200–400 tokens of body
   - Quoted reply blocks (separated)
   - Attachment text (if any)
2. Embed each chunk separately
3. Score email by max similarity across chunks

**Variants:**
- 034a: fixed-size chunks (200 tokens each)
- 034b: structure-aware chunks (headers vs body vs quoted text)

**Status:** Pending

---

### EXP-035 — Thread-Aware Retrieval

**Hypothesis:** "Responsive" content may appear in earlier messages; embedding only the last message loses it.

**Method:**
1. Build thread document by:
   - Concatenating the chain (with separators), or
   - Embedding each message and taking max/mean
2. Evaluate specifically on BURIED_IN_THREAD cases

**Status:** Pending

---

## Priority 6: Domain Adaptation (Fine-Tuning) — Stretch Goals

These require training and are "bigger bites" but could produce the largest step-change.

### EXP-036 — Contrastive Fine-Tuning with Hard Negatives

**Hypothesis:** A small amount of task-specific fine-tuning removes adjacent-topic false positives while keeping semantic matches.

**Method:**
1. Create training triples (q, pos, neg):
   - pos = labeled responsive
   - neg = hard negatives drawn from top-ranked false positives of mpnet
2. Fine-tune a compact model (e.g., MiniLM-based bi-encoder) with contrastive loss
3. Evaluate on PFAS without retraining

**Stretch:** Mix in synthetic pairs from LLM-generated "near misses" to teach nuance.

**Status:** Stretch goal

---

### EXP-037 — Fine-Tuned Reranker on Non-Lexical Relevance

**Hypothesis:** A reranker can work if trained on examples that explicitly *don't* share keywords.

**Method:**
1. Use the v2 corpus + additional synthetic data to train a small cross-encoder/reranker
2. Goal: remove MS-MARCO lexical bias
3. Deploy only on top_k candidates

**Status:** Stretch goal

---

## Implementation Notes

### Add a Generic "Multi-Query" Retrieval Interface

So query expansion, facets, and prototypes all plug into the same mechanism:
- Generate list of query strings (and maybe weights)
- Embed them
- Compute per-doc score aggregation (max/avg/weighted/RRF)

### Add a "Candidate Set" Abstraction

Many approaches require:
- Stage 1: candidate generation (high recall)
- Stage 2: filtering/reranking (precision)

Make that a first-class concept so we can test LLM verify, stacked classifier, etc.

### Standardize Reporting

For every experiment:
- Precision/recall/F1/MAP
- **Precision at recall ≥ 94%** (primary metric)
- Recall@K & Precision@K
- Breakdown by challenge type
- List top 20 FPs and FNs (for qualitative inspection)
- Generalization: results on both primary and validation corpora

### Corpus Structure (for generalization)

Each corpus contains a `request.json` with:
```json
{
  "id": "request-id",
  "title": "Request Title",
  "request_text": "Full CPRA request text describing what records are sought...",
  "keywords": ["keyword1", "keyword2", ...],
  "date_range": {"start": "...", "end": "..."}
}
```

All experiment prompts should use these fields dynamically:
- `request.title` — Short description of the request
- `request.request_text` — Full CPRA request for LLM understanding
- `request.keywords` — For disambiguation and negative pattern generation

### Available Local LLMs (via Ollama)

**Generative models for verification/generation tasks:**
- `qwen3:8b` — Strong reasoning, flagship Qwen
- `qwen3:1.7b` — Fast, good quality
- `qwen3:0.6b` — Ultra-fast for simple tasks
- `gemma3:12b` — Google's latest, larger
- `gemma3:4b` — Google's latest, efficient
- `gemma2:2b` — Previous gen, fast
- `phi4-mini:3.8b` — Microsoft's efficient model
- `phi4-mini-reasoning:3.8b` — Reasoning-focused variant
- `phi3:mini` — Previous gen, stable
- `granite3.3:8b` — IBM, good for structured output
- `granite3.3:2b` — IBM, fast
- `deepseek-r1:8b` — Reasoning-focused
- `deepseek-r1:1.5b` — Fast reasoning
- `ministral-3:14b` — Mistral large
- `ministral-3:8b` — Mistral medium
- `ministral-3:3b` — Mistral small/fast
- `llama3:8b-instruct-q5_K_M` — Meta, instruction-tuned
- `gpt-oss:20b` — Largest available
- `olmo-3:7b` — Allen AI open model
- `functiongemma:270m` — Function calling specialist

**Embedding models (also available):**
- `qwen3-embedding:8b` — Qwen embedding large
- `qwen3-embedding:4b` — Qwen embedding medium
- `qwen3-embedding:0.6b` — Qwen embedding fast
- `embeddinggemma:300m` — Gemma-based embeddings
- `nomic-embed-text` — Efficient text embeddings
- `mxbai-embed-large` — Strong MTEB performer

### Embedding Models with Cached Results

Already computed (in `.cache/embeddings/`):
- all-mpnet-base-v2 (best overall)
- jina-embeddings-v3
- mxbai-embed-large
- bge-large-en-v1.5
- nomic-embed-text

---

## Quick Recommendations (What to Run First)

If we want the fastest path to "meaningfully better than mpnet alone":

1. **EXP-020** (validation sanity check) — ensure we're not overfitting
2. **EXP-025** (positive/negative prototypes) — directly targets keyword false positives
3. **EXP-027** (RRF ensemble) — easy win if it works, low complexity
4. **EXP-031** (LLM verifier with evidence) — likely biggest precision jump

---

## Cross-Encoder Experiments (Completed)

Testing whether different cross-encoder training objectives perform better than MS-MARCO.

### Results Summary

| Training Type | Best MAP | Problem |
|---------------|----------|---------|
| Retrieval (BGE, MS-MARCO) | 0.74 | Saturates at 100% recall, no discrimination |
| NLI (DeBERTa, MiniLM) | 0.67 | Scores everything as "entails" |
| STS (RoBERTa) | 0.41 | Expects similar-length inputs |
| Paraphrase (Quora) | 0.41 | Complete failure (1.29% recall) |

**Key finding**: Cross-encoders do NOT outperform bi-encoders on keyword-free corpora.

**Why they fail**: All cross-encoders were trained on data with lexical overlap or sentence-pair similarity. Our v2 corpus eliminates keyword overlap, exposing their reliance on surface-level patterns rather than true semantic understanding.

### Detailed Results

| # | Name | Recall | Precision | MAP | Status |
|---|------|--------|-----------|-----|--------|
| 011 | Cross-Encoder MiniLM (MS-MARCO) | 98.71% | 47.22% | 0.74 | Complete |
| 012 | Cross-Encoder NLI DeBERTa Base | 100% | 45.72% | 0.52 | Complete |
| 013 | Cross-Encoder NLI DeBERTa Large | 100% | 45.86% | 0.40 | Complete |
| 014 | Cross-Encoder NLI MiniLM | 100% | 45.72% | 0.67 | Complete |
| 015 | Cross-Encoder STS-B RoBERTa Large | 100% | 45.72% | 0.41 | Complete |
| 016 | Cross-Encoder STS-B DistilRoBERTa | 100% | 45.72% | 0.38 | Complete |
| 017 | Cross-Encoder Quora RoBERTa | 1.29% | 28.57% | 0.41 | Complete (failed) |
| 018 | BGE Reranker Base | 100% | 45.72% | 0.74 | Complete |
| 019 | BGE Reranker Large | 100% | 45.72% | 0.71 | Complete |

---

## Completed Bi-Encoder Experiments

| # | Name | Recall | Precision | F1 | MAP | Meets 94%? |
|---|------|--------|-----------|----|----|------------|
| 001 | Keyword Baseline | 83.87% | 55.32% | 66.67% | — | No |
| 002 | Snowflake Arctic L v2.0 | 81.29% | 70.39% | 75.45% | 0.82 | No |
| 003 | Jina v3 | 98.06% | 51.70% | 67.73% | 0.85 | Yes |
| 004 | BGE-M3 | 100% | 46.83% | 63.79% | 0.85 | Yes |
| 005 | Embedding Gemma | 100% | 49.36% | 66.10% | 0.87 | Yes |
| 006 | **all-mpnet-base-v2** | **98.71%** | **57.74%** | **72.86%** | **0.89** | **Yes (Best)** |
| 007 | mxbai-embed-large | 98.71% | 51.17% | 67.41% | 0.86 | Yes |
| 008 | nomic-embed-text | 99.35% | 46.11% | 62.99% | 0.81 | Yes |
| 009 | BGE Large EN v1.5 | 99.35% | 47.24% | 64.04% | 0.84 | Yes |
| 010 | Qwen3 Embedding 0.6B | 89.03% | 77.53% | 82.87% | 0.87 | No |
