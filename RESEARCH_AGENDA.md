# Research Agenda: CPRA Semantic Search (v2 corpus)

> Last updated: 2026-01-28

This is the "what do we try next?" plan for pushing **precision up** while keeping **recall ≥ 94%** (legal requirement).

---

## Where We Are (Baseline)

**Current baselines on primary (lead) corpus:**
- `all-mpnet-base-v2` (Exp 006): 98.71% recall, 57.74% precision at threshold 0.30
- `voyage-4-nano-asymmetric` (Exp 021): 98.06% recall, 65.24% precision at threshold 0.35
- `BGE Large EN v1.5` (Exp 009): 99.35% recall, 47.24% precision at threshold 0.50

**EXP-020 (validation corpus) showed different models excel on different corpora** — no single model dominates across both. This suggests we should continue exploring rather than committing to one embedder.

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

## LLM Model Recommendations

Based on EXP-000 testing (16 models evaluated). For each task type, **try candidate models in order** — the first may not always work best in your specific pipeline context.

### Classification Tasks (EXP-031, EXP-032)

For document responsiveness classification:

| Priority | Model | Accuracy | Latency | Prompt Style | Notes |
|----------|-------|----------|---------|--------------|-------|
| 1 | gemma3:4b | 100% | 3.3s | Few-shot | Fastest perfect scorer |
| 2 | ministral-3:3b | 100% | 11.3s | Ternary | Also excellent at extraction |
| 3 | phi4-mini:3.8b | 90% | 6.7s | Multi-shot | Well-rounded alternative |
| 4 | qwen3:8b | 95% | 45.6s | Zero-shot | High accuracy but slow |

**Fallback (speed priority):** gemma2:2b (90% @ 2.2s) with ternary prompts

### Generation Tasks (EXP-021, EXP-025)

For paraphrases, example documents, facet queries:

| Priority | Model | Paraphrase | Email Gen | Latency | Notes |
|----------|-------|------------|-----------|---------|-------|
| 1 | phi4-mini:3.8b | 100% | 50%* | 22s | Best paraphrase; *pos example fails |
| 2 | ministral-3:3b | 100% | 100% | 44s | All tasks work, slower |
| 3 | granite3.3:2b | 100% | 100% | 33s | Reliable across all tasks |
| 4 | qwen3:0.6b | 100% | 100% | 9s | Fast, lower diversity |

**Note:** gemma3:4b excels at email generation (100%) but fails paraphrase format.

### Extraction Tasks (EXP-023, EXP-024, EXP-031, EXP-033)

For quote extraction, keyword extraction, evidence retrieval:

| Priority | Model | Quote Accuracy | Keyword Format | Latency | Notes |
|----------|-------|----------------|----------------|---------|-------|
| 1 | gemma3:12b | 96% | ✓ | 19s | Best accuracy, slower |
| 2 | ministral-3:3b | 96% | ✓ | 15s | Ties gemma3:12b, faster |
| 3 | phi4-mini:3.8b | 76% | ✓ | 4s | Best speed/accuracy ratio |
| 4 | gemma2:2b | 41% | ✗ | 11s | Budget option |

**Warning:** qwen3:8b and smaller qwen models have higher hallucination rates on extraction.

### Speed vs Accuracy Tradeoffs

| Scenario | Model | Task Accuracy | Time for 339 docs |
|----------|-------|---------------|-------------------|
| **Fastest usable** | gemma2:2b | 90% classification | ~12 min |
| **Best balance** | gemma3:4b | 100% classification | ~19 min |
| **Best extraction** | ministral-3:3b | 96% quotes | ~85 min |
| **Maximum accuracy** | gemma3:12b | 96% extraction | ~107 min |

### Model Selection Guidelines

1. **Start with the top candidate** for each task type
2. **Test with your actual prompts** — EXP-000 results may not transfer perfectly
3. **Have a fallback ready** — if top choice fails on edge cases, try #2
4. **Consider pipeline position:**
   - Upfront tasks (paraphrase gen): prefer faster models
   - Verification (few candidates): accuracy matters more than speed
5. **Watch for prompt sensitivity** — if a model performs poorly, try a different prompt style before switching models

---

## Priority 0: Foundation & Sanity Checks

### EXP-000 — Local LLM Capability Assessment ✅ COMPLETE

**Goal:** Identify which local LLMs to use for which tasks in subsequent experiments. Different models excel at different tasks (classification vs generation vs extraction), and latency matters when processing 339+ documents.

**Status:** ✅ **COMPLETE** — 16 models tested. See `LLM_Capability_Assessment.md` for detailed results.

#### Key Findings

**1. Two models achieve 100% classification accuracy:**

| Model | Accuracy | Latency | Best Prompt Style |
|-------|----------|---------|-------------------|
| gemma3:4b | **100%** | 3.3s | Few-shot |
| ministral-3:3b | **100%** | 11.3s | Ternary (yes/no/maybe) |

**2. Optimal prompt strategy varies dramatically by model:**

| Model | Few-Shot | Multi-Shot | Zero-Shot Binary | Ternary |
|-------|----------|------------|------------------|---------|
| gemma3:4b | **100%** ⭐ | 95% | 95% | 70% |
| ministral-3:3b | 80% | 55% | 90% | **100%** ⭐ |
| olmo-3:7b | **5%** 💀 | 10% | 72% | — |
| qwen3:8b | 60% | 60% | **95%** | 95% |

**Critical insight:** A model that excels with one prompt approach may fail catastrophically with another. Always test multiple strategies.

**3. Bigger ≠ Better:**

| Model | Size | Best Accuracy |
|-------|------|---------------|
| gemma3:4b | 3.3 GB | **100%** |
| ministral-3:3b | 3.0 GB | **100%** |
| gpt-oss:20b | 13 GB | 87.5% |

**4. Models to avoid:**

| Model | Issue |
|-------|-------|
| Reasoning models (phi4-mini-reasoning, deepseek-r1) | Extended thinking breaks output parsing |
| Function calling models (functiongemma) | Refuses text classification tasks |
| olmo-3:7b | Catastrophic few-shot failure (5%) |

#### Model Recommendations by Task

See **LLM Model Recommendations** section below for per-task candidate lists.

---

### EXP-020 — Validation Corpus Sanity Check ✅ COMPLETE

**Hypothesis:** Models that perform well on Lead corpus also perform well on PFAS; if not, we may be overfitting.

**Method:**
1. Run experiments 003–010 (top bi-encoders) + Voyage models on `corpus/validation`
2. Produce:
   - Precision/recall/F1/MAP
   - Recall@K curves (K=25/50; validation set is small)
   - Challenge-type breakdown (responsive vs non-responsive)

**Status:** ✅ **COMPLETE** — See EXPERIMENT_LOG.md for detailed results.

**Key Findings:**
- Different models excel on different corpora — no single "winner"
- **Jina v3, mxbai-embed-large, BGE Large EN v1.5** achieve 100% recall on PFAS at default threshold
- **all-mpnet-base-v2** struggles on BURIED_IN_THREAD for PFAS (0% vs 90% on lead)
- **Qwen3 0.6B** fails to generalize (24% recall on PFAS vs 89% on lead)
- **BGE Large EN v1.5** has best precision at 94%+ recall on validation (70.59%)

**Implication:** We should continue exploring different models and approaches rather than committing to a single embedder. Model choice may depend on the specific CPRA request characteristics.

---

## Priority 1: Query Expansion (LLM-Assisted Multi-Query Retrieval)

**Goal:** Increase recall and/or create a better ranking distribution so we can raise the threshold without losing recall.

### EXP-021 — LLM Paraphrase Expansion (Multi-Query "OR")

**Hypothesis:** Multiple semantically-different phrasings reduce vocabulary mismatch and boost recall at higher thresholds.

**Method:**
1. Use local LLM to generate N paraphrases of the CPRA request (N ∈ {3, 5, 10})
2. Embed: original request + paraphrases
3. Score each email by aggregation across queries

**Candidate models (see LLM Model Recommendations):**
1. phi4-mini:3.8b — 100% paraphrase success, good diversity
2. ministral-3:3b — 100% success, slower
3. qwen3:0.6b — 100% success, fastest but lower diversity

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

**Candidate models (see LLM Model Recommendations):**
1. phi4-mini:3.8b — Good structured output, fast
2. ministral-3:3b — Reliable generation
3. granite3.3:2b — 100% JSON compliance if structured output needed

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

**Candidate models (see LLM Model Recommendations):**
1. granite3.3:2b — 100% format compliance, good keyword extraction
2. qwen3:0.6b — Fast (4s), 100% format compliance
3. ministral-3:3b — Comprehensive extraction (23 terms avg)

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

**Candidate models (see LLM Model Recommendations):**
1. ministral-3:3b — Best extraction (96% quote accuracy), comprehensive
2. phi4-mini:3.8b — 76% extraction, much faster
3. gemma3:12b — 96% extraction but slow; use if accuracy critical

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

**Candidate models (see LLM Model Recommendations):**
1. ministral-3:3b — 100% success on both positive and negative examples
2. granite3.3:2b — 100% success, good structure compliance
3. gemma3:4b — 100% email generation, but fails paraphrase format

**Note:** phi4-mini succeeds on negative examples but fails positive example structure.

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

**Variants:**
- 025a: `max_sim(email, positives)` — positive prototypes only
- 025b: `max_sim(email, positives) - λ * max_sim(email, negatives)` — contrastive with max
- 025c: `avg_sim(email, positives) - λ * avg_sim(email, negatives)` — contrastive with mean

**Expected win:** Precision bump without sacrificing recall.

**Status:** Pending

---

### EXP-025d — Query Prototypes with Asymmetric Encoding

**Hypothesis:** Asymmetric models (like Voyage 4 Nano) are trained for query→document matching. By generating **short query-like prototypes** instead of full pseudo-emails, we use the model the way it was trained — potentially improving discrimination.

**Key insight:** EXP-025a-c generate document prototypes (full emails) and compare documents to documents. But asymmetric encoding optimizes for query→document, not document→document. Phrasing prototypes as queries may leverage this training better.

**Method:**
1. LLM generates SHORT query-like descriptions (10-30 words each):
   - P positive queries describing what responsive content looks like
   - N negative queries describing false positive patterns
2. Encode prototypes as **queries** via `encode_query()`
3. Encode corpus emails as **documents** via `encode_document()`
4. Score: `max_sim(doc, positive_queries) - λ * max_sim(doc, negative_queries)`

**Positive query prototype prompt:**
```
Given this CPRA request:
{request.request_text}

Generate {P} SHORT search queries (10-30 words each) that would find responsive documents.

Cover different aspects:
- Direct discussions of the core topic
- Technical/regulatory terminology and jargon
- Indirect references (e.g., project names, related activities)
- Historical events or future planning related to the topic

Output one query per line. Be specific and concrete.
```

**Negative query prototype prompt:**
```
Given this CPRA request:
{request.request_text}

The request mentions these keywords: {request.keywords}

Generate {N} SHORT search queries (10-30 words each) that would find FALSE POSITIVES — documents that seem relevant but aren't.

Target these patterns:
- "{keyword}" used in unrelated contexts (e.g., "lead" meaning leadership)
- Adjacent topics in the same domain but different subject
- Administrative/procedural content tangentially related

Output one query per line. Be specific about what makes each a false positive.
```

**Example output (for lead contamination request):**

Positive queries:
- "water testing results showing elevated lead levels in residential samples"
- "lead service line replacement project timeline and contractor communications"
- "EPA action level exceedances and required public notification"
- "corrosion control treatment adjustments to reduce lead leaching"

Negative queries:
- "leadership transition planning for water department director position"
- "leading the infrastructure modernization initiative kickoff meeting"
- "general water main replacement project unrelated to lead pipes"
- "budget allocation for water system improvements no contamination mentioned"

**Why this might work better:**
1. Voyage asymmetric is trained on query→document pairs, not document→document
2. Short queries match the input distribution the query encoder expects
3. Queries express "what to look for" which aligns with retrieval training
4. Avoids generating long synthetic emails that may not match real email distribution

**Comparison to 025a-c:**

| Aspect | 025a-c (Document Prototypes) | 025d (Query Prototypes) |
|--------|------------------------------|-------------------------|
| Prototype format | Full pseudo-emails (100-300 words) | Short queries (10-30 words) |
| Encoding | `embed()` (symmetric) | `encode_query()` (asymmetric) |
| Comparison | document ↔ document | query → document |
| Model fit | General embedding models | Asymmetric retrieval models |
| LLM generation | Harder (realistic emails) | Easier (short descriptions) |

**Variants:**
- 025d-i: Voyage 4 Nano asymmetric with max aggregation
- 025d-ii: Voyage 4 Nano asymmetric with mean aggregation
- 025d-iii: Compare symmetric (025b) vs asymmetric (025d) on same model

**Implementation notes:**
- Requires `ContrastivePipeline` to support asymmetric encoding (encode prototypes as queries)
- May need to adjust λ since query-document similarities have different distributions than document-document

**Status:** Pending — depends on EXP-025a-c results for comparison

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

**Hypothesis:** Combining diverse embedders increases precision at the compliance threshold by stabilizing ranking. EXP-020 showed models have complementary strengths across corpora and challenge types.

**Candidates (updated based on EXP-020):**
- `all-mpnet-base-v2` — Best precision on Lead (57.74%), but 0% BURIED_IN_THREAD on PFAS
- `mxbai-embed-large` — 100% recall on PFAS, consistent across challenge types, best MAP on validation (0.9551)
- `bge-large-en-v1.5` — Best precision on PFAS (70.59%), 100% recall on both corpora
- `jina-embeddings-v3` — 100% recall on both corpora, good generalization

**Not recommended (based on EXP-020):**
- `qwen3-embedding-0.6b` — Fails to generalize (24% recall on PFAS vs 89% on Lead)
- `embeddinggemma` — Only 68% recall on PFAS at default threshold

**Method:**
1. Run each model, get ranked list (all documents, ranked by similarity)
2. Fuse with RRF: `score(d) = Σ 1/(k + rank_m(d))` where k=60 (standard)
3. Evaluate on both primary (Lead) and validation (PFAS) corpora

**Variants:**

| Variant | Models | Rationale |
|---------|--------|-----------|
| 027a | mpnet + mxbai | Complementary BURIED_IN_THREAD coverage (mpnet: 90% Lead/0% PFAS, mxbai: ~90% Lead/100% PFAS) |
| 027b | mpnet + BGE-Large | Best precision on each corpus (mpnet: Lead, BGE: PFAS) |
| 027c | mpnet + mxbai + BGE-Large | 3-model ensemble for maximum coverage |
| 027d | mpnet + mxbai + BM25 | Add lexical signal — keywords work well on PFAS (92% recall, 65.71% precision) |
| 027e | mxbai + BGE-Large | Skip mpnet — both have 100% recall on PFAS |

**Expected outcomes:**
- 027a/027c should improve BURIED_IN_THREAD coverage on PFAS
- 027b should improve precision on both corpora
- 027d tests whether hybrid (embedding + lexical) helps

**Status:** Pending — High priority based on EXP-020 findings

---

### EXP-028 — Precision Specialist Rerank (Two-Stage Scoring)

**Hypothesis:** A high-precision model as second-stage scorer can improve precision if a high-recall model supplies the candidate set.

**⚠️ Updated based on EXP-020:** Qwen3 fails to generalize (24% recall on PFAS) — do not use as reranker. Consider Voyage asymmetric instead (best average precision at 94%+ recall: 66.91%).

**Method:**
1. Stage 1: retrieve candidates with high-recall model (mxbai or BGE-Large, top_k=100-200)
2. Stage 2: re-score candidates with precision-focused model
3. Final score: weighted combination or RRF

**Variants:**

| Variant | Stage 1 | Stage 2 | Rationale |
|---------|---------|---------|-----------|
| 028a | mxbai (100% recall both) | Voyage-asym | Voyage has best avg precision but misses BURIED_IN_THREAD |
| 028b | BGE-Large (100% recall both) | mpnet | mpnet has better Lead precision |
| 028c | mxbai | BGE-Large | Both generalize well, different strengths |

**What to watch:** If recall drops below 94%, expand candidate set or reduce Stage 2 weight.

**Status:** Pending — Lower priority than EXP-027 (ensemble is simpler)

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

**Stage 2 (LLM judge):** For each candidate email, ask for structured output

**Candidate models (see LLM Model Recommendations):**

| Priority | Model | Classification | Extraction | Speed | Notes |
|----------|-------|---------------|------------|-------|-------|
| 1 | gemma3:4b | 100% | 16% | 3.3s | Best classification, weak extraction |
| 2 | ministral-3:3b | 100% | 96% | 11.3s | Best all-rounder |
| 3 | phi4-mini:3.8b | 90% | 76% | 6.7s | Good balance |

**Recommendation:** Start with gemma3:4b for speed. If evidence extraction quality matters, use ministral-3:3b or phi4-mini.

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

**Status:** Pending

---

### EXP-032 — LLM Verifier with Few-Shot Examples

**Concept:** Same as 031 but with examples in the prompt.

**Candidate models (see LLM Model Recommendations):**
1. gemma3:4b — 100% with few-shot (best prompt style for this model)
2. ministral-3:3b — Use ternary prompts instead (few-shot hurts this model!)
3. phi4-mini:3.8b — 90% with multi-shot (6 examples)

**Critical insight from EXP-000:** Few-shot helps some models dramatically (gemma3:4b: 70%→100%) but hurts others (ministral-3:3b: 100%→80%, qwen3:8b: 95%→60%). Match prompt style to model.

**Implementation options:**

a) **Use corpus examples:** Sample 3-5 responsive and 3-5 non-responsive documents from ground truth

b) **LLM-generated examples:** Before verification, ask LLM to generate hypothetical examples based on `request.request_text`

c) **Challenge-type coverage:** Select examples covering edge cases:
   - Responsive: indirect references, technical jargon, buried context
   - Non-responsive: keyword false positives, adjacent topics

**Hypothesis:** Few-shot examples improve LLM accuracy on edge cases (for models that respond well to few-shot).

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

**Candidate models (see LLM Model Recommendations):**
1. ministral-3:3b — 96% quote accuracy, best for verbatim extraction
2. gemma3:12b — 96% quote accuracy, slower
3. phi4-mini:3.8b — 76% accuracy, fastest

**Note:** Extraction accuracy matters here since we're quoting from the document. Avoid models with high hallucination rates (qwen3 family).

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

**Tested & Recommended:**
| Model | Status | Best Use | Notes |
|-------|--------|----------|-------|
| `gemma3:4b` | ✅ **TOP PICK** | Classification | 100% @ 3.3s (few-shot) |
| `ministral-3:3b` | ✅ **TOP PICK** | Classification + Extraction | 100% class, 96% extraction |
| `phi4-mini:3.8b` | ✅ Recommended | Generation + Extraction | 100% paraphrase, 76% extraction |
| `gemma3:12b` | ✅ Recommended | Extraction | 96% quote accuracy |
| `granite3.3:2b` | ✅ Recommended | Generation | 100% JSON compliance |
| `gemma2:2b` | ✅ Speed option | Classification | 90% @ 2.2s (fastest) |
| `qwen3:8b` | ✅ Usable | Classification | 95% but slow (45s) |
| `qwen3:0.6b` | ✅ Usable | Fast generation | 85% class, 100% gen |
| `qwen3:1.7b` | ✅ Usable | Classification | 85% binary |

**Tested & NOT Recommended:**
| Model | Status | Issue |
|-------|--------|-------|
| `ministral-3:8b` | ❌ Skip | Worse than 3b, slower, timeouts |
| `gpt-oss:20b` | ❌ Skip | 87.5% accuracy, extremely slow (~60s/doc) |
| `olmo-3:7b` | ❌ Skip | Catastrophic few-shot failure (5%) |
| `phi4-mini-reasoning:3.8b` | ❌ Skip | 46% accuracy, reasoning breaks parsing |
| `deepseek-r1:1.5b` | ❌ Skip | 65% max, high hallucination |
| `phi3:mini` | ❌ Skip | 71% max, slower than phi4-mini |
| `functiongemma:270m` | ❌ Skip | 50% (random), refuses tasks |

**Not Yet Tested (Low Priority):**
| Model | Notes |
|-------|-------|
| `granite3.3:8b` | 2b version only reached 85% |
| `llama3:8b-instruct-q5_K_M` | Have 100% models already |
| `ministral-3:14b` | 3b already achieves 100% |
| `deepseek-r1:8b` | Reasoning models don't work |

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

## Quick Recommendations (What to Run Next)

Based on EXP-020 findings (models have complementary strengths):

1. ✅ **EXP-020** (validation sanity check) — COMPLETE. Found no single model dominates.

2. **EXP-027** (RRF ensemble) — **HIGH PRIORITY**
   - EXP-020 showed models have complementary strengths (mxbai: PFAS, mpnet: Lead)
   - Low complexity, directly tests the insight
   - Start with 027a (mpnet + mxbai) for BURIED_IN_THREAD coverage

3. **EXP-025** (positive/negative prototypes) — targets keyword false positives
   - More relevant for Lead corpus (the "lead/leadership" ambiguity)
   - Use ministral-3:3b (100% both pos/neg examples)

4. **EXP-031** (LLM verifier with evidence) — likely biggest precision jump
   - Start with gemma3:4b (100% @ 3.3s)
   - Fall back to ministral-3:3b if extraction quality matters

### Model Quick Reference

| Task | First Choice | Fallback | Avoid |
|------|--------------|----------|-------|
| Classification | gemma3:4b (few-shot) | ministral-3:3b (ternary) | reasoning models, olmo-3 |
| Paraphrase gen | phi4-mini:3.8b | ministral-3:3b | gemma3:4b (fails format) |
| Example gen | ministral-3:3b | granite3.3:2b | phi4-mini (pos fails) |
| Quote extraction | ministral-3:3b | gemma3:12b | qwen models (hallucinate) |
| Keyword extraction | granite3.3:2b | qwen3:0.6b | — |

See **LLM Model Recommendations** section for complete details.

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
| 006 | all-mpnet-base-v2 | 98.71% | 57.74% | 72.86% | 0.89 | Yes |
| 007 | mxbai-embed-large | 98.71% | 51.17% | 67.41% | 0.86 | Yes |
| 008 | nomic-embed-text | 99.35% | 46.11% | 62.99% | 0.81 | Yes |
| 009 | BGE Large EN v1.5 | 99.35% | 47.24% | 64.04% | 0.84 | Yes |
| 010 | Qwen3 Embedding 0.6B | 89.03% | 77.53% | 82.87% | 0.87 | No |
