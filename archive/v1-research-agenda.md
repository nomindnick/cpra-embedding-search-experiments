# Research Agenda: Improving CPRA Document Responsiveness Detection

## Current Baseline

| Model | Recall | Precision | F1 | Notes |
|-------|--------|-----------|-----|-------|
| Keywords | 94.13% | 56.66% | 70.74% | Legal minimum recall |
| embeddinggemma @0.50 | 95.47% | 63.70% | 76.41% | **Current best** - exceeds keyword recall |

**Constraint**: Must maintain ≥94% recall for legal compliance.

---

## Ideas to Test

### 1. More Embedding Models
**Status**: 🔲 Not started
**Priority**: High
**Effort**: Low

embeddinggemma significantly outperformed other models. Worth testing more:
- [ ] BGE models (BAAI/bge-large-en-v1.5, bge-m3)
- [ ] E5 models (intfloat/e5-large-v2, multilingual-e5-large)
- [ ] GTE models (Alibaba-NLP/gte-large-en-v1.5)
- [ ] Cohere embed-v3 (API)
- [ ] OpenAI text-embedding-3-large (API)
- [ ] Voyage AI embeddings (API)

**Hypothesis**: Model architecture matters more than we expected. Legal/formal text may benefit from specific training data.

---

### 2. Two-Stage: Embedding + Cross-Encoder Reranking
**Status**: 🔲 Not started
**Priority**: High
**Effort**: Medium

Use embeddinggemma with a low threshold (0.40?) to achieve ~100% recall, then apply a cross-encoder to rerank and filter for precision.

**Pipeline**:
```
emails → embeddinggemma (threshold 0.40) → candidates (~1700 docs, 100% recall)
      → cross-encoder rerank → top N or threshold → final results
```

**Cross-encoder options**:
- [ ] cross-encoder/ms-marco-MiniLM-L-12-v2
- [ ] BAAI/bge-reranker-v2-m3
- [ ] Cohere rerank-v3 (API)

**Why this might work**: Cross-encoders see query+document together, enabling much richer comparison than bi-encoder cosine similarity. They're too slow for full corpus but perfect for reranking 1000-2000 candidates.

---

### 3. Positive/Negative Example Scoring
**Status**: 🔲 Not started
**Priority**: High
**Effort**: Medium-High

Have an LLM generate synthetic examples of documents that would/wouldn't be responsive to each CPRA request, then use similarity to these examples as additional signals.

**Scoring formula**:
```
final_score = w1 * similarity_to_request
            + w2 * max(similarity_to_positive_examples)
            - w3 * max(similarity_to_red_herrings)
```

**Implementation steps**:
- [ ] Prompt LLM to generate 5-10 positive example snippets per request
- [ ] Prompt LLM to generate 5-10 "red herring" snippets (seem related but aren't responsive)
- [ ] Embed all examples
- [ ] Test different weight combinations (w1, w2, w3)
- [ ] Possibly learn weights via grid search or simple optimization

**Why this might work**:
- Positive examples expand what "responsive" means beyond the request text
- Red herrings create a negative boundary (e.g., for "Lead Testing in Water Systems", a red herring might be about "lead" as in leadership)
- This is essentially few-shot semantic classification

---

### 4. Query Expansion with LLM
**Status**: 🔲 Not started
**Priority**: Medium
**Effort**: Medium

Use LLM to expand the CPRA request into related terms, synonyms, and concepts.

**Approaches**:
- [ ] **Term expansion**: "water testing" → "water quality analysis, contamination testing, potability assessment"
- [ ] **Concept expansion**: Generate a paragraph describing what responsive documents might contain
- [ ] **HyDE (Hypothetical Document Embedding)**: Generate synthetic "ideal" responsive documents, embed those

**Why this might work**: The request text is often terse. "COVID Relief Fund Allocation" doesn't mention "CARES Act" or "stimulus" which might appear in responsive emails.

---

### 5. Hybrid: Keywords OR Embeddings
**Status**: 🔲 Not started
**Priority**: Medium
**Effort**: Low

Simple union of keyword matches and embedding matches above threshold.

```
responsive = keyword_matches ∪ embedding_matches
```

**Why this might work**: Keywords catch exact matches embeddings might miss; embeddings catch semantic matches keywords miss. Union should maximize recall.

**Risk**: Precision might drop significantly.

---

### 6. Per-Request Threshold Tuning
**Status**: 🔲 Not started
**Priority**: Medium
**Effort**: Low

Different CPRA requests may need different thresholds. "Special Education Program Changes" has broad concepts; "Lead Testing in Water Systems" is narrow.

- [ ] Analyze per-request optimal thresholds
- [ ] Test if using request-specific thresholds improves overall metrics

---

### 7. Field-Weighted Embeddings
**Status**: 🔲 Not started
**Priority**: Low
**Effort**: Low

Currently we concatenate subject + body. Test:
- [ ] Subject only
- [ ] Body only
- [ ] Weighted combination: `0.3 * sim(query, subject) + 0.7 * sim(query, body)`

---

### 8. Ensemble of Embedding Models
**Status**: 🔲 Not started
**Priority**: Low
**Effort**: Medium

Different models might catch different things. Combine scores:
```
final_score = mean(embeddinggemma_score, mxbai_score, mpnet_score)
```

Or use learned weights based on validation performance.

---

### 9. Active Learning / Uncertainty Sampling
**Status**: 🔲 Not started
**Priority**: Low (for later)
**Effort**: High

For production: flag uncertain predictions for human review, use feedback to improve.

---

## Experiments Completed

| # | Experiment | Result | Notes |
|---|------------|--------|-------|
| 001 | Keyword baseline | 70.74% F1, 94.13% recall | Legal minimum |
| 002 | st:all-mpnet-base-v2 | 83.43% F1 @0.60, 80.53% recall | Fails recall requirement |
| 003 | ollama:nomic-embed-text | 82.58% F1 @0.70, 78.40% recall | Fails recall requirement |
| 004 | ollama:mxbai-embed-large | 87.02% F1 @0.70, 81.33% recall | Best MAP (0.9818), fails recall |
| 005 | ollama:qwen3-embedding-4b | Timeout | Too slow |
| 006 | ollama:embeddinggemma | 76.41% F1 @0.50, 95.47% recall | **First to beat keywords on recall** |

---

## Next Steps (Prioritized)

1. **Test more embedding models** - Low effort, potentially high reward given embeddinggemma results
2. **Implement cross-encoder reranking** - Clear path to improving precision while maintaining recall
3. **Generate positive/negative examples** - Novel approach that could provide significant lift
4. **Query expansion** - Standard technique worth benchmarking

---

## Notes

- All experiments use the same 2,500 email synthetic corpus with 5 CPRA requests
- Ground truth: 375 total responsive emails (75 per request)
- Challenge types: ambiguous_terms, near_miss, indirect_reference, temporal_mismatch, partial_match
- embeddinggemma struggles with "indirect_reference" (77% recall) and "near_miss" (80% recall) - these are targets for improvement
