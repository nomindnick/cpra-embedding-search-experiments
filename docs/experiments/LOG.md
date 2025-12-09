# Experiment Log

This document tracks all experiments, their key findings, and learnings.

## Summary Table

| #   | Name             | Date       | Embedding Model        | LLM | F1     | vs Baseline | Key Finding                                                  |
| --- | ---------------- | ---------- | ---------------------- | --- | ------ | ----------- | ------------------------------------------------------------ |
| 001 | Baseline Keyword | 2025-12-09 | N/A                    | N/A | 70.74% | —           | High recall (94%), but low precision (57%)                   |
| 002 | Simple Embedding | 2025-12-09 | st:all-mpnet-base-v2   | N/A | 56.57% | -14.17%     | Great semantic understanding, but can't match keyword recall |
| 003 | Ollama nomic     | 2025-12-09 | ollama:nomic-embed-text| N/A | 26.09% | -44.65%     | Similar pattern to ST, best F1 82.58% @0.70 but 78% recall   |
| 004 | Ollama mxbai     | 2025-12-09 | ollama:mxbai-embed-large| N/A | 26.33% | -44.41%    | Best MAP (0.98), best F1 87.02% @0.70 but 81% recall         |
| 005 | Ollama gemma     | 2025-12-09 | ollama:embeddinggemma  | N/A | 76.41% | +5.67%      | **Best default results!** 95% recall @0.5, 99% precision @0.6 |

## Experiment Details

---

### 001 - Baseline Keyword Search

**Date:** 2025-12-09

**Hypothesis:** Establish baseline performance with traditional keyword matching to understand what we're trying to beat.

**Configuration:** `configs/experiments/001_baseline_keyword.yaml`

**Approach:**

- Use keywords from CPRA request definitions (primary + secondary)
- Boolean OR matching (any keyword matches)
- Exclude emails matching exclusion keywords
- No semantic understanding

**Results:**

| Metric    | Overall | Lead Testing | COVID Funds | SpEd   | EdTech  | Safety |
| --------- | ------- | ------------ | ----------- | ------ | ------- | ------ |
| Precision | 56.66%  | 100.00%      | 98.63%      | 24.59% | 100.00% | 90.24% |
| Recall    | 94.13%  | 81.33%       | 96.00%      | 97.33% | 100.00% | 98.67% |
| F1        | 70.74%  | 89.71%       | 97.30%      | 39.29% | 100.00% | 94.27% |

**By Challenge Type:**

| Challenge Type     | Precision | Recall  | F1      |
| ------------------ | --------- | ------- | ------- |
| Ambiguous Terms    | 100.00%   | 100.00% | 100.00% |
| Near Miss          | 76.00%    | 76.00%  | 76.00%  |
| Indirect Reference | 90.00%    | 51.43%  | 65.45%  |
| Temporal Mismatch  | 100.00%   | 100.00% | 100.00% |
| Partial Match      | 100.00%   | 66.67%  | 80.00%  |

**Observations:**

- Overall MAP is strong (0.9295), indicating good ranking quality
- **Special Education request is problematic**: Only 24.59% precision due to 230 false positives (many emails contain education keywords)
- **Indirect references are missed**: Only 51.43% recall for documents that don't explicitly use request keywords
- **Partial matches miss documents**: 66.67% recall - keyword matching misses documents with only partial keyword overlap
- Lead Testing and EdTech perform excellently - their keywords are more specific/unique

**Next Steps:**

- Test embedding search to see if semantic understanding helps with indirect references
- Investigate Special Education false positives - likely need better keyword specificity

---

### 002 - Simple Embedding Search

**Date:** 2025-12-09

**Hypothesis:** Pure semantic similarity search might beat keyword matching by understanding meaning rather than relying on exact keyword matches.

**Configuration:** `configs/experiments/002_simple_embedding.yaml`

**Key Changes from Previous:**

- No keyword matching - pure embedding similarity
- Uses sentence-transformers `all-mpnet-base-v2` model
- Embeds request search text + email subject/body
- Ranks by cosine similarity

**Results at Default Threshold (0.5):**

| Metric    | Overall | Lead Testing | COVID Funds | SpEd   | EdTech  | Safety |
| --------- | ------- | ------------ | ----------- | ------ | ------- | ------ |
| Precision | 39.91%  | 100.00%      | 98.63%      | 10.18% | 100.00% | 92.50% |
| Recall    | 97.07%  | 88.00%       | 96.00%      | 98.67% | 100.00% | 98.67% |
| F1        | 56.57%  | 93.62%       | 97.30%      | 18.45% | 100.00% | 95.48% |

**By Challenge Type:**

| Challenge Type     | Precision | Recall  | F1      | vs 001 Recall |
| ------------------ | --------- | ------- | ------- | ------------- |
| Ambiguous Terms    | 100.00%   | 100.00% | 100.00% | =             |
| Near Miss          | 76.92%    | 80.00%  | 78.43%  | +4.00%        |
| Indirect Reference | 54.24%    | 91.43%  | 68.09%  | **+40.00%**   |
| Temporal Mismatch  | 100.00%   | 100.00% | 100.00% | =             |
| Partial Match      | 100.00%   | 91.67%  | 95.65%  | +25.00%       |

**Threshold Analysis:**

| Threshold | Precision  | Recall     | F1         | Predicted | TP      | FP     | FN     |
| --------- | ---------- | ---------- | ---------- | --------- | ------- | ------ | ------ |
| 0.30      | 16.82%     | 100.00%    | 28.80%     | 2229      | 375     | 1854   | 0      |
| 0.40      | 23.43%     | 99.73%     | 37.95%     | 1596      | 374     | 1222   | 1      |
| 0.50      | 39.91%     | 97.07%     | 56.57%     | 912       | 364     | 548    | 11     |
| **0.60**  | **86.53%** | **80.53%** | **83.43%** | **349**   | **302** | **47** | **73** |
| 0.70      | 97.08%     | 35.47%     | 51.95%     | 137       | 133     | 4      | 242    |

**Observations:**

1. **Threshold tuning matters, but no threshold solves the problem**:
   
   - At 0.5: 97% recall but only 40% precision (too many false positives)
   - At 0.6: 86% precision but only 80% recall (missing 20% of documents is legally unacceptable)
   - No single threshold achieves both high recall AND high precision

2. **The recall problem is critical**: In legal discovery, missing 20% of responsive documents (73 documents at 0.60 threshold) creates significant liability risk. We need to match or exceed keyword search's 94% recall while improving precision.

3. **Semantic understanding is valuable**:
   
   - Indirect references: 91.43% recall vs 51.43% (+40%)
   - Partial matches: 91.67% vs 66.67% (+25%)
   - Embeddings find documents that don't contain exact keywords — this capability is worth preserving

4. **The 0.60-0.70 cliff**: Sharp drop in recall between 0.60 (80.53%) and 0.70 (35.47%). The model is confident about ~300 documents but uncertain about ~200 more that are actually responsive. We need to rescue those uncertain documents.

5. **Pure embeddings are not sufficient**: While embeddings excel at semantic understanding, using them alone forces an unacceptable precision/recall tradeoff. A hybrid approach is needed.

**Key Insight:**
Embeddings alone cannot match keyword search recall while improving precision. The semantic understanding is valuable (finding indirect references), but we need a hybrid approach that:

- Ensures high recall (≥94%) to meet legal obligations
- Uses embeddings to improve precision by better ranking/filtering
- Doesn't sacrifice the semantic understanding gains

**Next Steps:**

- **Priority: Hybrid approach** — Use keywords for high-recall initial set, then embeddings to rank/rerank
- Test query expansion to help embeddings find the uncertain middle tier
- Investigate why 73 documents are missed at 0.60 threshold — are they indirect references that need lower threshold?
- Consider two-pass approach: high-recall embedding pass + keyword verification

---

### 003 - Ollama nomic-embed-text

**Date:** 2025-12-09

**Hypothesis:** Test whether a different embedding model (nomic-embed-text via Ollama) provides better precision/recall balance than sentence-transformers.

**Configuration:** `configs/experiments/003_ollama_nomic.yaml`

**Key Changes from Previous:**

- Uses Ollama's nomic-embed-text model (768 dimensions)
- Local model, no API costs

**Threshold Analysis:**

| Threshold | Precision | Recall  | F1     | Predicted | TP  | FP   | FN  |
| --------- | --------- | ------- | ------ | --------- | --- | ---- | --- |
| 0.30-0.50 | 15.00%    | 100.00% | 26.09% | 2500      | 375 | 2125 | 0   |
| 0.60      | 17.90%    | 98.40%  | 30.28% | 2062      | 369 | 1693 | 6   |
| **0.70**  | **87.24%**| **78.40%** | **82.58%** | 337  | 294 | 43   | 81  |
| 0.80      | 100.00%   | 4.27%   | 8.18%  | 16        | 16  | 0    | 359 |

**Observations:**

- MAP: 0.9052 (lower than sentence-transformers 0.9494)
- Same fundamental tradeoff: can't achieve both high recall AND high precision
- Best F1 at threshold 0.70 (vs 0.60 for ST) — scores are distributed differently
- 78% recall at best F1 is still unacceptable for legal discovery

---

### 004 - Ollama mxbai-embed-large

**Date:** 2025-12-09

**Hypothesis:** Test whether a larger, higher-quality embedding model (mxbai-embed-large, 1024 dimensions) improves results.

**Configuration:** `configs/experiments/004_ollama_mxbai.yaml`

**Key Changes from Previous:**

- Uses Ollama's mxbai-embed-large model (1024 dimensions)
- Larger model, potentially better semantic understanding

**Threshold Analysis:**

| Threshold | Precision  | Recall     | F1         | Predicted | TP  | FP   | FN  |
| --------- | ---------- | ---------- | ---------- | --------- | --- | ---- | --- |
| 0.30-0.50 | 15.00%     | 100.00%    | 26.09%     | 2500      | 375 | 2125 | 0   |
| 0.60      | 27.34%     | 99.73%     | 42.91%     | 1368      | 374 | 994  | 1   |
| **0.70**  | **93.56%** | **81.33%** | **87.02%** | 326       | 305 | 21   | 70  |
| 0.80      | 100.00%    | 11.73%     | 21.00%     | 44        | 44  | 0    | 331 |

**Observations:**

1. **Best MAP so far**: 0.9818 — excellent ranking quality
2. **Best F1 so far**: 87.02% at threshold 0.70 (vs 83.43% for ST, 82.58% for nomic)
3. **Still fails the recall requirement**: 81.33% recall means missing 70 documents
4. **Better precision at equivalent recall**: At ~80% recall, mxbai has 93.56% precision vs 86.53% for ST

**Key Finding:**
mxbai-embed-large produces the best results so far, but the fundamental problem remains: **no embedding model alone can achieve both ≥94% recall AND high precision**. Different models shift the precision/recall curve but don't solve the tradeoff.

---

### 005 - Ollama embeddinggemma

**Date:** 2025-12-09

**Hypothesis:** Test Google's embeddinggemma model — different architecture may produce different precision/recall characteristics.

**Configuration:** `configs/experiments/006_ollama_embeddinggemma.yaml`

**Key Changes from Previous:**

- Uses Google's embeddinggemma:300m model (768 dimensions)
- Different training methodology than other models

**Results at Default Threshold (0.5):**

| Metric    | Overall | Lead Testing | COVID Funds | SpEd   | EdTech | Safety |
| --------- | ------- | ------------ | ----------- | ------ | ------ | ------ |
| Precision | 63.70%  | 98.51%       | 100.00%     | 31.91% | 93.75% | 58.40% |
| Recall    | 95.47%  | 88.00%       | 92.00%      | 100.00%| 100.00%| 97.33% |
| F1        | 76.41%  | 92.96%       | 95.83%      | 48.39% | 96.77% | 73.00% |

**Threshold Analysis:**

| Threshold | Precision  | Recall     | F1         | Predicted | TP  | FP   | FN  |
| --------- | ---------- | ---------- | ---------- | --------- | --- | ---- | --- |
| 0.30      | 15.03%     | 100.00%    | 26.13%     | 2495      | 375 | 2120 | 0   |
| 0.40      | 21.69%     | 100.00%    | 35.65%     | 1729      | 375 | 1354 | 0   |
| **0.50**  | **63.70%** | **95.47%** | **76.41%** | 562       | 358 | 204  | 17  |
| 0.60      | 99.00%     | 78.93%     | 87.83%     | 299       | 296 | 3    | 79  |
| 0.70      | 100.00%    | 8.80%      | 16.18%     | 33        | 33  | 0    | 342 |

**Observations:**

1. **Best recall at usable precision**: At 0.50 threshold, achieves **95.47% recall** — very close to keyword's 94.13%!
2. **Dramatically different score distribution**: Sharp transition between 0.50 and 0.60 (vs gradual for other models)
3. **Best F1 at default threshold**: 76.41% F1 at 0.50 beats keywords (70.74%) while nearly matching recall
4. **Exceptional precision at 0.60**: 99% precision with only 3 false positives
5. **Special Education still problematic**: 31.91% precision, but 100% recall (finds all responsive docs)

**Key Finding:**
embeddinggemma is the **first model to nearly match keyword recall** (95.47% vs 94.13%) while improving precision (63.70% vs 56.66%). This is the most promising result so far for a single-model approach.

**Challenge Type Performance:**

| Challenge Type     | Precision | Recall  | F1      |
| ------------------ | --------- | ------- | ------- |
| Ambiguous Terms    | 100.00%   | 100.00% | 100.00% |
| Near Miss          | 97.56%    | 80.00%  | 87.91%  |
| Indirect Reference | 96.43%    | 77.14%  | 85.71%  |
| Temporal Mismatch  | 100.00%   | 100.00% | 100.00% |
| Partial Match      | 100.00%   | 100.00% | 100.00% |

Note: Lower recall on indirect_reference (77%) and near_miss (80%) compared to other embeddings — this is the tradeoff for higher overall precision.

---

## Model Comparison Summary

| Model                   | Best F1  | Recall@BestF1 | Precision@BestF1 | MAP    |
| ----------------------- | -------- | ------------- | ---------------- | ------ |
| Keywords (baseline)     | 70.74%   | 94.13%        | 56.66%           | 0.9295 |
| st:all-mpnet-base-v2    | 83.43%   | 80.53%        | 86.53%           | 0.9494 |
| ollama:nomic-embed-text | 82.58%   | 78.40%        | 87.24%           | 0.9052 |
| ollama:mxbai-embed-large| 87.02%   | 81.33%        | 93.56%           | 0.9818 |
| **ollama:embeddinggemma** | **87.83%** | 78.93%     | **99.00%**       | 0.9781 |

**Updated Analysis:**

embeddinggemma shows a notably different score distribution:
- At **0.50 threshold**: 95.47% recall, 63.70% precision — **closest to matching keyword recall!**
- At **0.60 threshold**: 78.93% recall, 99.00% precision — highest precision of any model

This suggests embeddinggemma may be the best candidate for a hybrid approach, as it can achieve near-keyword recall at a reasonable threshold.

**Conclusion:** While no single threshold achieves both ≥94% recall AND high precision, **embeddinggemma at threshold 0.50 comes closest** (95% recall). A hybrid approach using embeddinggemma + keywords could potentially achieve the best of both worlds.

---

## Template for New Experiments

```markdown
### NNN - Experiment Name

**Date:** YYYY-MM-DD

**Hypothesis:** What we're testing and why.

**Configuration:** `configs/experiments/NNN_name.yaml`

**Key Changes from Previous:**
- What's different from the last experiment

**Results:**

| Metric    | Overall | vs 001 Baseline |
| --------- | ------- | --------------- |
| Precision | X.XX    | +/- X.X%        |
| Recall    | X.XX    | +/- X.X%        |
| F1        | X.XX    | +/- X.X%        |

**Observations:**
- What worked
- What didn't
- Surprising findings

**Next Steps:**
- What to try next based on these results
```
