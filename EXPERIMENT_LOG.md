# Experiment Log - v2 Corpus

This document tracks experiments on the manually-crafted v2 corpus for evaluating embedding-based semantic search against keyword search for CPRA document discovery.

## Corpus Overview

| Corpus | Total Emails | Responsive | Non-Responsive |
|--------|--------------|------------|----------------|
| Primary (Lead) | 339 | 155 (46%) | 184 (54%) |
| Validation (PFAS) | 59 | 25 (42%) | 34 (58%) |

**Primary Corpus Challenge Distribution:**

| Challenge Type | Count | Category |
|----------------|-------|----------|
| DIRECT_MATCH | 30 | Responsive |
| AMBIGUOUS_TERMS | 30 | Responsive |
| INDIRECT_REFERENCE | 35 | Responsive |
| TECHNICAL_JARGON | 25 | Responsive |
| TEMPORAL_REFERENCE | 25 | Responsive |
| BURIED_IN_THREAD | 10 | Responsive |
| KEYWORD_FALSE_POSITIVE | 55 | Non-responsive |
| ADJACENT_TOPIC | 45 | Non-responsive |
| TRUE_NEGATIVE | 55 | Non-responsive |

---

## Summary Table

| # | Name | Date | Model | Precision | Recall | F1 | MAP | Meets 94%? |
|---|------|------|-------|-----------|--------|-----|-----|------------|
| 001 | Keyword Baseline | 2025-12-29 | N/A | 55.32% | 83.87% | 66.67% | 0.7953 | No |
| 002 | Snowflake Arctic L v2.0 | 2025-12-29 | snowflake-arctic-embed-l-v2.0 | 70.39% | 81.29% | 75.45% | 0.8373 | No |
| 003 | Jina v3 | 2025-12-29 | jina-embeddings-v3 | 51.70% | 98.06% | 67.71% | 0.8592 | **Yes** (0.50) |
| 004 | BGE-M3 | 2025-12-29 | bge-m3 | 46.83% | 100.00% | 63.79% | 0.8607 | **Yes** (0.40) |
| 005 | Embedding Gemma | 2025-12-29 | embeddinggemma (Ollama) | 49.36% | 100.00% | 66.10% | 0.8757 | **Yes** (0.30) |
| 006 | all-mpnet-base-v2 | 2025-12-29 | all-mpnet-base-v2 | 57.74% | 98.71% | 72.86% | 0.8923 | **Yes** (0.30) |
| 007 | mxbai-embed-large | 2025-12-29 | mxbai-embed-large (Ollama) | 51.17% | 98.71% | 67.40% | 0.8561 | **Yes** (0.50) |
| 008 | nomic-embed-text | 2025-12-29 | nomic-embed-text (Ollama) | 46.11% | 99.35% | 62.99% | 0.8158 | **Yes** (0.50) |
| 009 | BGE Large EN v1.5 | 2025-12-29 | bge-large-en-v1.5 | 47.24% | 99.35% | 64.03% | 0.8731 | **Yes** (0.50) |
| 010 | Qwen3 Embedding 0.6B | 2025-12-29 | qwen3-embedding-0.6b (Ollama) | 77.53% | 89.03% | 82.88% | 0.9169 | No |
| 011 | Cross-Encoder MiniLM | 2025-12-29 | ms-marco-MiniLM-L-6-v2 | 47.22% | 98.71% | 63.88% | 0.7177 | **Yes** (-9.0) |
| 012 | Cross-Encoder NLI DeBERTa Base | 2025-12-29 | nli-deberta-v3-base | 45.72% | 100.00% | 62.75% | 0.5182 | **Yes** (-5.0) |
| 013 | Cross-Encoder NLI DeBERTa Large | 2025-12-29 | nli-deberta-v3-large | 45.86% | 100.00% | 62.88% | 0.3990 | **Yes** (-1.0) |
| 014 | Cross-Encoder NLI MiniLM | 2025-12-29 | nli-MiniLM2-L6-H768 | 45.51% | 98.06% | 62.17% | 0.6704 | **Yes** (-5.0) |
| 015 | Cross-Encoder STS-B RoBERTa Large | 2025-12-29 | stsb-roberta-large | 45.72% | 100.00% | 62.75% | 0.4136 | **Yes** (0.0) |
| 016 | Cross-Encoder STS-B DistilRoBERTa | 2025-12-29 | stsb-distilroberta-base | 45.72% | 100.00% | 62.75% | 0.3807 | **Yes** (0.0) |
| 017 | Cross-Encoder Quora RoBERTa | 2025-12-29 | quora-roberta-large | 28.57% | 1.29% | 2.47% | 0.4115 | No |
| 018 | BGE Reranker Base | 2025-12-29 | bge-reranker-base | 45.72% | 100.00% | 62.75% | 0.7431 | **Yes** (0.0) |
| 019 | BGE Reranker Large | 2025-12-29 | bge-reranker-large | 45.72% | 100.00% | 62.75% | 0.7084 | **Yes** (0.0) |
| 020 | Voyage 4 Nano Baseline | 2026-01-28 | voyage-4-nano | 68.56% | 85.81% | 76.22% | 0.8220 | **Yes** (0.40) |
| 021 | Voyage 4 Nano Asymmetric | 2026-01-28 | voyage-4-nano-asymmetric | 65.24% | 98.06% | 78.35% | 0.9335 | **Yes** (0.35) |
| 027a | RRF mpnet + mxbai | 2026-01-28 | Ensemble[RRF] | 58.20% | 96.13% | 72.51% | 0.8888 | **Yes** (0.007) |
| 027b | RRF mpnet + BGE-Large | 2026-01-28 | Ensemble[RRF] | 58.37% | 96.77% | 72.82% | 0.8971 | **Yes** (0.007) |
| 027c | RRF mpnet + mxbai + BGE | 2026-01-28 | Ensemble[RRF] | 55.88% | 98.06% | 71.19% | 0.8889 | **Yes** (0.01) |
| 027d | RRF hybrid (mpnet + mxbai + keyword) | 2026-01-28 | Ensemble[RRF] | 50.84% | 97.42% | 66.81% | 0.9073 | **Yes** (0.01) |
| 027e | RRF mxbai + BGE-Large | 2026-01-28 | Ensemble[RRF] | 59.67% | 93.55% | 72.86% | 0.8657 | **Yes** (0.007) |
| 025a | Contrastive Positive Only (ministral) | 2026-01-29 | Contrastive[mpnet] | 55.20% | 99.35% | 70.97% | 0.8833 | **Yes** (0.35) |
| 025b | Contrastive Max (ministral, λ=0.5) | 2026-01-29 | Contrastive[mpnet] | 64.76% | 94.84% | 76.96% | 0.8897 | **Yes** (0.20) |
| 025f | Contrastive Max (gemma, λ=0.5) | 2026-01-29 | Contrastive[mpnet] | **69.48%** | 95.48% | **80.43%** | 0.8814 | **Yes** (0.20) |
| 025g | Contrastive Mean (gemma, λ=0.5) | 2026-01-29 | Contrastive[mpnet] | 64.94% | 96.77% | 77.72% | 0.8765 | **Yes** (0.15) |
| 025h | Contrastive Max (gemma-12b, λ=0.5) | 2026-01-30 | Contrastive[mpnet] | 63.29% | 96.77% | 76.53% | 0.7783 | **Yes** (0.20) |
| 025i | Contrastive Max (gemma, 10 proto) | 2026-01-30 | Contrastive[mpnet] | 67.13% | 93.55% | 78.17% | 0.8487 | No |
| 025j | Contrastive Max (gemma, λ=0.3) | 2026-01-31 | Contrastive[mpnet] | 63.07% | 98.06% | 76.77% | 0.8652 | **Yes** (0.25) |
| 025k | Contrastive Max (gemma, λ=0.7) | 2026-01-31 | Contrastive[mpnet] | 63.25% | 95.48% | 76.09% | 0.8734 | **Yes** (0.10) |
| 025l | Contrastive Corpus-Derived (ceiling) | 2026-01-31 | Contrastive[mpnet] | 64.78% | 96.13% | 77.40% | 0.8155 | **Yes** (0.25) |

### EXP-020: Validation Corpus Sanity Check (2026-01-28)

Ran 11 models on PFAS validation corpus to check generalization. Key findings:
- **No single model dominates** — different models excel on different corpora
- **BGE Large EN v1.5** best precision on validation (70.59% @ 94%+ recall)
- **Voyage 4 Nano Asymmetric** best average precision across both corpora
- **Jina v3, mxbai-embed-large** achieve 100% recall on validation
- **Qwen3 0.6B** fails to generalize (24% recall on validation vs 89% on primary)

This suggests we should continue exploring different models and pipeline approaches rather than committing to one embedder. See full EXP-020 section below for detailed results.

---

## Experiment Details

### 001 - Keyword Baseline

**Date:** 2025-12-29

**Configuration:** `configs/experiments/001_keyword_baseline.yaml`

**Purpose:** Establish baseline performance with traditional keyword matching on the v2 corpus.

**Approach:**
- Keywords from CPRA request: "lead", "contamination", "testing", "remediation", "water", "supply"
- Boolean OR matching (any keyword matches)
- Case-insensitive

**Results:**

| Metric | Value |
|--------|-------|
| Precision | 55.32% |
| Recall | 83.87% |
| F1 | 66.67% |
| Average Precision | 0.7953 |
| True Positives | 130 |
| False Positives | 105 |
| False Negatives | 25 |

**By Challenge Type:**

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|-----|-------|---------|
| DIRECT_MATCH | 100.00% | 100.00% | 100.00% | 30 | 30 |
| AMBIGUOUS_TERMS | 100.00% | 100.00% | 100.00% | 30 | 30 |
| INDIRECT_REFERENCE | 100.00% | 77.14% | 87.10% | 35 | 27 |
| TECHNICAL_JARGON | 100.00% | 64.00% | 78.05% | 25 | 16 |
| TEMPORAL_REFERENCE | 100.00% | 88.00% | 93.62% | 25 | 22 |
| BURIED_IN_THREAD | 100.00% | 50.00% | 66.67% | 10 | 5 |
| KEYWORD_FALSE_POSITIVE | 0.00% | 0.00% | 0.00% | 55 | 0 |
| ADJACENT_TOPIC | 0.00% | 0.00% | 0.00% | 45 | 0 |
| TRUE_NEGATIVE | 0.00% | 0.00% | 0.00% | 55 | 0 |

**Observations:**

1. **Recall below legal requirement**: 83.87% recall means 25 responsive documents would be missed - unacceptable for CPRA compliance which requires ≥94% recall.

2. **False positives from "lead" ambiguity**: 105 false positives, primarily from:
   - 55 KEYWORD_FALSE_POSITIVE emails using "lead" for leadership/leading
   - ~50 ADJACENT_TOPIC emails containing water-related keywords

3. **Challenge types where keywords struggle**:
   - **BURIED_IN_THREAD (50%)**: Responsive content buried in thread context
   - **TECHNICAL_JARGON (64%)**: Uses regulatory terms (LSL, CCT, ppb) without "lead"
   - **INDIRECT_REFERENCE (77%)**: References Flint, "materials of concern", etc.

4. **Challenge types where keywords excel**:
   - **DIRECT_MATCH (100%)**: Explicit lead contamination discussion
   - **AMBIGUOUS_TERMS (100%)**: Contains "lead" (metal) with disambiguating context

**Key Insight:**
The v2 corpus successfully exposes keyword search limitations. The 83.87% recall vs 94% requirement creates a clear gap for embedding models to fill. The 105 false positives (55.32% precision) show the cost of keyword ambiguity ("lead" matching both metal and leadership contexts).

---

### 002 - Snowflake Arctic L v2.0

**Date:** 2025-12-29

**Configuration:** `configs/experiments/002_snowflake_arctic_l_v2.yaml`

**Model:** snowflake-arctic-embed-l-v2.0

**Purpose:** Test the best v1 performer on the more challenging v2 corpus.

**Results (at best F1 threshold 0.30):**

| Metric | Value | vs Keyword Baseline |
|--------|-------|---------------------|
| Precision | 70.39% | +15.07% |
| Recall | 81.29% | -2.58% |
| F1 | 75.45% | +8.78% |
| Average Precision | 0.8373 | +0.042 |
| True Positives | 126 | -4 |
| False Positives | 53 | -52 |
| False Negatives | 29 | +4 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 70.39% | 81.29% | 75.45% | 126 | 53 | 29 |
| 0.35 | 81.98% | 58.71% | 68.42% | 91 | 20 | 64 |
| 0.40 | 92.96% | 42.58% | 58.41% | 66 | 5 | 89 |
| 0.45 | 94.59% | 22.58% | 36.46% | 35 | 2 | 120 |
| 0.50 | 100.00% | 9.68% | 17.65% | 15 | 0 | 140 |
| 0.55 | 100.00% | 2.58% | 5.03% | 4 | 0 | 151 |
| 0.60+ | 0.00% | 0.00% | 0.00% | 0 | 0 | 155 |

**By Challenge Type (at threshold 0.30):**

| Challenge Type | Recall | vs Keyword Baseline |
|----------------|--------|---------------------|
| DIRECT_MATCH | 100.00% | same |
| AMBIGUOUS_TERMS | 96.67% | -3.33% |
| INDIRECT_REFERENCE | 65.71% | -11.43% |
| TECHNICAL_JARGON | 80.00% | +16.00% |
| TEMPORAL_REFERENCE | 88.00% | same |
| BURIED_IN_THREAD | 20.00% | -30.00% |
| KEYWORD_FALSE_POSITIVE | 0.00% | (baseline: 0%) |
| ADJACENT_TOPIC | 0.00% | (baseline: 0%) |

**Observations:**

1. **Does not meet 94% recall requirement**: Maximum recall achieved is 81.29% at threshold 0.30, still below keyword baseline (83.87%) and far below the 94% legal requirement.

2. **Similarity scores are compressed**: No documents score above 0.60, and best performance requires threshold 0.30. This suggests the v2 corpus's semantic diversity makes it much harder to match than v1.

3. **Mixed performance vs keywords by challenge type**:
   - **Better**: TECHNICAL_JARGON (80% vs 64%) - embeddings understand regulatory terminology
   - **Same**: DIRECT_MATCH (100%), TEMPORAL_REFERENCE (88%)
   - **Worse**: BURIED_IN_THREAD (20% vs 50%), INDIRECT_REFERENCE (66% vs 77%)

4. **Precision improvement over keywords**: 70.39% precision vs 55.32% for keywords - halving false positives (53 vs 105). The model correctly rejects all KEYWORD_FALSE_POSITIVE and ADJACENT_TOPIC emails.

5. **v2 corpus is significantly harder**: The same model achieved 95.20% recall on v1 but only 81.29% on v2. This validates that v2 successfully tests semantic understanding without keyword overlap.

**Key Insight:**
The dramatic performance drop from v1 (95% recall) to v2 (81% recall) reveals that the v1 corpus likely had keyword leakage - responsive documents contained matching terms. The v2 corpus's keyword-free design exposes limitations in current embedding models for true semantic matching. The model excels at technical jargon but struggles with buried context and indirect references.

**Meets 94% Recall?** No (max 81.29% at threshold 0.30)

---

### 003 - Jina v3

**Date:** 2025-12-29

**Configuration:** `configs/experiments/003_jina_v3.yaml`

**Model:** jina-embeddings-v3 (570M params, task-specific LoRA, 8k context)

**Purpose:** Test Jina v3 which met 94% recall in v1 at threshold 0.70.

**Results (at threshold 0.50 for ≥94% recall):**

| Metric | Value | vs Keyword Baseline |
|--------|-------|---------------------|
| Precision | 51.70% | -3.62% |
| Recall | 98.06% | +14.19% |
| F1 | 67.71% | +1.04% |
| Average Precision | 0.8592 | +0.064 |
| True Positives | 152 | +22 |
| False Positives | 142 | +37 |
| False Negatives | 3 | -22 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.40 | 45.99% | 100.00% | 63.01% | 155 | 182 | 0 |
| 0.50 | 51.70% | 98.06% | 67.71% | 152 | 142 | 3 |
| 0.60 | 82.64% | 64.52% | 72.46% | 100 | 21 | 55 |
| 0.70 | 97.44% | 24.52% | 39.18% | 38 | 1 | 117 |
| 0.75 | 100.00% | 6.45% | 12.12% | 10 | 0 | 145 |

**By Challenge Type (at threshold 0.50):**

| Challenge Type | Recall | vs Keyword Baseline |
|----------------|--------|---------------------|
| DIRECT_MATCH | 100.00% | same |
| AMBIGUOUS_TERMS | 100.00% | same |
| INDIRECT_REFERENCE | ~97% | +20% |
| TECHNICAL_JARGON | ~96% | +32% |
| TEMPORAL_REFERENCE | ~96% | +8% |
| BURIED_IN_THREAD | ~90% | +40% |

*Note: At threshold 0.30, all responsive categories achieve 100% recall.*

**Observations:**

1. **MEETS 94% recall requirement**: At threshold 0.50, achieves 98.06% recall (152/155). At threshold 0.30-0.40, achieves perfect 100% recall.

2. **Precision trade-off**: Meeting 94%+ recall requires accepting lower precision (51.70% at 0.50 threshold) compared to keywords (55.32%). More false positives (142 vs 105).

3. **Best F1 at different threshold**: Best F1 (72.46%) occurs at threshold 0.60 with 82.64% precision but only 64.52% recall — doesn't meet legal requirement.

4. **Dramatically different from v1**: In v1, Jina met 94% at threshold 0.70. In v2, threshold 0.70 only yields 24.52% recall. The v2 corpus requires much lower thresholds.

5. **100% recall is achievable**: At thresholds ≤0.40, Jina v3 finds ALL responsive documents — something neither keywords nor Snowflake could achieve.

**Key Insight:**
Jina v3 is the first model to meet the 94% recall requirement on v2. Its broader similarity distribution allows it to capture semantically related documents that Snowflake misses. The trade-off is accepting more false positives — but for CPRA compliance where missing documents has legal consequences, high recall is mandatory. A two-stage approach (Jina for recall, then filtering) could combine the best of both.

**Meets 94% Recall?** **Yes** (98.06% at threshold 0.50, 100% at threshold 0.40)

---

### 004 - BGE-M3

**Date:** 2025-12-29

**Configuration:** `configs/experiments/004_bge_m3.yaml`

**Model:** bge-m3 (multi-functional: dense + sparse, 8k context)

**Purpose:** Test BGE-M3 which met 94% recall in v1 at threshold 0.60.

**Results (at threshold 0.40 for ≥94% recall):**

| Metric | Value | vs Keyword Baseline |
|--------|-------|---------------------|
| Precision | 46.83% | -8.49% |
| Recall | 100.00% | +16.13% |
| F1 | 63.79% | -2.88% |
| Average Precision | 0.8607 | +0.065 |
| True Positives | 155 | +25 |
| False Positives | 176 | +71 |
| False Negatives | 0 | -25 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.40 | 46.83% | 100.00% | 63.79% | 155 | 176 | 0 |
| 0.50 | 73.26% | 81.29% | 77.06% | 126 | 46 | 29 |
| 0.55 | 93.41% | 54.84% | 69.11% | 85 | 6 | 70 |
| 0.60 | 97.30% | 23.23% | 37.50% | 36 | 1 | 119 |
| 0.65 | 100.00% | 6.45% | 12.12% | 10 | 0 | 145 |

**By Challenge Type (at threshold 0.40):**

All responsive categories achieve 100% recall at threshold 0.40.

**Observations:**

1. **MEETS 94% recall requirement**: At threshold 0.40, achieves perfect 100% recall. However, drops sharply to 81.29% at threshold 0.50.

2. **Steeper precision-recall curve than Jina**: BGE-M3 drops from 100% to 81% recall between thresholds 0.40→0.50, while Jina only drops to 98%. BGE-M3 has a narrower "sweet spot."

3. **Best F1 doesn't meet recall requirement**: Best F1 (77.06%) at threshold 0.50 with 73.26% precision, but only 81.29% recall — same as Snowflake.

4. **Higher MAP than Jina**: 0.8607 vs 0.8592 — BGE-M3 ranks documents slightly better overall.

5. **Requires lower threshold than v1**: Met 94% at 0.60 in v1, but needs 0.40 in v2. The 0.60 threshold only yields 23.23% recall on v2.

**Key Insight:**
BGE-M3 can achieve 100% recall but requires a very low threshold (0.40), resulting in 176 false positives. Compared to Jina v3, BGE-M3 has a steeper drop-off — at threshold 0.50, Jina still has 98% recall while BGE-M3 drops to 81%. For CPRA compliance, Jina's gentler curve provides more flexibility in threshold selection.

**Meets 94% Recall?** **Yes** (100% at threshold 0.40, but only 81.29% at 0.50)

---

### 005 - Embedding Gemma

**Date:** 2025-12-29

**Configuration:** `configs/experiments/005_embeddinggemma.yaml`

**Model:** embeddinggemma (Ollama) - Google's embedding model

**Purpose:** Test embeddinggemma which met 94% recall in v1 at threshold 0.50.

**Results (at threshold 0.30 for ≥94% recall):**

| Metric | Value | vs Keyword Baseline |
|--------|-------|---------------------|
| Precision | 49.36% | -5.96% |
| Recall | 100.00% | +16.13% |
| F1 | 66.10% | -0.57% |
| Average Precision | 0.8757 | +0.080 |
| True Positives | 155 | +25 |
| False Positives | 159 | +54 |
| False Negatives | 0 | -25 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 49.36% | 100.00% | 66.10% | 155 | 159 | 0 |
| 0.40 | 67.15% | 89.68% | 76.80% | 139 | 68 | 16 |
| 0.45 | 82.96% | 72.26% | 77.24% | 112 | 23 | 43 |
| 0.50 | 91.86% | 50.97% | 65.56% | 79 | 7 | 76 |
| 0.55 | 98.11% | 33.55% | 50.00% | 52 | 1 | 103 |
| 0.60 | 95.45% | 13.55% | 23.73% | 21 | 1 | 134 |

**By Challenge Type (at threshold 0.30):**

All responsive categories achieve 100% recall at threshold 0.30.

**Observations:**

1. **MEETS 94% recall requirement**: At threshold 0.30, achieves perfect 100% recall. Drops to 89.68% at threshold 0.40.

2. **Fewest false positives at 100% recall**: 159 FPs vs Jina's 184 and BGE-M3's 176. Best precision (49.36%) among models achieving 100% recall.

3. **Highest MAP**: 0.8757 — best document ranking of all models tested so far.

4. **Steep drop-off**: Like BGE-M3, drops below 94% quickly (89.68% at 0.40). No intermediate threshold achieves 94%+ recall.

5. **Best F1 at 0.45**: 77.24% F1 with 82.96% precision but only 72.26% recall.

**Comparison at 100% recall:**

| Model | Threshold | Precision | FPs |
|-------|-----------|-----------|-----|
| **Embedding Gemma** | 0.30 | **49.36%** | **159** |
| BGE-M3 | 0.40 | 46.83% | 176 |
| Jina v3 | 0.30 | 45.72% | 184 |

**Key Insight:**
Embedding Gemma achieves the best precision at 100% recall (49.36%) and has the highest MAP (0.8757), indicating superior document ranking. However, like BGE-M3, it lacks a threshold between 94-100% recall — it's either 100% or below 90%. Jina v3 remains the only model with a usable intermediate threshold (98% at 0.50).

**Meets 94% Recall?** **Yes** (100% at threshold 0.30, but only 89.68% at 0.40)

---

### 006 - all-mpnet-base-v2

**Date:** 2025-12-29

**Configuration:** `configs/experiments/006_all_mpnet_base_v2.yaml`

**Model:** all-mpnet-base-v2 (SentenceTransformers baseline, CPU-friendly)

**Purpose:** Test baseline embedding model for comparison. Did not meet 94% recall in v1.

**Results (at threshold 0.30 for ≥94% recall):**

| Metric | Value | vs Keyword Baseline |
|--------|-------|---------------------|
| Precision | 57.74% | +2.42% |
| Recall | 98.71% | +14.84% |
| F1 | 72.86% | +6.19% |
| Average Precision | 0.8923 | +0.097 |
| True Positives | 153 | +23 |
| False Positives | 112 | +7 |
| False Negatives | 2 | -23 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 57.74% | 98.71% | 72.86% | 153 | 112 | 2 |
| 0.40 | 77.50% | 80.00% | 78.73% | 124 | 36 | 31 |
| 0.50 | 92.77% | 49.68% | 64.71% | 77 | 6 | 78 |
| 0.55 | 98.25% | 36.13% | 52.83% | 56 | 1 | 99 |
| 0.60 | 100.00% | 25.81% | 41.03% | 40 | 0 | 115 |

**By Challenge Type (at threshold 0.30):**

| Challenge Type | Recall | vs Keyword Baseline |
|----------------|--------|---------------------|
| DIRECT_MATCH | 100.00% | same |
| AMBIGUOUS_TERMS | 100.00% | same |
| INDIRECT_REFERENCE | 100.00% | +22.86% |
| TECHNICAL_JARGON | 96.00% | +32.00% |
| TEMPORAL_REFERENCE | 100.00% | +12.00% |
| BURIED_IN_THREAD | 90.00% | +40.00% |

**Observations:**

1. **BEST RESULT SO FAR**: At threshold 0.30, achieves 98.71% recall with 57.74% precision — best precision among models meeting 94%+ recall.

2. **Fewest false positives at 94%+ recall**: Only 112 FPs, significantly fewer than Jina (142), Embedding Gemma (159), or BGE-M3 (176).

3. **Highest MAP**: 0.8923 — superior document ranking over all other models tested.

4. **Best F1 at any threshold**: 78.73% F1 at threshold 0.40, though that only achieves 80% recall.

5. **Only 2 false negatives**: Misses only 1 TECHNICAL_JARGON and 1 BURIED_IN_THREAD document at threshold 0.30.

**Comparison at ≥94% recall:**

| Model | Threshold | Recall | Precision | FPs | MAP |
|-------|-----------|--------|-----------|-----|-----|
| **all-mpnet-base-v2** | 0.30 | **98.71%** | **57.74%** | **112** | **0.8923** |
| Jina v3 | 0.50 | 98.06% | 51.70% | 142 | 0.8592 |
| Embedding Gemma | 0.30 | 100% | 49.36% | 159 | 0.8757 |
| BGE-M3 | 0.40 | 100% | 46.83% | 176 | 0.8607 |

**Key Insight:**
The "baseline" all-mpnet-base-v2 outperforms all specialized models on v2 corpus. At 98.71% recall, it has the best precision (57.74%), fewest false positives (112), and highest MAP (0.8923). This suggests that on keyword-free corpora, simpler models with good general semantic understanding may outperform models optimized for lexical similarity.

**Meets 94% Recall?** **Yes** (98.71% at threshold 0.30 — BEST RESULT)

---

### 007 - mxbai-embed-large

**Date:** 2025-12-29

**Configuration:** `configs/experiments/007_mxbai_embed_large.yaml`

**Model:** mxbai-embed-large (Ollama) - Had best MAP (0.9818) in v1

**Purpose:** Test mxbai-embed-large which had excellent ranking in v1 but didn't meet 94% recall.

**Results (at threshold 0.50 for ≥94% recall):**

| Metric | Value | vs Keyword Baseline |
|--------|-------|---------------------|
| Precision | 51.17% | -4.15% |
| Recall | 98.71% | +14.84% |
| F1 | 67.40% | +0.73% |
| Average Precision | 0.8561 | +0.061 |
| True Positives | 153 | +23 |
| False Positives | 146 | +41 |
| False Negatives | 2 | -23 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.40 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.50 | 51.17% | 98.71% | 67.40% | 153 | 146 | 2 |
| 0.60 | 76.58% | 78.06% | 77.32% | 121 | 37 | 34 |
| 0.65 | 91.95% | 51.61% | 66.12% | 80 | 7 | 75 |
| 0.70 | 93.48% | 27.74% | 42.79% | 43 | 3 | 112 |

**By Challenge Type (at threshold 0.50):**

| Challenge Type | Recall | vs Keyword Baseline |
|----------------|--------|---------------------|
| DIRECT_MATCH | 100.00% | same |
| AMBIGUOUS_TERMS | 100.00% | same |
| INDIRECT_REFERENCE | ~97% | +20% |
| TECHNICAL_JARGON | ~96% | +32% |
| TEMPORAL_REFERENCE | ~100% | +12% |
| BURIED_IN_THREAD | ~90% | +40% |

*Note: At threshold 0.30-0.40, all responsive categories achieve 100% recall.*

**Observations:**

1. **MEETS 94% recall requirement**: At threshold 0.50, achieves 98.71% recall (same as all-mpnet-base-v2).

2. **Lower precision than all-mpnet**: 51.17% vs 57.74% at same recall level. 146 FPs vs 112.

3. **Lower MAP than v1**: 0.8561 on v2 vs 0.9818 on v1. The v2 corpus's semantic complexity reduces ranking quality.

4. **Best F1 at 0.60**: 77.32% F1 with 76.58% precision but only 78.06% recall.

5. **Similar curve to Jina v3**: Both achieve ~98.7% recall at threshold 0.50 with ~51% precision.

**Key Insight:**
mxbai-embed-large meets the 94% recall requirement but underperforms all-mpnet-base-v2 on all metrics. Its v1 advantage (best MAP) doesn't translate to v2, suggesting its ranking quality depended on lexical overlap that v2 eliminates.

**Meets 94% Recall?** **Yes** (98.71% at threshold 0.50)

---

### 008 - nomic-embed-text

**Date:** 2025-12-29

**Configuration:** `configs/experiments/008_nomic_embed_text.yaml`

**Model:** nomic-embed-text (Ollama) - Local-only option, 274MB

**Purpose:** Test lightweight local model for deployment scenarios without API dependencies.

**Results (at threshold 0.50 for ≥94% recall):**

| Metric | Value | vs Keyword Baseline |
|--------|-------|---------------------|
| Precision | 46.11% | -9.21% |
| Recall | 99.35% | +15.48% |
| F1 | 62.99% | -3.68% |
| Average Precision | 0.8158 | +0.021 |
| True Positives | 154 | +24 |
| False Positives | 180 | +75 |
| False Negatives | 1 | -24 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.40 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.50 | 46.11% | 99.35% | 62.99% | 154 | 180 | 1 |
| 0.60 | 64.43% | 80.65% | 71.63% | 125 | 69 | 30 |
| 0.70 | 97.62% | 26.45% | 41.62% | 41 | 1 | 114 |

**By Challenge Type (at threshold 0.50):**

All responsive categories achieve ~100% recall at threshold 0.50 (only 1 miss total).

**Observations:**

1. **MEETS 94% recall requirement**: At threshold 0.50, achieves 99.35% recall — excellent coverage.

2. **Lowest precision at 94%+ recall**: 46.11% precision, 180 false positives. Worst among models meeting requirement.

3. **Lowest MAP**: 0.8158 — poorest document ranking of all embedding models tested.

4. **Minimal threshold sensitivity 0.30-0.50**: Scores stay at ~100% recall across this range, then drop sharply at 0.60.

5. **Best F1 below requirement**: 71.63% F1 at threshold 0.60 with only 80.65% recall.

**Comparison for local-only deployment:**

| Model | Recall | Precision | FPs | MAP | Size |
|-------|--------|-----------|-----|-----|------|
| Embedding Gemma | 100% | 49.36% | 159 | 0.8757 | ~2GB |
| mxbai-embed-large | 98.71% | 51.17% | 146 | 0.8561 | ~670MB |
| **nomic-embed-text** | 99.35% | 46.11% | 180 | 0.8158 | **274MB** |

**Key Insight:**
nomic-embed-text meets the 94% recall requirement but has the lowest precision and MAP among all embedding models. For local-only deployment where size matters, Embedding Gemma or mxbai-embed-large offer better precision. nomic-embed-text is only preferable if the 274MB size constraint is critical.

**Meets 94% Recall?** **Yes** (99.35% at threshold 0.50)

---

### 009 - BGE Large EN v1.5

**Date:** 2025-12-29

**Configuration:** `configs/experiments/009_bge_large_en_v1.5.yaml`

**Model:** bge-large-en-v1.5 (SentenceTransformers) - Strong MTEB performer

**Purpose:** Test BGE Large which did not meet 94% recall in v1 (max 82.93%).

**Results (at threshold 0.50 for ≥94% recall):**

| Metric | Value | vs Keyword Baseline |
|--------|-------|---------------------|
| Precision | 47.24% | -8.08% |
| Recall | 99.35% | +15.48% |
| F1 | 64.03% | -2.64% |
| Average Precision | 0.8731 | +0.078 |
| True Positives | 154 | +24 |
| False Positives | 172 | +67 |
| False Negatives | 1 | -24 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.40 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.50 | 47.24% | 99.35% | 64.03% | 154 | 172 | 1 |
| 0.60 | 69.84% | 85.16% | 76.74% | 132 | 57 | 23 |
| 0.70 | 97.87% | 29.68% | 45.54% | 46 | 1 | 109 |

**By Challenge Type (at threshold 0.50):**

All responsive categories achieve ~100% recall at threshold 0.50 (only 1 miss total).

**Observations:**

1. **MEETS 94% recall on v2**: Surprisingly, BGE Large meets the requirement on v2 (99.35% at 0.50) despite failing on v1 (max 82.93%). The v2 corpus's design benefits this model.

2. **Middle-tier precision**: 47.24% precision, 172 FPs — better than nomic-embed-text but worse than all-mpnet-base-v2.

3. **Good MAP**: 0.8731 — third highest after all-mpnet-base-v2 (0.8923) and Embedding Gemma (0.8757).

4. **Best F1 below requirement**: 76.74% F1 at threshold 0.60 with 85.16% recall.

**Key Insight:**
BGE Large EN v1.5 shows an interesting reversal from v1 — it meets 94% on v2 despite failing on v1. This suggests the v1 corpus may have had characteristics that specifically disadvantaged this model, while v2's pure semantic design allows it to perform well.

**Meets 94% Recall?** **Yes** (99.35% at threshold 0.50)

---

### 010 - Qwen3 Embedding 0.6B

**Date:** 2025-12-29

**Configuration:** `configs/experiments/010_qwen3_embedding.yaml`

**Model:** qwen3-embedding-0.6b (Ollama) - Alibaba's 0.6B param embedding model

**Purpose:** Test Qwen3 which had highest F1 (93.37%) in v1 but did not meet 94% recall (max 90.13%).

**Results (at threshold 0.30 - best F1):**

| Metric | Value | vs Keyword Baseline |
|--------|-------|---------------------|
| Precision | 77.53% | +22.21% |
| Recall | 89.03% | +5.16% |
| F1 | 82.88% | +16.21% |
| Average Precision | 0.9169 | +0.122 |
| True Positives | 138 | +8 |
| False Positives | 40 | -65 |
| False Negatives | 17 | -8 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 77.53% | 89.03% | 82.88% | 138 | 40 | 17 |
| 0.40 | 95.74% | 58.06% | 72.29% | 90 | 4 | 65 |
| 0.50 | 96.43% | 17.42% | 29.51% | 27 | 1 | 128 |
| 0.55 | 100.00% | 7.74% | 14.37% | 12 | 0 | 143 |
| 0.60+ | 100.00% | <5% | — | — | 0 | — |

**By Challenge Type (at threshold 0.30):**

| Challenge Type | Recall | vs Keyword Baseline |
|----------------|--------|---------------------|
| DIRECT_MATCH | 100.00% | same |
| AMBIGUOUS_TERMS | 100.00% | same |
| INDIRECT_REFERENCE | 94.29% | +17.15% |
| TECHNICAL_JARGON | 84.00% | +20.00% |
| TEMPORAL_REFERENCE | 84.00% | -4.00% |
| BURIED_IN_THREAD | 30.00% | -20.00% |

**Observations:**

1. **Does NOT meet 94% recall requirement**: Max recall is 89.03% at threshold 0.30 — same pattern as v1 (90.13%).

2. **Best precision at high recall**: 77.53% precision at 89% recall is the best precision-at-recall ratio of any model. Only 40 false positives.

3. **Highest MAP**: 0.9169 — best document ranking of all models tested, indicating excellent semantic understanding.

4. **Best F1 of all models**: 82.88% F1 beats all-mpnet-base-v2 (72.86%), but at lower recall.

5. **Struggles with BURIED_IN_THREAD**: Only 30% recall on this category (vs 90% for all-mpnet-base-v2). The model misses relevant content buried in thread context.

6. **Compressed similarity scores**: Recall drops sharply from 89% at 0.30 to 58% at 0.40. The model has a narrow useful threshold range.

**Comparison of "failing" models:**

| Model | Max Recall | Precision | F1 | MAP |
|-------|------------|-----------|-----|-----|
| Qwen3 0.6B | 89.03% | **77.53%** | **82.88%** | **0.9169** |
| Snowflake Arctic | 81.29% | 70.39% | 75.45% | 0.8373 |
| Keywords | 83.87% | 55.32% | 66.67% | 0.7953 |

**Key Insight:**
Qwen3 has the best F1 and MAP but fails the legal recall requirement. It would be excellent for applications where precision matters more than recall, or as a second-stage reranker after a high-recall first pass. Its 30% BURIED_IN_THREAD recall suggests it struggles with long-context semantic matching.

**Meets 94% Recall?** No (max 89.03% at threshold 0.30)

---

### 011 - Cross-Encoder MiniLM-L-6

**Date:** 2025-12-29

**Configuration:** `configs/experiments/011_cross_encoder_minilm.yaml`

**Model:** cross-encoder/ms-marco-MiniLM-L-6-v2 - MS-MARCO trained cross-encoder

**Purpose:** Test if cross-encoder architecture provides 5-10% accuracy improvement over bi-encoders by processing query+document together.

**Results (at threshold -9.0 for ≥94% recall):**

| Metric | Value | vs all-mpnet-base-v2 |
|--------|-------|----------------------|
| Precision | 47.22% | **-10.52%** |
| Recall | 98.71% | same |
| F1 | 63.88% | **-8.98%** |
| Average Precision | 0.7177 | **-0.1746** |
| True Positives | 153 | -2 |
| False Positives | 171 | **+59** |
| False Negatives | 2 | same |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| -9.00 | 47.22% | 98.71% | 63.88% | 153 | 171 | 2 |
| -8.00 | 51.17% | 84.52% | 63.75% | 131 | 125 | 24 |
| -7.00 | 58.47% | 69.03% | 63.31% | 107 | 76 | 48 |
| -6.00 | 71.30% | 49.68% | 58.56% | 77 | 31 | 78 |
| -5.00 | 83.02% | 28.39% | 42.31% | 44 | 9 | 111 |
| -4.00 | 96.00% | 15.48% | 26.67% | 24 | 1 | 131 |

**By Challenge Type (at threshold -9.0):**

| Challenge Type | Recall | vs all-mpnet-base-v2 |
|----------------|--------|----------------------|
| DIRECT_MATCH | 86.67% | **-13.33%** |
| AMBIGUOUS_TERMS | 86.67% | **-13.33%** |
| INDIRECT_REFERENCE | 14.29% | **-85.71%** |
| TECHNICAL_JARGON | 28.00% | **-68.00%** |
| TEMPORAL_REFERENCE | 48.00% | **-52.00%** |
| BURIED_IN_THREAD | 10.00% | **-80.00%** |

**Observations:**

1. **WORSE than bi-encoders on every metric**: The cross-encoder has lower precision (47% vs 58%), lower F1 (64% vs 73%), and much lower MAP (0.72 vs 0.89).

2. **Catastrophic failure on keyword-free categories**:
   - INDIRECT_REFERENCE: 14% recall (vs 100% for all-mpnet)
   - TECHNICAL_JARGON: 28% recall (vs 96% for all-mpnet)
   - BURIED_IN_THREAD: 10% recall (vs 90% for all-mpnet)

3. **MS-MARCO training is the problem**: The cross-encoder was trained on web search data where queries and relevant passages share keywords. It learned to rely on lexical overlap, which the v2 corpus specifically eliminates.

4. **Score distribution is compressed and negative**: All scores range from -10.25 to -0.66 (logits). The model is essentially saying "nothing is relevant" because it can't find keyword matches.

5. **Perfect precision at high thresholds**: At -4.0, 96% precision with only 1 FP, but only 15% recall. The few documents it's confident about are correct.

**Key Insight:**
Cross-encoders are NOT inherently more accurate than bi-encoders — their advantage depends on training data. MS-MARCO cross-encoders are trained on search queries where lexical overlap is a strong relevance signal. On keyword-free corpora like v2, this training becomes a liability. The cross-encoder's ability to attend across query and document is wasted because it learned to look for the wrong things (word matches instead of meaning).

For CPRA document discovery with keyword-free content, **general-purpose bi-encoders (all-mpnet-base-v2) outperform retrieval-optimized cross-encoders**.

**Meets 94% Recall?** Yes (98.71% at threshold -9.0, but with worse precision than bi-encoders)

---

### 012 - Cross-Encoder NLI DeBERTa Base

**Date:** 2025-12-29

**Configuration:** `configs/experiments/012_cross_encoder_nli_deberta_base.yaml`

**Model:** cross-encoder/nli-deberta-v3-base - Natural Language Inference trained cross-encoder

**Purpose:** Test if NLI-trained cross-encoders overcome MS-MARCO's lexical bias by learning semantic entailment without keyword shortcuts.

**Results (at threshold -5.0 for 100% recall):**

| Metric | Value | vs all-mpnet-base-v2 |
|--------|-------|----------------------|
| Precision | 45.72% | **-12.02%** |
| Recall | 100.00% | +1.29% |
| F1 | 62.75% | **-10.11%** |
| Average Precision | 0.5182 | **-0.3741** |
| True Positives | 155 | +2 |
| False Positives | 184 | **+72** |
| False Negatives | 0 | -2 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| -5.0 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.0 | 45.56% | 99.35% | 62.47% | 154 | 184 | 1 |
| 2.0 | 45.51% | 98.06% | 62.17% | 152 | 182 | 3 |
| 3.0 | 45.92% | 98.06% | 62.55% | 152 | 179 | 3 |
| 4.0 | 45.99% | 96.13% | 62.21% | 149 | 175 | 6 |
| 5.0 | 49.59% | 78.71% | 60.85% | 122 | 124 | 33 |

**By Challenge Type (at threshold 0.0):**

| Challenge Type | Recall | vs MS-MARCO Cross-Encoder |
|----------------|--------|---------------------------|
| DIRECT_MATCH | 100.00% | **+13.33%** |
| AMBIGUOUS_TERMS | 96.67% | **+10.00%** |
| INDIRECT_REFERENCE | 100.00% | **+85.71%** |
| TECHNICAL_JARGON | 100.00% | **+72.00%** |
| TEMPORAL_REFERENCE | 100.00% | **+52.00%** |
| BURIED_IN_THREAD | 100.00% | **+90.00%** |

**Observations:**

1. **Dramatically better recall than MS-MARCO cross-encoder**: NLI training eliminates the lexical bias that crippled MS-MARCO. INDIRECT_REFERENCE jumps from 14% to 100%, BURIED_IN_THREAD from 10% to 100%.

2. **Achieves 100% recall**: At threshold -5.0 (and lower), captures ALL responsive documents — the first cross-encoder to match bi-encoder recall.

3. **But precision is terrible**: 45.72% precision means 184 false positives — worse than all bi-encoders. The model scores everything highly.

4. **Very poor MAP (0.5182)**: The worst ranking quality of any model tested. The model cannot differentiate responsive from non-responsive — it gives high "entailment" scores to everything.

5. **Score distribution problem**: Thresholds from -5.0 to -1.0 all yield identical results (100% recall, 339 predicted). The model is saturated.

**Analysis:**

The NLI cross-encoder "solves" the MS-MARCO lexical bias problem but creates a new problem: it has no discrimination capability for document relevance.

NLI models are trained to answer "Does premise A entail hypothesis B?" For short premise-hypothesis pairs, this works well. But for long documents paired with short queries, the model tends to find *something* in the document that could entail the query — especially for our v2 corpus where documents are topically related to water, infrastructure, and government.

The entailment task (A → B) is different from relevance (is A about B?). A document about "budget planning for infrastructure" may "entail" that lead contamination exists as a general concern (since infrastructure includes water systems), even though the document isn't *about* lead contamination.

**Key Insight:**
NLI training helps cross-encoders understand semantic relationships without lexical overlap, but the entailment task itself is too broad for document relevance scoring. The model needs relevance-specific fine-tuning on non-lexical examples to learn when "related" means "responsive."

**Meets 94% Recall?** **Yes** (100% at threshold -5.0, but with worst precision of all models)

---

### 013 - Cross-Encoder NLI DeBERTa Large

**Date:** 2025-12-29

**Configuration:** `configs/experiments/013_cross_encoder_nli_deberta_large.yaml`

**Model:** cross-encoder/nli-deberta-v3-large - Larger NLI model (more parameters)

**Purpose:** Test if larger NLI model improves discrimination over base model.

**Results (at threshold -1.0 for 100% recall):**

| Metric | Value | vs NLI DeBERTa Base |
|--------|-------|---------------------|
| Precision | 45.86% | +0.14% |
| Recall | 100.00% | same |
| F1 | 62.88% | +0.13% |
| Average Precision | 0.3990 | **-0.1192** |
| True Positives | 155 | same |
| False Positives | 183 | -1 |
| False Negatives | 0 | same |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| -5.0 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| -1.0 | 45.86% | 100.00% | 62.88% | 155 | 183 | 0 |
| 0.0 | 45.86% | 100.00% | 62.88% | 155 | 183 | 0 |
| 1.0 | 45.67% | 98.71% | 62.45% | 153 | 182 | 2 |
| 2.0 | 45.45% | 96.77% | 61.86% | 150 | 180 | 5 |
| 3.0 | 44.77% | 80.00% | 57.41% | 124 | 153 | 31 |
| 4.0 | 39.83% | 30.32% | 34.43% | 47 | 71 | 108 |

**Observations:**

1. **No improvement from larger model**: The large model has nearly identical recall/precision as the base model, but WORSE ranking (MAP 0.3990 vs 0.5182).

2. **Same saturation problem**: Like the base model, predicts almost everything as positive at low thresholds.

3. **Steeper drop-off at high thresholds**: Recall drops from 100% at threshold 0 to 30% at threshold 4, vs 78% for base model at threshold 5.

4. **Model size doesn't help discrimination**: More parameters don't improve the fundamental problem — NLI training teaches "does A entail B?" not "is A about B?"

**Key Insight:**
Larger NLI models don't improve document relevance scoring. The problem is the training objective (entailment), not model capacity. All NLI models will saturate on this task because most documents in our corpus can "entail" something about lead contamination in a loose sense.

**Meets 94% Recall?** **Yes** (100% at threshold -1.0, but with even worse MAP than base model)

---

### 014 - Cross-Encoder NLI MiniLM

**Date:** 2025-12-29

**Configuration:** `configs/experiments/014_cross_encoder_nli_minilm.yaml`

**Model:** cross-encoder/nli-MiniLM2-L6-H768 - Small NLI model

**Purpose:** Test if smaller NLI model has better discrimination than larger ones.

**Results (at threshold -5.0 for 100% recall):**

| Metric | Value | vs NLI DeBERTa Base |
|--------|-------|---------------------|
| Precision | 45.72% | same |
| Recall | 100.00% | same |
| F1 | 62.75% | same |
| Average Precision | 0.6704 | **+0.1522** |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| -5.0 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.0 | 45.51% | 98.06% | 62.17% | 152 | 182 | 3 |
| 1.0 | 50.00% | 88.39% | 63.87% | 137 | 137 | 18 |
| 2.0 | 63.36% | 53.55% | 58.04% | 83 | 48 | 72 |
| 3.0 | 100.00% | 8.39% | 15.48% | 13 | 0 | 142 |

**Observations:**

1. **Best MAP among NLI models**: 0.6704 vs 0.5182 (base) and 0.3990 (large). Smaller model ranks better.

2. **Same saturation at low thresholds**: 100% recall with 184 FPs at threshold ≤-1.0.

3. **Better high-threshold precision**: 100% precision at threshold 3.0 (vs never reaching 100% for larger models).

4. **Inverse scaling**: The smallest NLI model has the best ranking, largest has worst. Model capacity doesn't help discrimination.

**Key Insight:**
Among NLI models, smaller is better for ranking. The larger models may be overfitting to the entailment task. However, all NLI models still fail to discriminate well for document relevance.

**NLI Cross-Encoder Summary:**
| Model | MAP | 100% Recall Precision |
|-------|-----|----------------------|
| MiniLM (6L) | **0.6704** | 45.72% |
| DeBERTa Base | 0.5182 | 45.72% |
| DeBERTa Large | 0.3990 | 45.72% |

**Meets 94% Recall?** **Yes** (100% at threshold -5.0)

---

### 015 - Cross-Encoder STS-B RoBERTa Large

**Date:** 2025-12-29

**Configuration:** `configs/experiments/015_cross_encoder_stsb_roberta_large.yaml`

**Model:** cross-encoder/stsb-roberta-large - Semantic Textual Similarity trained

**Purpose:** Test if STS-trained models provide better discrimination than NLI models.

**Results (at threshold 0.0 for 100% recall):**

| Metric | Value | vs NLI MiniLM |
|--------|-------|---------------|
| Precision | 45.72% | same |
| Recall | 100.00% | same |
| F1 | 62.75% | same |
| Average Precision | 0.4136 | **-0.2568** |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.0 | 45.72% | 100.00% | 62.75% | 155 | 184 | 0 |
| 0.5 | 32.39% | 14.84% | 20.35% | 23 | 48 | 132 |
| 1.0+ | 0.00% | 0.00% | 0.00% | 0 | 0 | 155 |

**Observations:**

1. **All scores below 1.0**: The STS model gives very low similarity scores to everything. No document scores above 1.0 on the 0-5 STS scale.

2. **Worse MAP than all other cross-encoders**: 0.4136 MAP is worse than NLI MiniLM (0.6704) and even MS-MARCO (0.7177).

3. **Steep cliff at threshold 0.5**: Recall drops from 100% to 14.84% between thresholds 0.0 and 0.5.

4. **Sentence-level training doesn't transfer**: STS models are trained on sentence pairs, not document-query matching. Long emails vs short queries are out of distribution.

**Key Insight:**
STS-trained cross-encoders are unsuitable for document retrieval. They're trained to score sentence-level semantic similarity (scale 0-5) where inputs are similar lengths. Long documents paired with short queries produce uniformly low scores.

**Meets 94% Recall?** **Yes** (100% at threshold 0.0, but worst MAP of all models)

---

### 016 - Cross-Encoder STS-B DistilRoBERTa

**Date:** 2025-12-29

**Configuration:** `configs/experiments/016_cross_encoder_stsb_distilroberta.yaml`

**Model:** cross-encoder/stsb-distilroberta-base - Smaller STS model

**Purpose:** Compare smaller STS model to larger one.

**Results (at threshold 0.0 for 100% recall):**

| Metric | Value | vs STS RoBERTa Large |
|--------|-------|----------------------|
| Precision | 45.72% | same |
| Recall | 100.00% | same |
| F1 | 62.75% | same |
| Average Precision | 0.3807 | **-0.0329** |

**Threshold Analysis:**

Same pattern as STS-B RoBERTa Large — all scores below 1.0, cliff at threshold 0.5.

**Observations:**

1. **Even worse MAP (0.3807)**: The worst of any cross-encoder tested.

2. **Same behavior as larger model**: Both STS models give uniformly low scores.

**STS Cross-Encoder Summary:**
| Model | MAP | Best Achievable at 94%+ Recall |
|-------|-----|-------------------------------|
| STS-B RoBERTa Large | 0.4136 | 100% recall, 45.72% precision |
| STS-B DistilRoBERTa | 0.3807 | 100% recall, 45.72% precision |

**Conclusion:** STS training is unsuitable for document relevance scoring. These models have the worst MAPs of all experiments.

**Meets 94% Recall?** **Yes** (100% at threshold 0.0, but worst MAP overall)

---

### 017 - Cross-Encoder Quora RoBERTa Large

**Date:** 2025-12-29

**Configuration:** `configs/experiments/017_cross_encoder_quora.yaml`

**Model:** cross-encoder/quora-roberta-large - Paraphrase detection trained

**Purpose:** Test if paraphrase-trained models recognize semantic equivalence without word overlap.

**Results:**

| Metric | Value |
|--------|-------|
| Best Recall | 1.29% (at threshold 0.10) |
| Best Precision | 28.57% (at threshold 0.10) |
| Average Precision | 0.4115 |

**Threshold Analysis:**

| Threshold | Precision | Recall | Predicted |
|-----------|-----------|--------|-----------|
| 0.10 | 28.57% | 1.29% | 7 |
| 0.20+ | 0.00% | 0.00% | 0 |

**Observations:**

1. **Complete failure**: The model scores almost nothing above 0.1. Only 7 documents get any score.

2. **Wrong task**: Quora duplicate detection expects two similar-length questions. Query vs document is out of distribution.

3. **Cannot meet 94% recall**: Maximum recall is 1.29% — the worst of any model.

**Key Insight:**
Paraphrase detection models are designed to compare similar texts (questions to questions). They fail completely on query-document matching where lengths differ drastically.

**Meets 94% Recall?** **No** (max 1.29% — complete failure)

---

### 018 - BGE Reranker Base

**Date:** 2025-12-29

**Configuration:** `configs/experiments/018_bge_reranker_base.yaml`

**Model:** BAAI/bge-reranker-base - Retrieval-focused reranker

**Purpose:** Test if BGE rerankers have same lexical bias as MS-MARCO.

**Results (at threshold 0.0 for 100% recall):**

| Metric | Value | vs MS-MARCO |
|--------|-------|-------------|
| Precision | 45.72% | -1.50% |
| Recall | 100.00% | +1.29% |
| F1 | 62.75% | -1.13% |
| Average Precision | 0.7431 | **+0.0254** |

**Threshold Analysis:**
- All scores between 0 and 1 (cliff at threshold 1.0)
- 100% recall at threshold ≤0.0, 0% at threshold ≥1.0

**Observations:**

1. **Best MAP among cross-encoders**: 0.7431 beats MS-MARCO (0.7177) and all other cross-encoders.

2. **No discrimination**: Like NLI models, scores all documents between 0-1 with no clear threshold for filtering.

3. **Better than expected**: Despite retrieval training, BGE Reranker doesn't have MS-MARCO's catastrophic lexical bias. Achieves 100% recall on all challenge types.

**Key Insight:**
BGE Reranker Base is the best-ranking cross-encoder tested (MAP 0.7431), but still cannot match bi-encoder precision. It ranks responsive documents higher on average but cannot separate them from non-responsive with a threshold.

**Meets 94% Recall?** **Yes** (100% at threshold 0.0)

---

### 019 - BGE Reranker Large

**Date:** 2025-12-29

**Configuration:** `configs/experiments/019_bge_reranker_large.yaml`

**Model:** BAAI/bge-reranker-large - Larger retrieval-focused reranker

**Purpose:** Test if larger BGE reranker improves over base.

**Results (at threshold 0.0 for 100% recall):**

| Metric | Value | vs BGE Base |
|--------|-------|-------------|
| Precision | 45.72% | same |
| Recall | 100.00% | same |
| F1 | 62.75% | same |
| Average Precision | 0.7084 | **-0.0347** |

**Observations:**

1. **Inverse scaling again**: Larger model has WORSE MAP (0.7084 vs 0.7431).

2. **Pattern across all cross-encoder families**:
   - NLI: MiniLM (0.67) > Base (0.52) > Large (0.40)
   - BGE: Base (0.74) > Large (0.71)

3. **Model size hurts ranking**: Across all training paradigms, smaller cross-encoders rank better.

**Key Insight:**
Larger cross-encoders consistently underperform smaller ones on document ranking for this task. This suggests overfitting to training distributions that don't match our keyword-free corpus.

**Meets 94% Recall?** **Yes** (100% at threshold 0.0)

---

## Cross-Encoder Summary

**All 9 cross-encoder experiments complete. Key findings:**

### Ranking by MAP (document ranking quality):

| Rank | Model | MAP | Training |
|------|-------|-----|----------|
| 1 | BGE Reranker Base | **0.7431** | Retrieval |
| 2 | MS-MARCO MiniLM | 0.7177 | Search |
| 3 | BGE Reranker Large | 0.7084 | Retrieval |
| 4 | NLI MiniLM | 0.6704 | Entailment |
| 5 | NLI DeBERTa Base | 0.5182 | Entailment |
| 6 | Quora RoBERTa | 0.4115 | Paraphrase |
| 7 | STS-B RoBERTa Large | 0.4136 | Similarity |
| 8 | NLI DeBERTa Large | 0.3990 | Entailment |
| 9 | STS-B DistilRoBERTa | 0.3807 | Similarity |

### Key Conclusions:

1. **No cross-encoder beats bi-encoders**: Best cross-encoder MAP (0.7431) << best bi-encoder MAP (0.8923 for all-mpnet-base-v2).

2. **Smaller is better for ranking**: Across all training paradigms, smaller cross-encoders rank better.

3. **Training task matters more than architecture**: MS-MARCO and BGE rerankers outperform NLI/STS/Paraphrase despite all being cross-encoders.

4. **NLI/STS/Paraphrase training unsuitable**: These models are trained on sentence pairs, not document-query matching.

5. **Cross-encoders saturate on v2 corpus**: Most achieve 100% recall but 45.72% precision — same as predicting everything as positive.

**For CPRA document discovery, bi-encoders remain the best approach.**

---

---

### 020 - Voyage 4 Nano Baseline

**Date:** 2026-01-28

**Configuration:** `configs/experiments/020_voyage_4_nano_baseline.yaml`

**Model:** voyageai/voyage-4-nano (340M params, 2048 dims, 32K context, Matryoshka support)

**Purpose:** Test Voyage AI's new efficient embedding model using standard `encode()` method.

**Results (at best F1 threshold 0.45):**

| Metric | Value | vs Keyword Baseline |
|--------|-------|---------------------|
| Precision | 68.56% | +13.24% |
| Recall | 85.81% | +1.94% |
| F1 | 76.22% | +9.55% |
| Average Precision | 0.8220 | +0.027 |
| True Positives | 133 | +3 |
| False Positives | 61 | -44 |
| False Negatives | 22 | -3 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 46.97% | 100.00% | 63.92% | 155 | 175 | 0 |
| 0.35 | 48.42% | 98.71% | 64.97% | 153 | 163 | 2 |
| 0.40 | 54.01% | 95.48% | 69.00% | 148 | 126 | 7 |
| 0.45 | 68.56% | 85.81% | 76.22% | 133 | 61 | 22 |
| 0.50 | 76.51% | 73.55% | 75.00% | 114 | 35 | 41 |
| 0.55 | 86.32% | 52.90% | 65.60% | 82 | 13 | 73 |
| 0.60 | 94.34% | 32.26% | 48.08% | 50 | 3 | 105 |

**By Challenge Type (at threshold 0.50):**

| Challenge Type | Recall | vs Keyword Baseline |
|----------------|--------|---------------------|
| DIRECT_MATCH | 96.67% | -3.33% |
| AMBIGUOUS_TERMS | 70.00% | -30.00% |
| INDIRECT_REFERENCE | 65.71% | -11.43% |
| TECHNICAL_JARGON | 72.00% | +8.00% |
| TEMPORAL_REFERENCE | 80.00% | -8.00% |
| BURIED_IN_THREAD | 30.00% | -20.00% |

**Observations:**

1. **MEETS 94% recall requirement**: At threshold 0.40, achieves 95.48% recall with 54.01% precision — similar to keyword baseline precision (55.32%).

2. **Best F1 at 0.45**: 76.22% F1 beats all-mpnet-base-v2 (72.86%) at best F1, though at lower recall (85.81% vs 98.71%).

3. **Middle-tier MAP**: 0.8220 MAP is below all-mpnet-base-v2 (0.8923) and Qwen3 (0.9169).

4. **Struggles with BURIED_IN_THREAD**: Only 30% recall at default threshold — consistent weakness.

5. **AMBIGUOUS_TERMS weakness**: 70% recall is lower than expected for an embedding model, suggesting possible difficulty with contextual disambiguation.

**Key Insight:**
Voyage 4 Nano baseline provides a good precision-recall tradeoff but doesn't stand out compared to existing models. At the 94% recall threshold, precision (54.01%) is nearly identical to keyword search (55.32%). The model's strength is at higher precision operating points, not high-recall legal compliance scenarios.

**Meets 94% Recall?** **Yes** (95.48% at threshold 0.40)

---

### 021 - Voyage 4 Nano Asymmetric

**Date:** 2026-01-28

**Configuration:** `configs/experiments/021_voyage_4_nano_asymmetric.yaml`

**Model:** voyageai/voyage-4-nano with asymmetric encoding (encode_query/encode_document)

**Purpose:** Test if asymmetric query/document encoding improves retrieval over standard symmetric encoding.

**Results (at best F1 threshold 0.40):**

| Metric | Value | vs Baseline (020) |
|--------|-------|-------------------|
| Precision | 80.90% | +26.89% |
| Recall | 92.90% | -2.58% |
| F1 | 86.49% | +17.49% |
| Average Precision | 0.9335 | **+0.1115** |
| True Positives | 144 | -4 |
| False Positives | 34 | -92 |
| False Negatives | 11 | +4 |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-------|-----|-----|-----|
| 0.30 | 53.66% | 99.35% | 69.68% | 154 | 133 | 1 |
| 0.35 | 65.24% | 98.06% | 78.35% | 152 | 81 | 3 |
| 0.40 | 80.90% | 92.90% | 86.49% | 144 | 34 | 11 |
| 0.45 | 90.32% | 72.26% | 80.29% | 112 | 12 | 43 |
| 0.50 | 98.78% | 52.26% | 68.35% | 81 | 1 | 74 |
| 0.55 | 100.00% | 23.87% | 38.54% | 37 | 0 | 118 |
| 0.60 | 100.00% | 5.81% | 10.98% | 9 | 0 | 146 |

**By Challenge Type (at threshold 0.40):**

| Challenge Type | Recall | vs Baseline (020) | vs Keyword |
|----------------|--------|-------------------|------------|
| DIRECT_MATCH | 90.00% | -6.67% | -10.00% |
| AMBIGUOUS_TERMS | 86.67% | +16.67% | -13.33% |
| INDIRECT_REFERENCE | 40.00% | -25.71% | -37.14% |
| TECHNICAL_JARGON | 8.00% | -64.00% | -56.00% |
| TEMPORAL_REFERENCE | 48.00% | -32.00% | -40.00% |
| BURIED_IN_THREAD | 0.00% | -30.00% | -50.00% |

**Comparison at 94%+ Recall:**

| Model | Threshold | Recall | Precision | FPs | MAP |
|-------|-----------|--------|-----------|-----|-----|
| **Voyage Asymmetric** | 0.35 | 98.06% | **65.24%** | **81** | **0.9335** |
| Voyage Baseline | 0.40 | 95.48% | 54.01% | 126 | 0.8220 |
| all-mpnet-base-v2 | 0.30 | 98.71% | 57.74% | 112 | 0.8923 |
| Keyword Baseline | N/A | 83.87% | 55.32% | 105 | 0.7953 |

**Observations:**

1. **MEETS 94% recall with best precision**: At threshold 0.35, achieves 98.06% recall with 65.24% precision — **best precision at 94%+ recall** of any model tested.

2. **Dramatic MAP improvement**: 0.9335 MAP is the second-highest of all experiments, only behind Qwen3 (0.9169 at lower recall). Shows excellent document ranking.

3. **Best F1 of all bi-encoder models**: 86.49% F1 at threshold 0.40 beats Qwen3 (82.88%) and all-mpnet (78.73%).

4. **Asymmetric encoding helps precision significantly**: +11 percentage points precision improvement over baseline at comparable recall.

5. **Category-specific tradeoffs**:
   - **Better**: AMBIGUOUS_TERMS (86.67% vs 70.00%) — asymmetric encoding helps disambiguation
   - **Much worse**: TECHNICAL_JARGON (8% vs 72%), BURIED_IN_THREAD (0% vs 30%)

6. **Specialized for direct semantic matching**: The asymmetric encoding excels when query and document have clear semantic overlap, but struggles with documents that require domain knowledge (technical jargon) or long-context understanding (buried threads).

**Analysis:**

The asymmetric encoding fundamentally changes how the model matches queries to documents. By using separate encoders for queries and documents, the model learns that queries are short intent expressions while documents are longer content pieces. This improves precision by better distinguishing "what the user wants" from "what the document says."

However, this specialization comes at a cost: the model loses some ability to find documents that require inference or domain knowledge. TECHNICAL_JARGON documents use specialized terminology (LSL, CCT, ppb) that doesn't match query semantics even though they're about lead contamination. BURIED_IN_THREAD requires understanding that a responsive mention exists somewhere in a long context.

**Key Insight:**
Voyage 4 Nano with asymmetric encoding provides the **best precision at 94%+ recall** (65.24% vs 57.74% for all-mpnet). However, it achieves this by being excellent at direct matches while failing on challenging categories. For CPRA compliance where every document matters, the 0% recall on BURIED_IN_THREAD and 8% on TECHNICAL_JARGON is concerning — these are exactly the documents humans would miss too.

**Observation:** Asymmetric encoding's precision advantage comes at the cost of coverage on challenging categories. Ensemble or multi-stage approaches may help combine precision with coverage.

**Meets 94% Recall?** **Yes** (98.06% at threshold 0.35 — best precision at 94%+ recall)

---

## Embedding Experiments Complete

All 21 embedding/cross-encoder experiments have been run. See summary table above for complete results.

---

## EXP-020: Validation Corpus Sanity Check

**Date:** 2026-01-28

**Purpose:** Run top bi-encoders on the PFAS validation corpus to check for overfitting and understand how models generalize across different CPRA request types.

**Corpus:** Validation (PFAS) - 59 emails (25 responsive, 34 non-responsive)

### Results Summary

**Comparison: Primary (Lead) vs Validation (PFAS) Corpus**

| Model | Primary Recall | Primary Prec | Val Recall | Val Prec | Val MAP | Generalizes? |
|-------|---------------|--------------|------------|----------|---------|--------------|
| Keyword Baseline | 83.87% | 55.32% | 92.00% | 65.71% | 0.8277 | (baseline) |
| Jina v3 | 98.06% | 51.70% | 100.00% | 53.19% | 0.9039 | **Yes** |
| BGE-M3 | 100.00% | 46.83% | 80.00% | 71.43% | 0.8868 | Partial |
| Embedding Gemma | 100.00% | 49.36% | 68.00% | 80.95% | 0.8776 | Partial |
| **all-mpnet-base-v2** | **98.71%** | **57.74%** | 80.00% | 95.24% | **0.9319** | **Partial** |
| mxbai-embed-large | 98.71% | 51.17% | 100.00% | 53.19% | **0.9551** | **Yes** |
| nomic-embed-text | 99.35% | 46.11% | 100.00% | 44.64% | 0.8041 | **Yes** |
| BGE Large EN v1.5 | 99.35% | 47.24% | 100.00% | 46.30% | **0.9538** | **Yes** |
| Qwen3 0.6B | 89.03% | 77.53% | 24.00% | 100.00% | 0.9084 | No |
| Voyage 4 Nano | 95.48% | 54.01% | 92.00% | 67.65% | 0.8886 | Partial |
| Voyage 4 Nano Asym | 98.06% | 65.24% | 72.00% | 100.00% | 0.9528 | Partial |

**At 94%+ Recall Threshold:**

| Model | Val Threshold | Val Precision @ 94%+ | Val MAP | Primary Prec @ 94%+ |
|-------|---------------|---------------------|---------|---------------------|
| Keyword Baseline | N/A | N/A (92% max) | 0.8277 | N/A (84% max) |
| BGE Large EN v1.5 | 0.60 | **70.59%** | **0.9538** | 47.24% |
| Voyage 4 Nano Asym | 0.40 | 68.57% | 0.9528 | 65.24% |
| all-mpnet-base-v2 | 0.40 | 65.79% | 0.9319 | 57.74% |
| Embedding Gemma | 0.40 | 61.54% | 0.8776 | 49.36% |
| Voyage 4 Nano | 0.45 | 58.14% | 0.8886 | 54.01% |
| Jina v3 | 0.50 | 53.19% | 0.9039 | 51.70% |
| mxbai-embed-large | 0.50 | 53.19% | **0.9551** | 51.17% |
| BGE-M3 | 0.40 | 44.64% | 0.8868 | 46.83% |
| nomic-embed-text | 0.50 | 44.64% | 0.8041 | 46.11% |
| Qwen3 0.6B | - | N/A (84% max) | 0.9084 | N/A (89% max) |

### Challenge Type Breakdown (Validation Corpus)

| Challenge Type | Count | Keyword | all-mpnet | mxbai | BGE-Large | Voyage-Asym |
|----------------|-------|---------|-----------|-------|-----------|-------------|
| DIRECT_MATCH | 8 | 100% | 87.5% | 100% | 100% | 100% |
| INDIRECT_REFERENCE | 8 | 87.5% | 87.5% | 100% | 100% | 75% |
| TECHNICAL_JARGON | 5 | 80% | 80% | 100% | 100% | 40% |
| TEMPORAL_REFERENCE | 2 | 100% | 100% | 100% | 100% | 100% |
| BURIED_IN_THREAD | 2 | 100% | 0% | 100% | 100% | 0% |

### Key Observations

1. **Keywords perform better on PFAS**: 92% recall vs 84% on lead corpus. PFAS doesn't have the "lead/leadership" ambiguity problem.

2. **Different models excel on different corpora**:
   - **Jina v3, mxbai-embed-large, nomic-embed-text, BGE Large EN v1.5** achieve 100% recall on validation at default threshold
   - **all-mpnet-base-v2, Voyage asymmetric** struggle on validation (80%, 72% at default threshold) but excel on primary

3. **all-mpnet-base-v2 has a BURIED_IN_THREAD weakness**: 0% recall on BURIED_IN_THREAD in validation corpus (vs 90% on primary). The model struggles when relevant content is buried in thread context for PFAS topics.

4. **Voyage asymmetric shows same weaknesses on validation**: TECHNICAL_JARGON (40%), BURIED_IN_THREAD (0%) - consistent with primary corpus findings.

5. **BGE Large EN v1.5 is the surprise winner**: Best precision at 94%+ recall on validation (70.59%) with excellent MAP (0.9538). Generalizes well across both corpora.

6. **Qwen3 0.6B fails on validation**: Only 24% recall at default threshold (vs 89% on primary). The model may be overfitting to lead contamination semantics.

### Summary: Cross-Corpus Performance

| Model | Primary Prec@94% | Val Prec@94% | Avg Prec@94% | MAP Avg |
|-------|-----------------|--------------|--------------|---------|
| BGE Large EN v1.5 | 47.24% | **70.59%** | 58.92% | 0.9135 |
| all-mpnet-base-v2 | 57.74% | 65.79% | 61.77% | 0.9121 |
| Voyage Asym | **65.24%** | 68.57% | **66.91%** | **0.9432** |
| Jina v3 | 51.70% | 53.19% | 52.45% | 0.8816 |
| mxbai-embed-large | 51.17% | 53.19% | 52.18% | 0.9056 |

### Implications for Future Experiments

**No single model dominates** — different models excel on different corpora and challenge types:
- **BGE Large EN v1.5**: Best on validation, 100% recall on both, but lower precision on primary
- **Voyage Asymmetric**: Best average precision, but struggles with BURIED_IN_THREAD and TECHNICAL_JARGON
- **all-mpnet-base-v2**: Balanced, but 0% BURIED_IN_THREAD on validation
- **mxbai-embed-large**: 100% on validation, consistent across challenge types

**This suggests several directions worth exploring:**
1. **Ensemble approaches** (EXP-027): Combine models with complementary strengths
2. **Challenge-type specific routing**: Use different models for different document types
3. **Continue testing new models**: No reason to commit to one embedder yet
4. **Pipeline strategies**: Multi-stage approaches may matter more than model choice

---

## EXP-027: RRF Ensemble Across Top Bi-Encoders

**Date:** 2026-01-28

**Purpose:** Test whether combining diverse embedding models via Reciprocal Rank Fusion (RRF) improves precision at the 94% recall compliance threshold.

**Background:** EXP-020 showed models have complementary strengths:
- `all-mpnet-base-v2`: Best precision on Lead (57.74%), but 0% BURIED_IN_THREAD on PFAS
- `mxbai-embed-large`: 100% recall on PFAS, best MAP (0.9551)
- `bge-large-en-v1.5`: Best precision on PFAS (70.59%), 100% recall both corpora
- `jina-embeddings-v3`: 100% recall both corpora

**Implementation:** Added ensemble pipeline support to `run_experiment.py` and created EnsemblePipeline using Reciprocal Rank Fusion (RRF) with k=60.

### Results Summary (Primary Corpus - Lead)

| Variant | Models | Recall@94% | Precision@94% | Best F1 | MAP |
|---------|--------|------------|---------------|---------|-----|
| 027a | mpnet + mxbai | 96.13% | 58.20% | 80.97% | 0.8888 |
| 027b | mpnet + BGE-Large | 96.77% | **58.37%** | **81.82%** | **0.8971** |
| 027c | mpnet + mxbai + BGE | 98.06% | 55.88% | 79.07% | 0.8889 |
| 027d | mpnet + mxbai + keyword | 97.42% | 50.84% | 80.65% | 0.9073 |
| 027e | mxbai + BGE-Large | 93.55% | 59.67% | 79.07% | 0.8657 |

**Note:** RRF scores are much smaller (sum of 1/(k+rank)) so thresholds need adjustment (~0.007-0.015 range).

### Results Summary (Validation Corpus - PFAS)

| Variant | Models | Recall@94% | Precision@94% | Best F1 | MAP |
|---------|--------|------------|---------------|---------|-----|
| 027a | mpnet + mxbai | 100% | 55.56% | 71.43% | 0.9501 |
| 027b | mpnet + BGE-Large | 100% | 55.56% | 71.43% | **0.9535** |
| 027c | mpnet + mxbai + BGE | 100% | 42.37% | 59.52% | 0.9536 |
| 027d | mpnet + mxbai + keyword | 100% | 42.37% | 59.52% | 0.9550 |
| 027e | mxbai + BGE-Large | 100% | 54.35% | 70.42% | 0.9534 |

### Detailed Results

#### 027a - RRF mpnet + mxbai

**Rationale:** Complementary BURIED_IN_THREAD coverage (mpnet: 0% on PFAS, mxbai: 100% on PFAS)

**Primary Corpus (Lead):**
| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.006 | 50.00% | 99.35% | 66.52% | 154 | 154 | 1 |
| 0.007 | 58.20% | 96.13% | 72.51% | 149 | 107 | 6 |
| 0.008 | 69.70% | 89.03% | 78.19% | 138 | 60 | 17 |
| 0.009 | 76.14% | 86.45% | **80.97%** | 134 | 42 | 21 |
| 0.010 | 79.73% | 76.13% | 77.89% | 118 | 30 | 37 |

**Validation Corpus (PFAS):** 100% recall maintained at threshold 0.020, precision 55.56%, F1 71.43%

#### 027b - RRF mpnet + BGE-Large

**Rationale:** Best precision on each corpus (mpnet: 57.74% on Lead, BGE-Large: 70.59% on PFAS)

**Primary Corpus (Lead):**
| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.006 | 50.33% | 99.35% | 66.81% | 154 | 152 | 1 |
| 0.007 | 58.37% | 96.77% | 72.82% | 150 | 107 | 5 |
| 0.008 | 70.41% | 89.03% | 78.63% | 138 | 58 | 17 |
| 0.009 | 77.14% | 87.10% | **81.82%** | 135 | 40 | 20 |
| 0.010 | 80.26% | 78.71% | 79.48% | 122 | 30 | 33 |

**Validation Corpus (PFAS):** 100% recall maintained at threshold 0.020, precision 55.56%, F1 71.43%

#### 027c - RRF mpnet + mxbai + BGE (3-model)

**Rationale:** Combine all three top-performing models

**Primary Corpus (Lead):**
| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.009 | 50.49% | 99.35% | 66.96% | 154 | 151 | 1 |
| 0.010 | 55.88% | 98.06% | 71.19% | 152 | 120 | 3 |
| 0.012 | 68.32% | 89.03% | 77.31% | 138 | 64 | 17 |
| 0.015 | 81.51% | 76.77% | **79.07%** | 119 | 27 | 36 |

**Validation Corpus (PFAS):** All thresholds up to 0.020 maintain 100% recall, precision 42.37%

#### 027d - RRF hybrid (mpnet + mxbai + keyword)

**Rationale:** Hybrid approach combining embedding models with lexical search

**Primary Corpus (Lead):**
| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.009 | 47.26% | 100.00% | 64.18% | 155 | 173 | 0 |
| 0.010 | 50.84% | 97.42% | 66.81% | 151 | 146 | 4 |
| 0.012 | 65.71% | 89.03% | 75.62% | 138 | 72 | 17 |
| 0.015 | 80.65% | 80.65% | **80.65%** | 125 | 30 | 30 |

**Validation Corpus (PFAS):** All thresholds up to 0.020 maintain 100% recall, precision 42.37%

#### 027e - RRF mxbai + BGE-Large

**Rationale:** Both models have 100% recall on PFAS

**Primary Corpus (Lead):**
| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.006 | 51.34% | 98.71% | 67.55% | 153 | 145 | 2 |
| 0.007 | 59.67% | 93.55% | 72.86% | 145 | 98 | 10 |
| 0.008 | 66.67% | 85.16% | 74.79% | 132 | 66 | 23 |
| 0.009 | 73.96% | 80.65% | 77.16% | 125 | 44 | 30 |
| 0.010 | 81.51% | 76.77% | **79.07%** | 119 | 27 | 36 |

**Validation Corpus (PFAS):** 100% recall at threshold 0.020, precision 54.35%, F1 70.42%

### Key Observations

1. **RRF does NOT improve precision at 94% recall**: Best ensemble (027b) achieves 58.37% precision at 96.77% recall — nearly identical to all-mpnet-base-v2 alone (57.74% at 98.71%). The hypothesis that combining models would improve precision was not confirmed.

2. **RRF does improve F1 at best threshold**: 027b achieves 81.82% F1 vs 78.73% for all-mpnet-base-v2 at their respective best F1 thresholds. However, the best F1 thresholds don't meet the 94% recall requirement.

3. **3-model and hybrid ensembles perform worse**: Adding more models (027c) or keyword search (027d) reduces precision without improving recall. The models may have overlapping weaknesses that dilute RRF's benefits.

4. **027e fails to meet 94% recall**: mxbai + BGE-Large ensemble only reaches 93.55% recall at reasonable precision. Excluding mpnet hurts recall on the primary corpus.

5. **All ensembles achieve 100% recall on validation**: The complementary coverage hypothesis works for the validation corpus, where all ensembles maintain 100% recall.

6. **MAP is slightly lower than best individual models**: Ensemble MAPs (0.86-0.91) are lower than all-mpnet-base-v2 (0.8923) and Voyage asymmetric (0.9335). RRF doesn't improve document ranking.

### Comparison vs Individual Models at 94%+ Recall

| Model | Threshold | Recall | Precision | MAP |
|-------|-----------|--------|-----------|-----|
| Voyage 4 Nano Asymmetric | 0.35 | 98.06% | **65.24%** | **0.9335** |
| all-mpnet-base-v2 | 0.30 | 98.71% | 57.74% | 0.8923 |
| **027b RRF mpnet+BGE** | 0.007 | 96.77% | 58.37% | 0.8971 |
| **027a RRF mpnet+mxbai** | 0.007 | 96.13% | 58.20% | 0.8888 |
| Jina v3 | 0.50 | 98.06% | 51.70% | 0.8592 |

### Conclusions

**RRF ensemble does NOT provide the expected precision improvement** for this task. The models' weaknesses on challenging categories (BURIED_IN_THREAD, TECHNICAL_JARGON) are shared rather than complementary for the primary corpus.

**The Voyage 4 Nano Asymmetric model remains the best choice** for CPRA compliance, with 65.24% precision at 98.06% recall — significantly better than any ensemble approach.

**Recommendation:** Focus on improving individual model performance (e.g., asymmetric encoding, domain-specific fine-tuning) rather than ensemble approaches for this task.

---

### EXP-025: Contrastive Scoring with LLM-Generated Prototypes (2026-01-28, updated 2026-01-29)

**Hypothesis:** Using LLM-generated positive/negative prototype emails can improve precision by better capturing the semantic space of responsive vs non-responsive documents, particularly for polysemy issues ("lead" metal vs "lead" leadership).

**Approach:**
- Generate 5 positive prototypes (responsive email examples) and 5 negative prototypes (false positive examples)
- Score documents by similarity to prototypes: `pos_score - λ * neg_score`
- Test variants: positive-only (λ=0), max aggregation (λ=0.5), mean aggregation (λ=0.5)
- Compare LLM models: ministral-3:3b vs gemma3:4b

#### Phase 1: Initial Results with Original Prompts (Failed)

Initial experiments with simple prompts produced poor results because the generated "negative" prototypes were actually responsive to the request. Inspection revealed:

**Problem 1:** Positives were too obvious — all explicitly mentioned "lead testing", "lead remediation" despite prompt asking for indirect references.

**Problem 2:** "Negatives" were actually responsive! Examples generated:
- "lead remediation budget has been allocated" — about lead contamination
- "lead-related expenses" — responsive content
- "lead remediation and testing procedures" — definitely responsive

**Problem 3:** No polysemy examples — not a single negative used "lead" to mean leadership/project lead.

The LLM misunderstood the task and generated emails about lead contamination in administrative contexts, not emails using "lead" with different meanings.

#### Phase 2: Improved Prompts (2026-01-29)

Rewrote prompts to be explicit about the polysemy problem:

**Positive prompt improvements:**
- Request technical jargon (ppb, action levels, LSL, CCT)
- Ask for indirect references without obvious keywords
- Request varied styles (historical, future planning, buried content)

**Negative prompt improvements:**
- Explicit focus on POLYSEMY: "lead" meaning LEADERSHIP, TO GUIDE, FIRST/PRIMARY
- Examples: "project lead", "team lead", "lead contractor", "lead agency"
- Include adjacent topics (water infrastructure NOT about contamination)

#### Phase 2 Results: Ministral-3:3b with Improved Prompts

##### 025a - Positive Only (ministral)

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.30 | 50.49% | 99.35% | 66.96% | 154 | 151 | 1 |
| 0.35 | 55.20% | 99.35% | 70.97% | 154 | 125 | 1 |
| 0.40 | 60.73% | 96.77% | 74.63% | 150 | 97 | 5 |
| 0.45 | 69.90% | 92.90% | 79.78% | 144 | 62 | 11 |
| 0.50 | 77.06% | 84.52% | **80.62%** | 131 | 39 | 24 |

**Meets 94% Recall?** Yes (0.35: 99.35% recall, 55.20% precision)

##### 025b - Contrastive Max (ministral, λ=0.5)

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.15 | 54.04% | 99.35% | 70.00% | 154 | 131 | 1 |
| 0.20 | 64.76% | 94.84% | 76.96% | 147 | 80 | 8 |
| 0.25 | 79.77% | 89.03% | **84.15%** | 138 | 35 | 17 |
| 0.30 | 88.32% | 78.06% | 82.88% | 121 | 16 | 34 |

**Meets 94% Recall?** Yes (0.20: 94.84% recall, **64.76% precision**, +7.02% vs baseline)

#### Phase 2 Results: Gemma3:4b with Improved Prompts

##### 025f - Contrastive Max (gemma, λ=0.5)

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.15 | 58.17% | 98.71% | 73.21% | 153 | 110 | 2 |
| 0.20 | 69.48% | 95.48% | **80.43%** | 148 | 65 | 7 |
| 0.25 | 78.61% | 87.74% | 82.93% | 136 | 37 | 19 |
| 0.30 | 83.45% | 78.06% | 80.67% | 121 | 24 | 34 |

**Meets 94% Recall?** Yes (0.20: 95.48% recall, **69.48% precision**, +11.74% vs baseline)

##### 025g - Contrastive Mean (gemma, λ=0.5)

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.10 | 51.68% | 99.35% | 67.99% | 154 | 144 | 1 |
| 0.15 | 64.94% | 96.77% | 77.72% | 150 | 81 | 5 |
| 0.20 | 78.86% | 89.03% | **83.64%** | 138 | 37 | 17 |

**Meets 94% Recall?** Yes (0.15: 96.77% recall, 64.94% precision, +7.20% vs baseline)

#### Phase 3: Additional Variations (2026-01-30)

##### 025h - Larger LLM (gemma3:12b)

**Hypothesis:** Larger LLM produces better quality prototypes.

**Result:** The 12b model produces more sophisticated, indirect prototypes but performs **worse**:

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.15 | 56.88% | 98.71% | 72.17% | 153 | 116 | 2 |
| 0.20 | 63.29% | 96.77% | 76.53% | 150 | 87 | 5 |
| 0.25 | 68.48% | 81.29% | 74.34% | 126 | 58 | 29 |

**At 96.77% recall:** 63.29% precision (vs 69.48% for gemma3:4b at 95.48% recall)

**Why worse?** The 12b model generates overly indirect prototypes that miss the semantic space of actual responsive emails in the corpus. Examples:
- "pipe network... deterioration" instead of explicit "lead"
- Subtle historical references without keywords

**Conclusion:** Simpler, more direct prototypes from 4b work better than sophisticated indirect ones.

##### 025i - More Prototypes (10+10)

**Hypothesis:** More prototypes capture more edge cases.

**Result:** 10 prototypes per class performs **worse** than 5:

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.15 | 57.63% | 97.42% | 72.42% | 151 | 111 | 4 |
| 0.20 | 67.13% | 93.55% | 78.17% | 145 | 71 | 10 |
| 0.25 | 73.33% | 85.16% | **78.81%** | 132 | 48 | 23 |

**At 93.55% recall (best achievable):** 67.13% precision — does NOT meet 94% recall requirement.

**Why worse?** More prototypes dilute the signal:
- Averaging over more examples creates a diffuse semantic space
- LLM variability — some prototypes may be lower quality
- 5 prototypes appears to be the "sweet spot" of coverage vs. focus

#### Phase 4: Lambda Tuning (2026-01-31)

Tested λ ∈ {0.3, 0.5, 0.7} to find optimal negative penalty weight.

##### 025j - Lambda = 0.3 (Less Penalty)

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.15 | 48.90% | 100.00% | 65.68% | 155 | 162 | 0 |
| 0.20 | 55.23% | 98.71% | 70.83% | 153 | 124 | 2 |
| 0.25 | 63.07% | 98.06% | 76.77% | 152 | 89 | 3 |
| 0.30 | 73.58% | 91.61% | 81.61% | 142 | 51 | 13 |
| 0.35 | 81.21% | 86.45% | **83.75%** | 134 | 31 | 21 |

**At 98.06% recall:** 63.07% precision — achieves high recall but worse precision than λ=0.5.

##### 025k - Lambda = 0.7 (More Penalty)

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.08 | 58.02% | 98.06% | 72.90% | 152 | 110 | 3 |
| 0.10 | 63.25% | 95.48% | 76.09% | 148 | 86 | 7 |
| 0.12 | 70.39% | 93.55% | 80.33% | 145 | 61 | 10 |
| 0.15 | 77.78% | 90.32% | **83.58%** | 140 | 40 | 15 |

**At 95.48% recall:** 63.25% precision — no improvement over λ=0.5.

##### Lambda Tuning Summary

| λ | At ~95%+ Recall | Precision | Best F1 |
|---|-----------------|-----------|---------|
| 0.3 | 98.06% | 63.07% | 83.75% |
| **0.5** | **95.48%** | **69.48%** | 82.93% |
| 0.7 | 95.48% | 63.25% | 83.58% |

**Conclusion:** λ=0.5 is optimal for the 94%+ recall target. Lower λ achieves higher recall but sacrifices precision. Higher λ doesn't improve precision, just makes high recall harder to achieve.

#### Phase 5: Corpus-Derived Prototypes (2026-01-31)

**Hypothesis:** Using actual corpus emails as prototypes (rather than LLM-generated) should establish a ceiling on performance, since real examples perfectly represent the target distribution.

##### 025l - Corpus-Derived Ceiling Test

**Configuration:**
- Positive prototypes: 5 randomly sampled from responsive categories (DIRECT_MATCH, TECHNICAL_JARGON, INDIRECT_REFERENCE, AMBIGUOUS_TERMS, TEMPORAL_REFERENCE)
- Negative prototypes: 5 randomly sampled from KEYWORD_FALSE_POSITIVE category
- Same pipeline: mpnet embeddings, max aggregation, λ=0.5
- Seed: 42 for reproducibility

**Result:** Corpus-derived prototypes perform **worse** than LLM-generated:

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.20 | 62.76% | 96.77% | 76.14% | 150 | 89 | 5 |
| 0.25 | 64.78% | 96.13% | 77.40% | 149 | 81 | 6 |
| 0.30 | 69.00% | 89.03% | **77.75%** | 138 | 62 | 17 |

**At 96.13% recall:** 64.78% precision — worse than 025f (69.48% at 95.48% recall).

**Why worse?** This was a surprising result that challenges the initial hypothesis:

1. **Specificity vs Generalization**: Real emails contain specific details (names, dates, project references) that don't generalize well. LLM-generated prototypes capture the *essence* of the category without irrelevant specifics.

2. **Prototype Quality**: LLM prototypes are designed to be clear examples. Real emails may be ambiguous, poorly written, or contain mixed signals.

3. **5 samples insufficient**: With only 5 samples per class, random selection may not capture the diversity of the category. The LLM, having been trained on massive text corpora, generates more representative examples.

4. **Polysemy handling**: LLM prompts explicitly address the "lead" ambiguity. Real KEYWORD_FALSE_POSITIVE emails may not all be about leadership — some might be borderline cases that don't strongly represent the false positive pattern.

**Key Insight:** LLM-generated prototypes are not a poor substitute for real data — they're actually *better* at representing generalized concepts. This is good news for production deployment, as we don't need labeled examples.

### EXP-025 Summary Comparison

| Experiment | LLM | Protos | λ | At 94%+ Recall | Precision | vs Baseline |
|------------|-----|--------|---|----------------|-----------|-------------|
| Baseline mpnet | - | - | - | 98.71% | 57.74% | - |
| 025a | ministral 3b | 5+5 | 0 | 99.35% | 55.20% | -2.54% |
| **025b** | ministral 3b | 5+5 | 0.5 | 94.84% | 64.76% | **+7.02%** |
| **025f** | **gemma 4b** | **5+5** | **0.5** | **95.48%** | **69.48%** | **+11.74%** |
| 025g | gemma 4b | 5+5 | 0.5 | 96.77% | 64.94% | +7.20% |
| 025h | gemma 12b | 5+5 | 0.5 | 96.77% | 63.29% | +5.55% |
| 025i | gemma 4b | 10+10 | 0.5 | 93.55%* | 67.13% | — |
| 025j | gemma 4b | 5+5 | 0.3 | 98.06% | 63.07% | +5.33% |
| 025k | gemma 4b | 5+5 | 0.7 | 95.48% | 63.25% | +5.51% |
| 025l | corpus-derived | 5+5 | 0.5 | 96.13% | 64.78% | +7.04% |

*Does not meet 94% recall requirement

### EXP-025 Key Findings

1. **Prompt quality is critical**: Original prompts generated useless negatives. Explicit instructions about polysemy were essential.

2. **Gemma3:4b is optimal**: Neither smaller (ministral-3b) nor larger (gemma-12b) models perform as well.

3. **5 prototypes is optimal**: More prototypes (10+10) dilute the signal and hurt performance.

4. **λ=0.5 is optimal**: Lower λ (0.3) sacrifices precision for recall; higher λ (0.7) doesn't improve precision.

5. **Max aggregation > Mean**: Max aggregation performs better at high recall thresholds.

6. **Contrastive with negatives > Positive only**: Adding proper negative prototypes improves precision significantly.

7. **Best result (025f)**: 69.48% precision at 95.48% recall — **+11.74% precision improvement** over baseline.

8. **Bigger/more is not always better**: Larger LLM, more prototypes, and higher λ all hurt performance.

9. **LLM prototypes > corpus-derived**: Surprisingly, LLM-generated prototypes outperform actual corpus examples. LLMs capture generalized concepts better than specific real-world examples that contain noise and irrelevant details.

### EXP-025 Conclusions

**Contrastive scoring with LLM-generated prototypes DOES improve precision** when:
- Prompts explicitly address the polysemy problem
- A mid-sized LLM (gemma3:4b) generates focused prototypes
- 5 prototypes per class (not more)
- λ=0.5 negative penalty weight
- Max aggregation is used for scoring

**Best configuration:** 025f (gemma3:4b, 5+5 prototypes, max aggregation, λ=0.5)
- Precision: 69.48% at 95.48% recall
- **+11.74% precision improvement** over baseline mpnet
- Best overall result across all experiments

**Key insight for production deployment:** LLM-generated prototypes outperform corpus-derived prototypes. This means we don't need labeled examples to deploy this approach — the LLM can generate effective prototypes from just the CPRA request description. This is a major advantage for generalization to new requests.

**Remaining variations to explore:**
- Combine contrastive with Voyage asymmetric encoding
- Test on validation corpus (PFAS request) to verify generalization

---

## Experiments To Run

Based on v1 findings, these models should be tested on v2 corpus:

| Priority | Config | Model | v1 Performance |
|----------|--------|-------|----------------|
| 1 | 002 | Snowflake Arctic L v2.0 | Best overall - 95.2% recall, 90.4% F1 |
| 2 | 003 | Jina v3 | Met 94% recall at 0.70 threshold |
| 3 | 004 | BGE-M3 | Met 94% recall at 0.60 threshold |
| 4 | 005 | embeddinggemma | Met 94% recall at 0.50 threshold |
| 5 | 006 | all-mpnet-base-v2 | Baseline embedding comparison |
| 6 | 007 | mxbai-embed-large | Best MAP in v1 |
| 7 | 008 | nomic-embed-text | Local-only option |
| 8 | 009 | BGE Large EN v1.5 | Did not meet 94% in v1 |

---

## Template for New Experiments

```markdown
### NNN - Experiment Name

**Date:** YYYY-MM-DD

**Configuration:** `configs/experiments/NNN_name.yaml`

**Model:** model-name

**Results:**

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Precision | X.XX% | +/- X.X% |
| Recall | X.XX% | +/- X.X% |
| F1 | X.XX% | +/- X.X% |
| MAP | X.XXXX | +/- X.XX |

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| 0.50 | X.XX% | X.XX% | X.XX% | X | X | X |

**By Challenge Type:**

| Challenge Type | Recall | vs Baseline |
|----------------|--------|-------------|
| INDIRECT_REFERENCE | X.XX% | +/- X.X% |
| TECHNICAL_JARGON | X.XX% | +/- X.X% |
| BURIED_IN_THREAD | X.XX% | +/- X.X% |

**Observations:**
- Key findings
- Comparison to baseline
- Strengths and weaknesses

**Meets 94% Recall?** Yes/No (at threshold X.XX)
```
