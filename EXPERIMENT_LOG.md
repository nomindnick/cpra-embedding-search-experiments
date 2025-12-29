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
