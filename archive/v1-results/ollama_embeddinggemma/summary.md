# Ollama embeddinggemma

**Pipeline:** Embedding Search (ollama:embeddinggemma)
**Date:** 2025-12-09 12:15

## Overall Results

| Metric | Value |
|--------|-------|
| Precision | 63.70% |
| Recall | 95.47% |
| F1 | 76.41% |
| MAP | 0.9781 |
| Total Emails | 2,500 |
| Total Responsive | 375 |
| Total Predicted | 562 |

## Results by CPRA Request

| Request | Precision | Recall | F1 | TP | FP | FN |
|---------|-----------|--------|----|----|----|----|
| Lead Testing in Water Systems | 98.51% | 88.00% | 92.96% | 66 | 1 | 9 |
| COVID Relief Fund Allocation | 100.00% | 92.00% | 95.83% | 69 | 0 | 6 |
| Special Education Program Changes | 31.91% | 100.00% | 48.39% | 75 | 160 | 0 |
| EdTech Vendor Contracts | 93.75% | 100.00% | 96.77% | 75 | 5 | 0 |
| Student Safety Incidents | 58.40% | 97.33% | 73.00% | 73 | 52 | 2 |

## Results by Challenge Type

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|----| ------|---------|
| ambiguous_terms | 100.00% | 100.00% | 100.00% | 18 | 18 |
| near_miss | 97.56% | 80.00% | 87.91% | 50 | 40 |
| indirect_reference | 96.43% | 77.14% | 85.71% | 35 | 27 |
| temporal_mismatch | 100.00% | 100.00% | 100.00% | 24 | 24 |
| partial_match | 100.00% | 100.00% | 100.00% | 12 | 12 |

## Precision@K / Recall@K

| Request | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | P@375 | R@375 |
|---------|------|------|------|------|------|------|------|------|
| Lead Testing in Water Systems | 100.00% | 66.67% | 71.00% | 94.67% | 36.00% | 96.00% | 19.47% | 97.33% |
| COVID Relief Fund Allocation | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Special Education Program Changes | 100.00% | 66.67% | 71.00% | 94.67% | 37.50% | 100.00% | 20.00% | 100.00% |
| EdTech Vendor Contracts | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Student Safety Incidents | 100.00% | 66.67% | 72.00% | 96.00% | 37.00% | 98.67% | 19.73% | 98.67% |

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Predicted | TP | FP | FN |
|-----------|-----------|--------|----|-----------|----|----|----|
| 0.30 | 15.03% | 100.00% | 26.13% | 2495 | 375 | 2120 | 0 |
| 0.40 | 21.69% | 100.00% | 35.65% | 1729 | 375 | 1354 | 0 |
| 0.50 | 63.70% | 95.47% | 76.41% | 562 | 358 | 204 | 17 |
|  **0.60** | 99.00% | 78.93% |  **87.83%** | 299 | 296 | 3 | 79 |
| 0.70 | 100.00% | 8.80% | 16.18% | 33 | 33 | 0 | 342 |
| 0.80 | 0.00% | 0.00% | 0.00% | 0 | 0 | 0 | 375 |

**Best F1 (87.83%) at threshold 0.60** — Precision: 99.00%, Recall: 78.93%
