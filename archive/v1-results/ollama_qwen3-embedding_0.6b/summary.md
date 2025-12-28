# Ollama Qwen3-Embedding 0.6B

**Pipeline:** Embedding Search (ollama:qwen3-embedding-0.6b)
**Date:** 2025-12-09 14:18

## Overall Results

| Metric | Value |
|--------|-------|
| Precision | 96.85% |
| Recall | 90.13% |
| F1 | 93.37% |
| MAP | 0.9610 |
| Total Emails | 2,500 |
| Total Responsive | 375 |
| Total Predicted | 349 |

## Results by CPRA Request

| Request | Precision | Recall | F1 | TP | FP | FN |
|---------|-----------|--------|----|----|----|----|
| Lead Testing in Water Systems | 100.00% | 88.00% | 93.62% | 66 | 0 | 9 |
| COVID Relief Fund Allocation | 100.00% | 92.00% | 95.83% | 69 | 0 | 6 |
| Special Education Program Changes | 91.78% | 89.33% | 90.54% | 67 | 6 | 8 |
| EdTech Vendor Contracts | 100.00% | 100.00% | 100.00% | 75 | 0 | 0 |
| Student Safety Incidents | 92.42% | 81.33% | 86.52% | 61 | 5 | 14 |

## Results by Challenge Type

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|----| ------|---------|
| ambiguous_terms | 100.00% | 100.00% | 100.00% | 18 | 18 |
| near_miss | 100.00% | 42.00% | 59.15% | 50 | 21 |
| indirect_reference | 100.00% | 54.29% | 70.37% | 35 | 19 |
| temporal_mismatch | 100.00% | 100.00% | 100.00% | 24 | 24 |
| partial_match | 100.00% | 58.33% | 73.68% | 12 | 7 |

## Precision@K / Recall@K

| Request | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | P@375 | R@375 |
|---------|------|------|------|------|------|------|------|------|
| Lead Testing in Water Systems | 100.00% | 66.67% | 67.00% | 89.33% | 35.00% | 93.33% | 19.73% | 98.67% |
| COVID Relief Fund Allocation | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Special Education Program Changes | 100.00% | 66.67% | 74.00% | 98.67% | 37.50% | 100.00% | 20.00% | 100.00% |
| EdTech Vendor Contracts | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Student Safety Incidents | 100.00% | 66.67% | 66.00% | 88.00% | 34.00% | 90.67% | 19.20% | 96.00% |

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Predicted | TP | FP | FN |
|-----------|-----------|--------|----|-----------|----|----|----|
| 0.30 | 18.01% | 100.00% | 30.53% | 2082 | 375 | 1707 | 0 |
| 0.40 | 51.13% | 96.80% | 66.91% | 710 | 363 | 347 | 12 |
|  **0.50** | 96.85% | 90.13% |  **93.37%** | 349 | 338 | 11 | 37 |
| 0.60 | 100.00% | 75.20% | 85.84% | 282 | 282 | 0 | 93 |
| 0.70 | 100.00% | 6.40% | 12.03% | 24 | 24 | 0 | 351 |
| 0.80 | 0.00% | 0.00% | 0.00% | 0 | 0 | 0 | 375 |

**Best F1 (93.37%) at threshold 0.50** — Precision: 96.85%, Recall: 90.13%
