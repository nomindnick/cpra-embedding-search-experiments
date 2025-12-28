# Ollama nomic-embed-text

**Pipeline:** Embedding Search (ollama:nomic-embed-text)
**Date:** 2025-12-09 11:25

## Overall Results

| Metric | Value |
|--------|-------|
| Precision | 15.00% |
| Recall | 100.00% |
| F1 | 26.09% |
| MAP | 0.9052 |
| Total Emails | 2,500 |
| Total Responsive | 375 |
| Total Predicted | 2,500 |

## Results by CPRA Request

| Request | Precision | Recall | F1 | TP | FP | FN |
|---------|-----------|--------|----|----|----|----|
| Lead Testing in Water Systems | 3.03% | 100.00% | 5.89% | 75 | 2397 | 0 |
| COVID Relief Fund Allocation | 3.05% | 100.00% | 5.92% | 75 | 2383 | 0 |
| Special Education Program Changes | 3.07% | 98.67% | 5.95% | 74 | 2340 | 1 |
| EdTech Vendor Contracts | 3.12% | 100.00% | 6.05% | 75 | 2331 | 0 |
| Student Safety Incidents | 3.07% | 100.00% | 5.95% | 75 | 2371 | 0 |

## Results by Challenge Type

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|----| ------|---------|
| ambiguous_terms | 20.45% | 100.00% | 33.96% | 18 | 18 |
| near_miss | 19.92% | 98.00% | 33.11% | 50 | 49 |
| indirect_reference | 20.96% | 100.00% | 34.65% | 35 | 35 |
| temporal_mismatch | 22.64% | 100.00% | 36.92% | 24 | 24 |
| partial_match | 20.00% | 100.00% | 33.33% | 12 | 12 |

## Precision@K / Recall@K

| Request | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | P@375 | R@375 |
|---------|------|------|------|------|------|------|------|------|
| Lead Testing in Water Systems | 100.00% | 66.67% | 66.00% | 88.00% | 33.50% | 89.33% | 18.13% | 90.67% |
| COVID Relief Fund Allocation | 100.00% | 66.67% | 74.00% | 98.67% | 37.50% | 100.00% | 20.00% | 100.00% |
| Special Education Program Changes | 90.00% | 60.00% | 57.00% | 76.00% | 31.00% | 82.67% | 17.33% | 86.67% |
| EdTech Vendor Contracts | 98.00% | 65.33% | 67.00% | 89.33% | 35.50% | 94.67% | 19.73% | 98.67% |
| Student Safety Incidents | 100.00% | 66.67% | 72.00% | 96.00% | 36.50% | 97.33% | 19.73% | 98.67% |

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Predicted | TP | FP | FN |
|-----------|-----------|--------|----|-----------|----|----|----|
| 0.30 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.40 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.50 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.60 | 17.90% | 98.40% | 30.28% | 2062 | 369 | 1693 | 6 |
|  **0.70** | 87.24% | 78.40% |  **82.58%** | 337 | 294 | 43 | 81 |
| 0.80 | 100.00% | 4.27% | 8.18% | 16 | 16 | 0 | 359 |

**Best F1 (82.58%) at threshold 0.70** — Precision: 87.24%, Recall: 78.40%
