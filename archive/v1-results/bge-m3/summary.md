# BGE-M3

**Pipeline:** Embedding Search (st:bge-m3)
**Date:** 2025-12-09 19:18

## Overall Results

| Metric | Value |
|--------|-------|
| Precision | 16.70% |
| Recall | 100.00% |
| F1 | 28.63% |
| MAP | 0.9713 |
| Total Emails | 2,500 |
| Total Responsive | 375 |
| Total Predicted | 2,245 |

## Results by CPRA Request

| Request | Precision | Recall | F1 | TP | FP | FN |
|---------|-----------|--------|----|----|----|----|
| Lead Testing in Water Systems | 6.06% | 100.00% | 11.43% | 75 | 1162 | 0 |
| COVID Relief Fund Allocation | 25.42% | 100.00% | 40.54% | 75 | 220 | 0 |
| Special Education Program Changes | 3.79% | 100.00% | 7.31% | 75 | 1903 | 0 |
| EdTech Vendor Contracts | 10.11% | 100.00% | 18.36% | 75 | 667 | 0 |
| Student Safety Incidents | 5.22% | 100.00% | 9.91% | 75 | 1363 | 0 |

## Results by Challenge Type

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|----| ------|---------|
| ambiguous_terms | 43.90% | 100.00% | 61.02% | 18 | 18 |
| near_miss | 33.33% | 100.00% | 50.00% | 50 | 50 |
| indirect_reference | 36.84% | 100.00% | 53.85% | 35 | 35 |
| temporal_mismatch | 30.77% | 100.00% | 47.06% | 24 | 24 |
| partial_match | 28.57% | 100.00% | 44.44% | 12 | 12 |

## Precision@K / Recall@K

| Request | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | P@375 | R@375 |
|---------|------|------|------|------|------|------|------|------|
| Lead Testing in Water Systems | 100.00% | 66.67% | 70.00% | 93.33% | 35.50% | 94.67% | 19.73% | 98.67% |
| COVID Relief Fund Allocation | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Special Education Program Changes | 100.00% | 66.67% | 72.00% | 96.00% | 37.00% | 98.67% | 19.73% | 98.67% |
| EdTech Vendor Contracts | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Student Safety Incidents | 100.00% | 66.67% | 70.00% | 93.33% | 36.00% | 96.00% | 19.73% | 98.67% |

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Predicted | TP | FP | FN |
|-----------|-----------|--------|----|-----------|----|----|----|
| 0.30 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.40 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.50 | 16.70% | 100.00% | 28.63% | 2245 | 375 | 1870 | 0 |
| 0.60 | 43.73% | 94.93% | 59.88% | 814 | 356 | 458 | 19 |
|  **0.70** | 98.10% | 68.80% |  **80.88%** | 263 | 258 | 5 | 117 |
| 0.80 | 100.00% | 0.53% | 1.06% | 2 | 2 | 0 | 373 |

**Best F1 (80.88%) at threshold 0.70** — Precision: 98.10%, Recall: 68.80%
