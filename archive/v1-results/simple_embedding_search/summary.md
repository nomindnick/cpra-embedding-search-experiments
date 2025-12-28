# Simple Embedding Search

**Pipeline:** Embedding Search (st:all-mpnet-base-v2)
**Date:** 2025-12-09 11:02

## Overall Results

| Metric | Value |
|--------|-------|
| Precision | 39.91% |
| Recall | 97.07% |
| F1 | 56.57% |
| MAP | 0.9494 |
| Total Emails | 2,500 |
| Total Responsive | 375 |
| Total Predicted | 912 |

## Results by CPRA Request

| Request | Precision | Recall | F1 | TP | FP | FN |
|---------|-----------|--------|----|----|----|----|
| Lead Testing in Water Systems | 100.00% | 88.00% | 93.62% | 66 | 0 | 9 |
| COVID Relief Fund Allocation | 98.63% | 96.00% | 97.30% | 72 | 1 | 3 |
| Special Education Program Changes | 10.18% | 98.67% | 18.45% | 74 | 653 | 1 |
| EdTech Vendor Contracts | 100.00% | 100.00% | 100.00% | 75 | 0 | 0 |
| Student Safety Incidents | 92.50% | 98.67% | 95.48% | 74 | 6 | 1 |

## Results by Challenge Type

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|----| ------|---------|
| ambiguous_terms | 100.00% | 100.00% | 100.00% | 18 | 18 |
| near_miss | 76.92% | 80.00% | 78.43% | 50 | 40 |
| indirect_reference | 54.24% | 91.43% | 68.09% | 35 | 32 |
| temporal_mismatch | 100.00% | 100.00% | 100.00% | 24 | 24 |
| partial_match | 100.00% | 91.67% | 95.65% | 12 | 11 |

## Precision@K / Recall@K

| Request | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | P@375 | R@375 |
|---------|------|------|------|------|------|------|------|------|
| Lead Testing in Water Systems | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| COVID Relief Fund Allocation | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Special Education Program Changes | 84.00% | 56.00% | 63.00% | 84.00% | 34.00% | 90.67% | 19.47% | 97.33% |
| EdTech Vendor Contracts | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Student Safety Incidents | 100.00% | 66.67% | 74.00% | 98.67% | 37.00% | 98.67% | 20.00% | 100.00% |

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Predicted | TP | FP | FN |
|-----------|-----------|--------|----|-----------|----|----|----|
| 0.30 | 16.82% | 100.00% | 28.80% | 2229 | 375 | 1854 | 0 |
| 0.40 | 23.43% | 99.73% | 37.95% | 1596 | 374 | 1222 | 1 |
| 0.50 | 39.91% | 97.07% | 56.57% | 912 | 364 | 548 | 11 |
|  **0.60** | 86.53% | 80.53% |  **83.43%** | 349 | 302 | 47 | 73 |
| 0.70 | 97.08% | 35.47% | 51.95% | 137 | 133 | 4 | 242 |

**Best F1 (83.43%) at threshold 0.60** — Precision: 86.53%, Recall: 80.53%
