# BGE Large English v1.5

**Pipeline:** Embedding Search (st:bge-large-en-v1.5)
**Date:** 2025-12-09 20:40

## Overall Results

| Metric | Value |
|--------|-------|
| Precision | 15.00% |
| Recall | 100.00% |
| F1 | 26.09% |
| MAP | 0.9717 |
| Total Emails | 2,500 |
| Total Responsive | 375 |
| Total Predicted | 2,500 |

## Results by CPRA Request

| Request | Precision | Recall | F1 | TP | FP | FN |
|---------|-----------|--------|----|----|----|----|
| Lead Testing in Water Systems | 3.00% | 100.00% | 5.83% | 75 | 2422 | 0 |
| COVID Relief Fund Allocation | 4.35% | 100.00% | 8.34% | 75 | 1648 | 0 |
| Special Education Program Changes | 3.00% | 100.00% | 5.83% | 75 | 2424 | 0 |
| EdTech Vendor Contracts | 3.26% | 100.00% | 6.31% | 75 | 2227 | 0 |
| Student Safety Incidents | 3.01% | 100.00% | 5.85% | 75 | 2414 | 0 |

## Results by Challenge Type

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|----| ------|---------|
| ambiguous_terms | 20.00% | 100.00% | 33.33% | 18 | 18 |
| near_miss | 20.58% | 100.00% | 34.13% | 50 | 50 |
| indirect_reference | 20.35% | 100.00% | 33.82% | 35 | 35 |
| temporal_mismatch | 21.43% | 100.00% | 35.29% | 24 | 24 |
| partial_match | 20.34% | 100.00% | 33.80% | 12 | 12 |

## Precision@K / Recall@K

| Request | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | P@375 | R@375 |
|---------|------|------|------|------|------|------|------|------|
| Lead Testing in Water Systems | 100.00% | 66.67% | 71.00% | 94.67% | 35.50% | 94.67% | 20.00% | 100.00% |
| COVID Relief Fund Allocation | 100.00% | 66.67% | 72.00% | 96.00% | 36.50% | 97.33% | 19.73% | 98.67% |
| Special Education Program Changes | 98.00% | 65.33% | 74.00% | 98.67% | 37.00% | 98.67% | 20.00% | 100.00% |
| EdTech Vendor Contracts | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Student Safety Incidents | 100.00% | 66.67% | 74.00% | 98.67% | 37.00% | 98.67% | 19.73% | 98.67% |

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Predicted | TP | FP | FN |
|-----------|-----------|--------|----|-----------|----|----|----|
| 0.30 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.40 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.50 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.60 | 20.51% | 100.00% | 34.04% | 1828 | 375 | 1453 | 0 |
|  **0.70** | 91.47% | 82.93% |  **86.99%** | 340 | 311 | 29 | 64 |
| 0.80 | 100.00% | 7.73% | 14.36% | 29 | 29 | 0 | 346 |

**Best F1 (86.99%) at threshold 0.70** — Precision: 91.47%, Recall: 82.93%
