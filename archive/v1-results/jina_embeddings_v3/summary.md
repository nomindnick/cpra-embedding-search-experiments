# Jina Embeddings v3

**Pipeline:** Embedding Search (st:jina-embeddings-v3)
**Date:** 2025-12-09 18:41

## Overall Results

| Metric | Value |
|--------|-------|
| Precision | 15.00% |
| Recall | 100.00% |
| F1 | 26.09% |
| MAP | 0.9679 |
| Total Emails | 2,500 |
| Total Responsive | 375 |
| Total Predicted | 2,500 |

## Results by CPRA Request

| Request | Precision | Recall | F1 | TP | FP | FN |
|---------|-----------|--------|----|----|----|----|
| Lead Testing in Water Systems | 3.41% | 100.00% | 6.60% | 75 | 2123 | 0 |
| COVID Relief Fund Allocation | 3.06% | 100.00% | 5.94% | 75 | 2375 | 0 |
| Special Education Program Changes | 3.02% | 100.00% | 5.86% | 75 | 2410 | 0 |
| EdTech Vendor Contracts | 3.47% | 100.00% | 6.70% | 75 | 2088 | 0 |
| Student Safety Incidents | 3.01% | 100.00% | 5.84% | 75 | 2417 | 0 |

## Results by Challenge Type

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|----| ------|---------|
| ambiguous_terms | 20.00% | 100.00% | 33.33% | 18 | 18 |
| near_miss | 20.24% | 100.00% | 33.67% | 50 | 50 |
| indirect_reference | 21.60% | 100.00% | 35.53% | 35 | 35 |
| temporal_mismatch | 20.34% | 100.00% | 33.80% | 24 | 24 |
| partial_match | 20.00% | 100.00% | 33.33% | 12 | 12 |

## Precision@K / Recall@K

| Request | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | P@375 | R@375 |
|---------|------|------|------|------|------|------|------|------|
| Lead Testing in Water Systems | 100.00% | 66.67% | 72.00% | 96.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| COVID Relief Fund Allocation | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Special Education Program Changes | 100.00% | 66.67% | 70.00% | 93.33% | 36.50% | 97.33% | 19.73% | 98.67% |
| EdTech Vendor Contracts | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Student Safety Incidents | 100.00% | 66.67% | 68.00% | 90.67% | 36.00% | 96.00% | 19.47% | 97.33% |

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Predicted | TP | FP | FN |
|-----------|-----------|--------|----|-----------|----|----|----|
| 0.30 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.40 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.50 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.60 | 16.59% | 99.73% | 28.44% | 2255 | 374 | 1881 | 1 |
|  **0.70** | 68.92% | 95.20% |  **79.96%** | 518 | 357 | 161 | 18 |
| 0.80 | 100.00% | 54.93% | 70.91% | 206 | 206 | 0 | 169 |

**Best F1 (79.96%) at threshold 0.70** — Precision: 68.92%, Recall: 95.20%
