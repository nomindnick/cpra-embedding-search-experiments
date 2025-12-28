# Ollama mxbai-embed-large

**Pipeline:** Embedding Search (ollama:mxbai-embed-large)
**Date:** 2025-12-09 11:41

## Overall Results

| Metric | Value |
|--------|-------|
| Precision | 15.16% |
| Recall | 100.00% |
| F1 | 26.33% |
| MAP | 0.9818 |
| Total Emails | 2,500 |
| Total Responsive | 375 |
| Total Predicted | 2,473 |

## Results by CPRA Request

| Request | Precision | Recall | F1 | TP | FP | FN |
|---------|-----------|--------|----|----|----|----|
| Lead Testing in Water Systems | 3.19% | 100.00% | 6.19% | 75 | 2273 | 0 |
| COVID Relief Fund Allocation | 10.87% | 100.00% | 19.61% | 75 | 615 | 0 |
| Special Education Program Changes | 3.09% | 100.00% | 5.99% | 75 | 2354 | 0 |
| EdTech Vendor Contracts | 4.42% | 100.00% | 8.47% | 75 | 1622 | 0 |
| Student Safety Incidents | 3.20% | 100.00% | 6.20% | 75 | 2271 | 0 |

## Results by Challenge Type

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|----| ------|---------|
| ambiguous_terms | 22.50% | 100.00% | 36.73% | 18 | 18 |
| near_miss | 22.73% | 100.00% | 37.04% | 50 | 50 |
| indirect_reference | 21.60% | 100.00% | 35.53% | 35 | 35 |
| temporal_mismatch | 28.24% | 100.00% | 44.04% | 24 | 24 |
| partial_match | 24.49% | 100.00% | 39.34% | 12 | 12 |

## Precision@K / Recall@K

| Request | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | P@375 | R@375 |
|---------|------|------|------|------|------|------|------|------|
| Lead Testing in Water Systems | 100.00% | 66.67% | 72.00% | 96.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| COVID Relief Fund Allocation | 100.00% | 66.67% | 73.00% | 97.33% | 37.00% | 98.67% | 20.00% | 100.00% |
| Special Education Program Changes | 98.00% | 65.33% | 74.00% | 98.67% | 37.50% | 100.00% | 20.00% | 100.00% |
| EdTech Vendor Contracts | 100.00% | 66.67% | 75.00% | 100.00% | 37.50% | 100.00% | 20.00% | 100.00% |
| Student Safety Incidents | 100.00% | 66.67% | 74.00% | 98.67% | 37.00% | 98.67% | 19.73% | 98.67% |

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Predicted | TP | FP | FN |
|-----------|-----------|--------|----|-----------|----|----|----|
| 0.30 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.40 | 15.00% | 100.00% | 26.09% | 2500 | 375 | 2125 | 0 |
| 0.50 | 15.16% | 100.00% | 26.33% | 2473 | 375 | 2098 | 0 |
| 0.60 | 27.34% | 99.73% | 42.91% | 1368 | 374 | 994 | 1 |
|  **0.70** | 93.56% | 81.33% |  **87.02%** | 326 | 305 | 21 | 70 |
| 0.80 | 100.00% | 11.73% | 21.00% | 44 | 44 | 0 | 331 |

**Best F1 (87.02%) at threshold 0.70** — Precision: 93.56%, Recall: 81.33%
