# Snowflake Arctic Embed L v2.0

**Pipeline:** Embedding Search (st:snowflake-arctic-embed-l-v2.0)
**Date:** 2025-12-27 16:51

## Overall Results

| Metric | Value |
|--------|-------|
| Precision | 91.04% |
| Recall | 83.30% |
| F1 | 87.00% |
| MAP | 0.9486 |
| Total Emails | 5,000 |
| Total Responsive | 1,000 |
| Total Predicted | 833 |

## Results by CPRA Request

| Request | Precision | Recall | F1 | TP | FP | FN |
|---------|-----------|--------|----|----|----|----|
| Lead Testing in Water Systems | 100.00% | 61.00% | 75.78% | 122 | 0 | 78 |
| COVID Relief Fund Allocation | 100.00% | 74.50% | 85.39% | 149 | 0 | 51 |
| Special Education Program Changes | 70.50% | 98.00% | 82.01% | 196 | 82 | 4 |
| EdTech Vendor Contracts | 100.00% | 100.00% | 100.00% | 200 | 0 | 0 |
| Student Safety Incidents | 100.00% | 83.00% | 90.71% | 166 | 0 | 34 |

## Results by Challenge Type

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|----| ------|---------|
| ambiguous_terms | 100.00% | 100.00% | 100.00% | 20 | 20 |
| near_miss | 100.00% | 62.16% | 76.67% | 74 | 46 |
| indirect_reference | 0.00% | 0.00% | 0.00% | 60 | 0 |
| temporal_mismatch | 66.67% | 100.00% | 80.00% | 50 | 50 |
| partial_match | 100.00% | 96.55% | 98.25% | 29 | 28 |

## Precision@K / Recall@K

| Request | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | P@375 | R@375 |
|---------|------|------|------|------|------|------|------|------|
| Lead Testing in Water Systems | 100.00% | 25.00% | 100.00% | 50.00% | 92.50% | 92.50% | 52.27% | 98.00% |
| COVID Relief Fund Allocation | 100.00% | 25.00% | 100.00% | 50.00% | 79.50% | 79.50% | 42.40% | 79.50% |
| Special Education Program Changes | 100.00% | 25.00% | 100.00% | 50.00% | 98.00% | 98.00% | 52.80% | 99.00% |
| EdTech Vendor Contracts | 100.00% | 25.00% | 100.00% | 50.00% | 100.00% | 100.00% | 53.33% | 100.00% |
| Student Safety Incidents | 100.00% | 25.00% | 100.00% | 50.00% | 95.50% | 95.50% | 50.93% | 95.50% |

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Predicted | TP | FP | FN |
|-----------|-----------|--------|----|-----------|----|----|----|
| 0.30 | 9.10% | 95.00% | 16.60% | 3631 | 950 | 9494 | 50 |
| 0.40 | 29.03% | 94.50% | 44.42% | 2145 | 945 | 2310 | 55 |
|  **0.50** | 91.04% | 83.30% |  **87.00%** | 833 | 833 | 82 | 167 |
| 0.60 | 100.00% | 48.00% | 64.86% | 480 | 480 | 0 | 520 |
| 0.70 | 100.00% | 32.70% | 49.28% | 327 | 327 | 0 | 673 |
| 0.80 | 100.00% | 8.20% | 15.16% | 82 | 82 | 0 | 918 |

**Best F1 (87.00%) at threshold 0.50** — Precision: 91.04%, Recall: 83.30%
