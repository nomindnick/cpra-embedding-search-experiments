# Baseline Keyword Search

**Pipeline:** Keyword Search
**Date:** 2025-12-27 16:50

## Overall Results

| Metric | Value |
|--------|-------|
| Precision | 93.39% |
| Recall | 53.70% |
| F1 | 68.19% |
| MAP | 0.5188 |
| Total Emails | 5,000 |
| Total Responsive | 1,000 |
| Total Predicted | 571 |

## Results by CPRA Request

| Request | Precision | Recall | F1 | TP | FP | FN |
|---------|-----------|--------|----|----|----|----|
| Lead Testing in Water Systems | 100.00% | 40.00% | 57.14% | 80 | 0 | 120 |
| COVID Relief Fund Allocation | 89.29% | 37.50% | 52.82% | 75 | 9 | 125 |
| Special Education Program Changes | 98.67% | 74.00% | 84.57% | 148 | 2 | 52 |
| EdTech Vendor Contracts | 98.76% | 79.50% | 88.09% | 159 | 2 | 41 |
| Student Safety Incidents | 75.00% | 37.50% | 50.00% | 75 | 25 | 125 |

## Results by Challenge Type

| Challenge Type | Precision | Recall | F1 | Total | Correct |
|----------------|-----------|--------|----| ------|---------|
| ambiguous_terms | 100.00% | 100.00% | 100.00% | 20 | 20 |
| near_miss | 100.00% | 62.16% | 76.67% | 74 | 46 |
| indirect_reference | 0.00% | 0.00% | 0.00% | 60 | 0 |
| temporal_mismatch | 100.00% | 100.00% | 100.00% | 50 | 50 |
| partial_match | 100.00% | 96.55% | 98.25% | 29 | 28 |

## Precision@K / Recall@K

| Request | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | P@375 | R@375 |
|---------|------|------|------|------|------|------|------|------|
| Lead Testing in Water Systems | 100.00% | 25.00% | 100.00% | 40.00% | 100.00% | 40.00% | 100.00% | 40.00% |
| COVID Relief Fund Allocation | 100.00% | 25.00% | 89.29% | 37.50% | 89.29% | 37.50% | 89.29% | 37.50% |
| Special Education Program Changes | 100.00% | 25.00% | 100.00% | 50.00% | 98.67% | 74.00% | 98.67% | 74.00% |
| EdTech Vendor Contracts | 100.00% | 25.00% | 100.00% | 50.00% | 98.76% | 79.50% | 98.76% | 79.50% |
| Student Safety Incidents | 78.00% | 19.50% | 75.00% | 37.50% | 75.00% | 37.50% | 75.00% | 37.50% |
