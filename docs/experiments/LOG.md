# Experiment Log

This document tracks all experiments, their key findings, and learnings.

## Summary Table

| #   | Name              | Date | Embedding Model | LLM | F1    | vs Baseline | Key Finding |
| --- | ----------------- | ---- | --------------- | --- | ----- | ----------- | ----------- |
| 001 | Baseline Keyword  | TBD  | N/A             | N/A | TBD   | —           | TBD         |

## Experiment Details

---

### 001 - Baseline Keyword Search

**Date:** TBD

**Hypothesis:** Establish baseline performance with traditional keyword matching to understand what we're trying to beat.

**Configuration:** `configs/experiments/001_baseline_keyword.yaml`

**Approach:**
- Use keywords from CPRA request definitions
- Boolean OR matching
- No semantic understanding

**Results:**

| Metric       | Overall | Lead Testing | COVID Funds | SpEd | EdTech | Safety |
| ------------ | ------- | ------------ | ----------- | ---- | ------ | ------ |
| Precision    | TBD     | TBD          | TBD         | TBD  | TBD    | TBD    |
| Recall       | TBD     | TBD          | TBD         | TBD  | TBD    | TBD    |
| F1           | TBD     | TBD          | TBD         | TBD  | TBD    | TBD    |

**By Challenge Type:**

| Challenge Type     | Precision | Recall | F1  |
| ------------------ | --------- | ------ | --- |
| Direct Match       | TBD       | TBD    | TBD |
| Ambiguous Terms    | TBD       | TBD    | TBD |
| Near Miss          | TBD       | TBD    | TBD |
| Indirect Reference | TBD       | TBD    | TBD |

**Observations:**
- TBD

**Next Steps:**
- TBD

---

## Template for New Experiments

```markdown
### NNN - Experiment Name

**Date:** YYYY-MM-DD

**Hypothesis:** What we're testing and why.

**Configuration:** `configs/experiments/NNN_name.yaml`

**Key Changes from Previous:**
- What's different from the last experiment

**Results:**

| Metric    | Overall | vs 001 Baseline |
| --------- | ------- | --------------- |
| Precision | X.XX    | +/- X.X%        |
| Recall    | X.XX    | +/- X.X%        |
| F1        | X.XX    | +/- X.X%        |

**Observations:**
- What worked
- What didn't
- Surprising findings

**Next Steps:**
- What to try next based on these results
```
