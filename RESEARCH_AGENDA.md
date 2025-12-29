# Research Agenda

This document tracks planned experiments and research directions.

## Current Best Result

**all-mpnet-base-v2** at threshold 0.30:
- Recall: 98.71%
- Precision: 57.74%
- F1: 72.86%
- MAP: 0.8923
- False Positives: 112

---

## Cross-Encoder Experiments

Testing whether different cross-encoder training objectives perform better than MS-MARCO (which failed due to lexical bias).

### NLI-Trained Cross-Encoders

These models learn semantic entailment — "Does A mean B?" — without relying on keyword overlap.

| # | Config | Model | Training Task | Status |
|---|--------|-------|---------------|--------|
| 012 | `012_cross_encoder_nli_deberta_base.yaml` | cross-encoder/nli-deberta-v3-base | NLI (SNLI, MultiNLI) | **Complete** |
| 013 | `013_cross_encoder_nli_deberta_large.yaml` | cross-encoder/nli-deberta-v3-large | NLI | **Complete** |
| 014 | `014_cross_encoder_nli_minilm.yaml` | cross-encoder/nli-MiniLM2-L6-H768 | NLI | **Complete** |

**Hypothesis:** NLI training teaches semantic equivalence without lexical shortcuts. Should perform better on keyword-free v2 corpus.

**Results:**
- **012 (Base):** 100% recall, 45.72% precision, MAP 0.5182 — scores everything as "entails"
- **013 (Large):** 100% recall, 45.86% precision, MAP 0.3990 — even worse ranking!
- **014 (MiniLM):** 100% recall, 45.72% precision, MAP 0.6704 — best ranking among NLI models

**Conclusion:** NLI training is fundamentally unsuited for document relevance. All models saturate at 100% recall with ~46% precision. Smaller models rank better (inverse scaling). The entailment task is too broad for relevance.

### STS-Trained Cross-Encoders

These models learn semantic similarity scores from paraphrase datasets.

| # | Config | Model | Training Task | Status |
|---|--------|-------|---------------|--------|
| 015 | `015_cross_encoder_stsb_roberta_large.yaml` | cross-encoder/stsb-roberta-large | STS Benchmark | **Complete** |
| 016 | `016_cross_encoder_stsb_distilroberta.yaml` | cross-encoder/stsb-distilroberta-base | STS Benchmark | **Complete** |

**Hypothesis:** STS training on paraphrase detection may help recognize semantic similarity without word overlap.

**Results:**
- **015 (Large):** 100% recall, 45.72% precision, MAP 0.4136 — worst cross-encoder
- **016 (Distil):** 100% recall, 45.72% precision, MAP 0.3807 — worst overall

**Conclusion:** STS training is unsuitable for document retrieval. Models trained on sentence-level similarity give uniformly low scores to document-query pairs (all scores < 1.0 on 0-5 scale).

### Paraphrase-Trained Cross-Encoders

Explicitly trained to detect when different words mean the same thing.

| # | Config | Model | Training Task | Status |
|---|--------|-------|---------------|--------|
| 017 | `017_cross_encoder_quora.yaml` | cross-encoder/quora-roberta-large | Quora duplicate questions | **Complete** |

**Hypothesis:** Paraphrase detection is exactly what we need — recognizing same meaning with different words.

**Result (017):** Complete failure — max 1.29% recall. Model expects similar-length texts (question vs question), not query vs long document.

### BGE Rerankers

Different architecture but still retrieval-focused. May have same lexical bias as MS-MARCO.

| # | Config | Model | Training Task | Status |
|---|--------|-------|---------------|--------|
| 018 | `018_bge_reranker_base.yaml` | BAAI/bge-reranker-base | Retrieval | **Complete** |
| 019 | `019_bge_reranker_large.yaml` | BAAI/bge-reranker-large | Retrieval | **Complete** |

**Hypothesis:** May perform similarly to MS-MARCO (poorly) due to retrieval training, but worth testing.

**Results:**
- **018 (Base):** 100% recall, 45.72% precision, MAP 0.7431 — **best cross-encoder**
- **019 (Large):** 100% recall, 45.72% precision, MAP 0.7084 — inverse scaling again

**Conclusion:** BGE rerankers are the best cross-encoders but still can't match bi-encoder performance. Smaller models rank better.

---

## Future Research Directions

### Two-Stage Pipelines

Use high-recall bi-encoder + precision-focused re-ranker:

| Approach | Stage 1 | Stage 2 | Rationale |
|----------|---------|---------|-----------|
| Bi-encoder + Cross-encoder | all-mpnet (98.7% recall) | NLI cross-encoder | Best of both worlds |
| Bi-encoder + Qwen3 | all-mpnet (98.7% recall) | Qwen3 (best MAP) | Use Qwen3's ranking |
| Bi-encoder + LLM | all-mpnet (98.7% recall) | Claude/GPT | LLM as final judge |

### Query Expansion

Expand the CPRA request text to improve matching:

| Approach | Method | Rationale |
|----------|--------|-----------|
| Synonym expansion | Add synonyms to query | Cover more vocabulary |
| LLM rewriting | Claude rewrites query multiple ways | Generate paraphrases |
| Hypothetical document | Generate what a responsive email might look like | Match against archetype |

### Ensemble Methods

Combine multiple models:

| Approach | Method | Rationale |
|----------|--------|-----------|
| Score averaging | Average scores from multiple bi-encoders | Reduce individual model bias |
| Voting | Document responsive if N of M models agree | Consensus filtering |
| Stacking | Train meta-model on model outputs | Learn optimal combination |

### Fine-Tuning

Train models specifically for CPRA document discovery:

| Approach | Data Needed | Effort |
|----------|-------------|--------|
| Fine-tune bi-encoder | Labeled CPRA corpus | Medium |
| Fine-tune cross-encoder | Labeled CPRA corpus | Medium |
| Train from scratch | Large CPRA corpus | High |

---

## Completed Experiments

| # | Name | Result | Meets 94%? |
|---|------|--------|------------|
| 001 | Keyword Baseline | 83.87% recall, 55.32% precision | No |
| 002 | Snowflake Arctic L v2.0 | 81.29% recall, 70.39% precision | No |
| 003 | Jina v3 | 98.06% recall, 51.70% precision | Yes |
| 004 | BGE-M3 | 100% recall, 46.83% precision | Yes |
| 005 | Embedding Gemma | 100% recall, 49.36% precision | Yes |
| 006 | all-mpnet-base-v2 | **98.71% recall, 57.74% precision** | **Yes (Best)** |
| 007 | mxbai-embed-large | 98.71% recall, 51.17% precision | Yes |
| 008 | nomic-embed-text | 99.35% recall, 46.11% precision | Yes |
| 009 | BGE Large EN v1.5 | 99.35% recall, 47.24% precision | Yes |
| 010 | Qwen3 Embedding 0.6B | 89.03% recall, 77.53% precision | No |
| 011 | Cross-Encoder MiniLM (MS-MARCO) | 98.71% recall, 47.22% precision | Yes (but worse than bi-encoder) |
| 012 | Cross-Encoder NLI DeBERTa Base | 100% recall, 45.72% precision | Yes (but worst precision) |
| 013 | Cross-Encoder NLI DeBERTa Large | 100% recall, 45.86% precision | Yes (even worse MAP) |
| 014 | Cross-Encoder NLI MiniLM | 100% recall, 45.72% precision | Yes (best NLI MAP: 0.67) |
| 015 | Cross-Encoder STS-B RoBERTa Large | 100% recall, 45.72% precision | Yes (MAP: 0.41) |
| 016 | Cross-Encoder STS-B DistilRoBERTa | 100% recall, 45.72% precision | Yes (worst MAP: 0.38) |
| 017 | Cross-Encoder Quora RoBERTa | 1.29% recall, 28.57% precision | **No** (complete failure) |
| 018 | BGE Reranker Base | 100% recall, 45.72% precision | Yes (best CE MAP: 0.74) |
| 019 | BGE Reranker Large | 100% recall, 45.72% precision | Yes (MAP: 0.71) |
