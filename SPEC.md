# CPRA Embedding Search Experiments

## Problem Statement

The California Public Records Act (CPRA) requires public agencies to provide access to public records upon request. For agencies like school districts, cities, and counties, responding to CPRA requests—particularly for email—is a significant operational burden.

The current standard practice is **keyword search**: attorneys or staff develop a list of keywords, run them against email archives, and manually review the results. This approach has two fundamental problems:

1. **Low Recall (Missing Documents)**: Relevant documents that don't contain the exact keywords are missed. A document discussing "contamination in the water supply" won't be found by a search for "lead testing."

2. **Low Precision (Too Much Noise)**: Keywords with multiple meanings surface irrelevant documents. Searching for "lead" to find documents about lead pipes also returns every email about "leading a team," "lead teacher," or "taking the lead on a project."

### The "Lead" Example

A real-world illustration: A mass CPRA request went to California school districts asking for records about environmental hazards at school sites. The keyword "lead" was suggested to find documents about lead pipes, lead testing, etc. The result:

- Thousands of irrelevant documents about leadership, leading committees, lead teachers
- Narrowing to "lead testing" improved precision but missed documents about "lead in pipes," "lead contamination," "elevated lead levels"

This is the problem we're trying to solve.

## Approach

We hypothesize that **embedding-based semantic search** can outperform keyword search, particularly on ambiguous and indirect cases.

### Core Idea

Instead of matching keywords, we:

1. **Embed documents** into a semantic vector space where meaning, not just words, determines proximity
2. **Expand the query** using an LLM to generate:
   - **Positive candidates**: Example documents/passages that SHOULD be relevant
   - **Negative candidates**: "Red herrings" that might match keywords but are NOT relevant
3. **Retrieve** documents close to positive candidates and far from negative candidates
4. **Rerank** top candidates using a cross-encoder or LLM for higher precision
5. **Iterate** using confirmed matches to refine subsequent searches

### Why This Should Work

- Embeddings capture semantic similarity: "lead contamination" and "elevated lead levels in water" are close in embedding space even without shared keywords
- Negative candidates help with polysemy: by explicitly modeling "lead as leadership," we can push those documents away
- LLM query expansion bridges the vocabulary gap between the request and the documents

## Constraints

### Final Solution Requirements

- **CPU-only**: Must run on standard agency hardware without GPU
- **Local/Offline**: No cloud dependencies for production use (data sensitivity)
- **Simple deployment**: Ollama as the target runtime for LLMs and embeddings
- **Open source**: Intended for public release to agencies

### Experimentation Phase

- Cloud models (OpenAI, Anthropic) are acceptable for rapid iteration
- Focus on finding the right approach before optimizing for local execution
- Document which approaches require cloud vs. work locally

## Architecture

```
cpra-embedding-search-experiments/
├── corpus/                      # Test data (v2 manually-crafted corpus)
│   ├── primary/                 # Lead contamination request (339 emails)
│   └── validation/              # PFAS request (59 emails)
│
├── archive/                     # V1 corpus and old experiments
│   ├── cpra-golden-emails/      # Original corpus generator
│   └── v1-experiment-log.md     # V1 results for reference
│
├── src/
│   ├── data/
│   │   └── corpus.py            # Email, CPRARequest, Corpus data structures
│   │
│   ├── models/
│   │   └── embeddings.py        # Model abstractions (SentenceTransformer, Ollama, OpenAI)
│   │
│   ├── pipeline/
│   │   ├── base.py              # SearchPipeline base class
│   │   ├── keyword.py           # KeywordSearchPipeline
│   │   ├── embedding.py         # EmbeddingSearchPipeline
│   │   └── cross_encoder.py     # CrossEncoderSearchPipeline
│   │
│   ├── evaluation/
│   │   ├── evaluator.py         # Evaluator class
│   │   ├── metrics.py           # Precision, recall, F1, MAP computation
│   │   └── reporter.py          # Rich console output
│   │
│   └── run_experiment.py        # CLI entry point
│
├── configs/
│   ├── models.yaml              # Model registry
│   └── experiments/             # Per-experiment configurations (001-019)
│
├── results/                     # Experiment outputs (gitignored)
│
├── SPEC.md                      # This file - project specification
├── EXPERIMENT_LOG.md            # All experiment results and analysis
├── RESEARCH_AGENDA.md           # Future research directions
├── GENERATION_PLAN.md           # Corpus design documentation
└── CLAUDE.md                    # AI assistant context
```

### Component Responsibilities

**Model Layer** (`src/models/`)

- Abstracts away model provider differences
- Consistent interface whether using Ollama, OpenAI, or sentence-transformers
- Handles batching, retries, rate limiting

**Pipeline Layer** (`src/pipeline/`)

- Each component is independent and composable
- Experiments configure which components to use and their parameters
- Easy to add new retrieval or reranking strategies

**Evaluation Layer** (`src/evaluation/`)

- Loads ground truth, runs predictions through metrics
- Breaks down results by CPRA request, challenge type, confidence threshold
- Generates comparison reports across experiments

## Experiment Framework

### Experiment Numbering

- Sequential: 001, 002, 003, ...
- Use suffixes for variations: 005a, 005b
- Each experiment has:
  - Config file: `configs/experiments/NNN_name.yaml`
  - Script: `experiments/NNN_name.py`
  - Results: `results/NNN_name/`
  - Log entry: Added to `docs/experiments/LOG.md`
  - Optional detailed report: `docs/experiments/NNN-name.md`

### Running Experiments

```bash
# Run a specific experiment
python scripts/run_experiment.py 001

# Compare results across experiments
python scripts/compare_results.py 001 002 003
```

### Configuration Format

```yaml
# configs/experiments/NNN_example.yaml
name: "Descriptive experiment name"
description: "What hypothesis this tests"

embedding_model: "ollama:nomic-embed-text"
llm_model: "ollama:llama3.2"

pipeline:
  query_expansion:
    enabled: true
    strategy: "positive_and_negative"
    num_positive: 5
    num_negative: 3

  retrieval:
    method: "cosine_similarity"
    top_k: 100

  reranking:
    enabled: false

evaluation:
  threshold: 0.5
  metrics: ["precision", "recall", "f1", "map"]
```

## Key Hypotheses

These are the core questions we're trying to answer:

### H1: Embeddings Beat Keywords ✅ CONFIRMED

Basic embedding retrieval outperforms keyword search, especially on:

- Ambiguous terms (polysemy)
- Indirect references
- Semantic similarity without keyword overlap

**Result:** Confirmed decisively on v2 corpus. all-mpnet-base-v2 achieves 98.71% recall vs 83.87% for keywords (+14.84%). Embeddings excel on TECHNICAL_JARGON (96% vs 64%) and BURIED_IN_THREAD (90% vs 50%).

### H2: Query Expansion Improves Recall ❓ NOT TESTED

LLM-generated positive candidates ("what would a relevant document look like?") improve recall by covering vocabulary the original request didn't use.

**Status:** Not yet tested. Planned for future experiments.

### H3: Negative Candidates Improve Precision ❓ NOT TESTED

LLM-generated negative candidates ("what are the red herrings?") help filter out false positives from ambiguous terms.

**Status:** Not yet tested. Planned for future experiments.

### H4: Cross-Encoder Reranking Improves Precision@K ❌ REFUTED

A second-pass reranker that looks at query-document pairs together improves the ranking of top results.

**Result:** Refuted on keyword-free corpora. All 9 cross-encoder experiments showed worse performance than bi-encoders:
- MS-MARCO cross-encoders: Catastrophic lexical bias (14% recall on INDIRECT_REFERENCE)
- NLI cross-encoders: No discrimination (score everything as relevant)
- Best cross-encoder MAP (0.74) << best bi-encoder MAP (0.89)

Cross-encoders may help on corpora with keyword overlap, but fail on semantic-only matching.

### H5: Local Models Are Sufficient ✅ CONFIRMED

The approach works with Ollama-hosted models without significant degradation from cloud models.

**Result:** Confirmed. Local models meeting 94% recall:
- embeddinggemma (Ollama): 100% recall, 49.36% precision
- mxbai-embed-large (Ollama): 98.71% recall, 51.17% precision
- nomic-embed-text (Ollama): 99.35% recall, 46.11% precision

No cloud-only model significantly outperforms these local options.

## Metrics

### Primary Metrics

- **Precision@K**: Of the top K results, what fraction are relevant?
- **Recall@K**: Of all relevant documents, what fraction appear in top K?
- **F1@K**: Harmonic mean of precision and recall
- **MAP (Mean Average Precision)**: Average precision across all recall levels

### Breakdown Dimensions

- **By CPRA Request**: Some requests may be inherently harder
- **By Challenge Type**: Ambiguous, near-miss, indirect reference, partial match
- **By Confidence Threshold**: Precision-recall tradeoff curves

### Baseline Comparison

All experiments report improvement/regression vs. Experiment 001 (keyword baseline).

## Test Data

The `corpus/` directory contains manually-crafted email corpora designed to test semantic search capabilities:

### Corpus v2 (Current)

Located in `corpus/primary/` and `corpus/validation/`:

**Primary Corpus (Lead Contamination):** 339 emails
- 155 responsive (46%)
- 184 non-responsive (54%)
- Single CPRA request about lead contamination in water supply

**Validation Corpus (PFAS):** 59 emails
- 25 responsive (42%)
- 34 non-responsive (58%)
- Single CPRA request about PFAS contamination

**Corpus Files:**
```
corpus/
├── primary/
│   ├── request.json      # CPRA request definition
│   ├── emails.json       # All 339 emails with content
│   └── ground_truth.json # Responsiveness labels with challenge types
└── validation/
    ├── request.json
    ├── emails.json
    └── ground_truth.json
```

### Challenge Types

**Responsive categories:**
| Type | Count | Description |
|------|-------|-------------|
| DIRECT_MATCH | 30 | Explicit lead contamination discussion |
| AMBIGUOUS_TERMS | 30 | Uses "lead" (metal) with disambiguating context |
| INDIRECT_REFERENCE | 35 | Discusses topic without "lead" keyword |
| TECHNICAL_JARGON | 25 | Uses regulatory terms (LSL, CCT, ppb) |
| TEMPORAL_REFERENCE | 25 | Historical events or future planning |
| BURIED_IN_THREAD | 10 | Relevant content in thread context |

**Non-responsive categories:**
| Type | Count | Description |
|------|-------|-------------|
| KEYWORD_FALSE_POSITIVE | 55 | "Lead" used for leadership/leading |
| ADJACENT_TOPIC | 45 | Related domain but not lead-specific |
| TRUE_NEGATIVE | 55 | Clearly unrelated content |

### Keyword Baseline Performance

The v2 corpus successfully exposes keyword search limitations:
- **Recall: 83.87%** (misses 25 documents — below 94% legal requirement)
- **Precision: 55.32%** (105 false positives from "lead" ambiguity)
- **Hardest categories**: BURIED_IN_THREAD (50%), TECHNICAL_JARGON (64%), INDIRECT_REFERENCE (77%)

### V1 Corpus (Archived)

The original v1 corpus (2,500 LLM-generated emails, 5 CPRA requests) is archived in `archive/`. It proved too easy — keyword baseline achieved 94% recall, leaving no room for embedding models to demonstrate value.

## Current Status

> *Last updated: 2025-12-29*

### Completed

- [x] Project structure defined
- [x] V2 corpus manually crafted (339 primary + 59 validation emails)
- [x] Evaluation framework with per-challenge-type breakdowns
- [x] **19 experiments completed** on v2 corpus:
  - 001: Keyword baseline (83.87% recall, 55.32% precision)
  - 002-010: Bi-encoder embedding models (8 of 10 meet 94% recall)
  - 011-019: Cross-encoder models (9 experiments, all underperform bi-encoders)

### Best Results

| Model | Recall | Precision | F1 | MAP | Meets 94%? |
|-------|--------|-----------|-----|-----|------------|
| **all-mpnet-base-v2** | 98.71% | 57.74% | 72.86% | 0.8923 | **Yes (Best)** |
| Jina v3 | 98.06% | 51.70% | 67.71% | 0.8592 | Yes |
| nomic-embed-text | 99.35% | 46.11% | 62.99% | 0.8158 | Yes |
| Qwen3 0.6B | 89.03% | 77.53% | 82.88% | 0.9169 | No (best F1) |

### Next Priorities

1. Two-stage pipelines: high-recall bi-encoder + precision-focused reranker
2. Query expansion with LLM-generated paraphrases
3. Ensemble methods combining multiple models
4. Test on validation corpus (PFAS) to verify generalization

### Key Learnings

1. **Bi-encoders beat cross-encoders**: On keyword-free corpora, general-purpose bi-encoders (all-mpnet-base-v2) outperform specialized cross-encoders
2. **Cross-encoder training matters**: MS-MARCO cross-encoders fail catastrophically (lexical bias); NLI/STS models saturate with no discrimination
3. **Smaller is better for cross-encoders**: Inverse scaling observed — smaller models rank better
4. **7 of 10 embedding models meet 94% recall**: Most bi-encoders can achieve legal compliance threshold
5. **Precision-recall tradeoff is unavoidable**: Best precision at 94%+ recall is 57.74% (all-mpnet); models with higher precision (Qwen3: 77.53%) can't reach 94% recall
6. **Challenge type analysis reveals model strengths**: Keywords struggle with BURIED_IN_THREAD (50%) and TECHNICAL_JARGON (64%); embeddings excel on these

## Conventions

### Code Style

- Python 3.12+
- Type hints throughout
- Async where beneficial for batching
- Docstrings for public functions

### File Naming

- Experiments: `NNN_descriptive_name.py` (snake_case)
- Configs: `NNN_descriptive_name.yaml`
- Reports: `NNN-descriptive-name.md` (kebab-case for markdown)

### Git

- Results directory is gitignored (large files)
- Configs and experiment scripts are committed
- Generated test data may be gitignored depending on size

### Documentation

- Update LOG.md after each experiment
- Update "Current Status" in SPEC.md when priorities shift
- Detailed reports only for significant experiments
