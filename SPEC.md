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
├── cpra-golden-emails/          # Test data generator (existing)
├── SPEC.md                      # This file - project context
│
├── docs/
│   └── experiments/
│       ├── LOG.md               # Experiment overview and learnings
│       └── NNN-experiment.md    # Detailed individual reports
│
├── src/
│   ├── models/                  # Model abstractions
│   │   ├── embeddings.py        # Unified embedding interface
│   │   └── llm.py               # Unified LLM interface
│   │
│   ├── pipeline/                # Search pipeline components
│   │   ├── query_expansion.py   # LLM-based candidate generation
│   │   ├── retrieval.py         # Embedding similarity search
│   │   ├── reranking.py         # Cross-encoder / LLM reranking
│   │   └── scoring.py           # Relevance score computation
│   │
│   ├── evaluation/              # Evaluation framework
│   │   ├── metrics.py           # Precision, recall, F1, MAP, etc.
│   │   ├── baseline.py          # Keyword search implementation
│   │   └── analysis.py          # Results breakdown and comparison
│   │
│   └── data/
│       └── loader.py            # Corpus and ground truth loading
│
├── experiments/                 # Executable experiment scripts
│   └── NNN_experiment_name.py
│
├── configs/
│   ├── models.yaml              # Model registry
│   └── experiments/             # Per-experiment configurations
│       └── NNN_experiment.yaml
│
├── results/                     # Experiment outputs (gitignored)
│   └── NNN_experiment_name/
│       ├── metrics.json
│       ├── predictions.csv
│       └── by_challenge_type.json
│
└── scripts/                     # Utility scripts
    ├── run_experiment.py
    └── compare_results.py
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

### H1: Embeddings Beat Keywords

Basic embedding retrieval (embed query, find similar docs) outperforms keyword search, especially on:

- Ambiguous terms (polysemy)
- Indirect references
- Semantic similarity without keyword overlap

### H2: Query Expansion Improves Recall

LLM-generated positive candidates ("what would a relevant document look like?") improve recall by covering vocabulary the original request didn't use.

### H3: Negative Candidates Improve Precision

LLM-generated negative candidates ("what are the red herrings?") help filter out false positives from ambiguous terms.

### H4: Cross-Encoder Reranking Improves Precision@K

A second-pass reranker that looks at query-document pairs together improves the ranking of top results.

### H5: Local Models Are Sufficient

The approach works with Ollama-hosted models (mistral3, embeddinggemma) without significant degradation from cloud models.

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

The `cpra-golden-emails/` directory contains a synthetic email corpus generator:

- **2,500 emails** across a fictional school district
- **5 CPRA request scenarios** including the "lead testing" case
- **~15% responsive** emails with known ground truth
- **~30% challenge cases**: ambiguous terms, near-miss, indirect references
- **Complete labels**: Per-email, per-request responsiveness with confidence and reasoning

This provides a controlled environment to test retrieval approaches before deploying on real data.

## Current Status

> *This section should be updated as the project progresses.*

### Completed

- [x] Project structure defined
- [x] Test data generator available (cpra-golden-emails)
- [x] SPEC.md created

### In Progress

- [ ] Implement baseline keyword search (Experiment 001)
- [ ] Set up evaluation framework
- [ ] Implement model abstraction layer

### Next Priorities

1. Generate test corpus
2. Implement and run keyword baseline
3. Implement basic embedding retrieval
4. Compare results, identify where embeddings help/hurt

### Key Learnings

> *To be populated as experiments complete.*

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
