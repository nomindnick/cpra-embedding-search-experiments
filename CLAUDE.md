# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

This project evaluates whether embedding-based semantic search can outperform traditional keyword search for California Public Records Act (CPRA) document discovery. The core problem: keyword search for "lead" (contamination) returns thousands of false positives about "leadership" while missing documents that discuss lead issues without using that keyword.

**Goal:** Achieve ≥94% recall (legal requirement) while significantly improving precision over keyword search.

**Current Phase:** Generating a new high-quality test corpus with manually-crafted emails.

## Key Documents

- **GENERATION_PLAN.md**: Primary guide for corpus generation. Contains:
  - Project background and lessons from v1 experiments
  - CPRA request definitions (primary: lead contamination, validation: PFAS)
  - Challenge type definitions with examples
  - Generation batch checklists (track progress here)
  - Verification protocols

- **SPEC.md**: Original project specification and hypotheses

- **archive/**: Contains v1 corpus generator, experiment results, and findings for reference

## Current Corpus Structure

```
corpus/
├── primary/           # Lead contamination request (~355 emails)
│   ├── request.json   # The CPRA request definition
│   ├── emails.json    # All emails with content and metadata
│   └── ground_truth.json
└── validation/        # PFAS request (~75 emails)
    ├── request.json
    ├── emails.json
    └── ground_truth.json
```

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run an experiment (after corpus is complete)
python -m src.run_experiment \
  --config configs/experiments/NNN_name.yaml \
  --corpus corpus/primary \
  --threshold 0.5

# Lint
ruff check src/

# Tests
pytest tests/
```

## Architecture

```
src/run_experiment.py (CLI entry point)
    ├── Load config (YAML) & corpus (JSON)
    ├── Create pipeline (KeywordSearchPipeline or EmbeddingSearchPipeline)
    │   └── EmbeddingSearchPipeline caches to .cache/embeddings/
    ├── Run Evaluator → computes precision/recall/F1/MAP
    └── Output to console (rich) and results/ directory
```

**Key modules:**
- `src/data/corpus.py`: Email, CPRARequest, Corpus data structures
- `src/models/embeddings.py`: Model abstractions (SentenceTransformer, Ollama, OpenAI)
- `src/pipeline/`: SearchPipeline base + KeywordSearchPipeline + EmbeddingSearchPipeline
- `src/evaluation/`: Evaluator, metrics computation, reporting
- `configs/models.yaml`: Model registry with all embedding models

## Challenge Types (Primary Corpus)

**Responsive categories:**
- DIRECT_MATCH: Explicit lead contamination discussion
- AMBIGUOUS_TERMS: Uses "lead" (metal) with disambiguating context
- INDIRECT_REFERENCE: Discusses topic without "lead" keyword
- TECHNICAL_JARGON: Uses regulatory/technical terminology
- TEMPORAL_REFERENCE: Historical events or future planning
- BURIED_IN_THREAD: Relevant content surrounded by unrelated messages

**Non-responsive categories:**
- KEYWORD_FALSE_POSITIVE: "Lead" used for leadership/leading
- ADJACENT_TOPIC: Related domain but not lead-specific
- TRUE_NEGATIVE: Clearly unrelated content

## Conventions

- Experiment configs: `configs/experiments/NNN_descriptive_name.yaml`
- Experiment results: `results/NNN_experiment_name/` (gitignored)
- Line length: 100 chars (ruff configured)
- Python 3.12+ with type hints

## V1 Findings (for reference)

Previous experiments with API-generated corpus found:
- Best model: Snowflake Arctic Embed L v2.0 (95.20% recall, 86.02% precision)
- Embedding search significantly outperformed keyword search
- Larger models generally performed better with diminishing returns above ~500M params

See `archive/v1-experiment-log.md` for full results.
