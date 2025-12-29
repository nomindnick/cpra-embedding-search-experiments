# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

This project evaluates whether embedding-based semantic search can outperform traditional keyword search for California Public Records Act (CPRA) document discovery. The core problem: keyword search for "lead" (contamination) returns thousands of false positives about "leadership" while missing documents that discuss lead issues without using that keyword.

**Goal:** Achieve ≥94% recall (legal requirement) while significantly improving precision over keyword search.

**Current Phase:** Running experiments on v2 corpus. Corpus generation complete.

## Key Documents

- **EXPERIMENT_LOG.md**: Tracks all experiments, results, and findings on v2 corpus
- **GENERATION_PLAN.md**: Corpus generation guide (complete). Contains:
  - Project background and lessons from v1 experiments
  - CPRA request definitions (primary: lead contamination, validation: PFAS)
  - Challenge type definitions with examples
  - Verification protocols
- **SPEC.md**: Original project specification and hypotheses
- **archive/**: Contains v1 corpus generator, experiment results, and findings for reference

## Corpus Structure

```
corpus/
├── primary/           # Lead contamination request (339 emails)
│   ├── request.json   # The CPRA request definition
│   ├── emails.json    # All emails with content and metadata
│   └── ground_truth.json
└── validation/        # PFAS request (59 emails)
    ├── request.json
    ├── emails.json
    └── ground_truth.json
```

**Primary corpus breakdown:**
- 155 responsive (46%): 30 DIRECT_MATCH, 30 AMBIGUOUS_TERMS, 35 INDIRECT_REFERENCE, 25 TECHNICAL_JARGON, 25 TEMPORAL_REFERENCE, 10 BURIED_IN_THREAD
- 184 non-responsive (54%): 55 KEYWORD_FALSE_POSITIVE, 45 ADJACENT_TOPIC, 55 TRUE_NEGATIVE

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run a single experiment
python -m src.run_experiment \
  --config configs/experiments/001_keyword_baseline.yaml \
  --corpus corpus/primary \
  --threshold 0.5

# Run all experiments
./scripts/run_all_experiments.sh

# Compare results across experiments
python scripts/compare_results.py

# Lint
ruff check src/

# Tests
pytest tests/
```

## Experiment Configs

| Config | Model | Notes |
|--------|-------|-------|
| 001_keyword_baseline.yaml | Keywords | Baseline (83.87% recall, 55.32% precision) |
| 002_snowflake_arctic_l_v2.yaml | Snowflake Arctic L v2.0 | Best v1 performer |
| 003_jina_v3.yaml | Jina v3 | Met 94% recall in v1 |
| 004_bge_m3.yaml | BGE-M3 | Met 94% recall in v1 |
| 005_embeddinggemma.yaml | embeddinggemma (Ollama) | Met 94% recall in v1 |
| 006_all_mpnet_base_v2.yaml | all-mpnet-base-v2 | Baseline embedding |
| 007_mxbai_embed_large.yaml | mxbai-embed-large (Ollama) | Best MAP in v1 |
| 008_nomic_embed_text.yaml | nomic-embed-text (Ollama) | Local-only option |
| 009_bge_large_en_v1.5.yaml | BGE Large EN v1.5 | Strong MTEB performer |

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
- TECHNICAL_JARGON: Uses regulatory/technical terminology (LSL, CCT, ppb)
- TEMPORAL_REFERENCE: Historical events or future planning
- BURIED_IN_THREAD: Relevant content surrounded by unrelated messages

**Non-responsive categories:**
- KEYWORD_FALSE_POSITIVE: "Lead" used for leadership/leading
- ADJACENT_TOPIC: Related domain (water/infrastructure) but not lead-specific
- TRUE_NEGATIVE: Clearly unrelated content (HR, IT, admin)

## Conventions

- Experiment configs: `configs/experiments/NNN_descriptive_name.yaml`
- Experiment results: `results/NNN_experiment_name/` (gitignored)
- Line length: 100 chars (ruff configured)
- Python 3.12+ with type hints

## V2 Baseline Results

Keyword search on v2 corpus:
- **Recall: 83.87%** (below 94% requirement - misses 25 documents)
- **Precision: 55.32%** (105 false positives from "lead" ambiguity)
- **Hardest categories**: BURIED_IN_THREAD (50%), TECHNICAL_JARGON (64%), INDIRECT_REFERENCE (77%)

## V1 Findings (for reference)

Previous experiments with API-generated corpus found:
- Best model: Snowflake Arctic Embed L v2.0 (95.20% recall, 86.02% precision)
- Embedding search significantly outperformed keyword search
- Larger models generally performed better with diminishing returns above ~500M params

See `archive/v1-experiment-log.md` for full v1 results.
