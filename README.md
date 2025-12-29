# CPRA Embedding Search Experiments

Experiments comparing embedding-based semantic search against keyword search for California Public Records Act (CPRA) document discovery.

## Project Goal

Achieve **≥94% recall** (legal requirement) while improving precision over keyword search for identifying responsive documents in CPRA requests.

**Key Finding:** Embedding models can meet the 94% recall requirement while reducing false positives by ~50% compared to keyword search.

## Project Structure

```
cpra-embedding-search-experiments/
├── corpus/                      # Test data (v2 manually-crafted corpus)
│   ├── primary/                 # Lead contamination request (339 emails)
│   └── validation/              # PFAS request (59 emails)
├── archive/                     # V1 corpus and experiments (reference)
├── src/                         # Experiment code
│   ├── data/                    # Corpus data structures
│   ├── models/                  # Embedding model wrappers
│   ├── pipeline/                # Search pipeline implementations
│   └── evaluation/              # Metrics and evaluation
├── configs/                     # Experiment configurations
│   ├── models.yaml              # Model definitions
│   └── experiments/             # Per-experiment configs (001-019)
├── results/                     # Experiment outputs (gitignored)
├── EXPERIMENT_LOG.md            # Detailed experiment results
├── RESEARCH_AGENDA.md           # Future research directions
└── SPEC.md                      # Project specification
```

## Test Data

### V2 Corpus (Current)

Manually-crafted corpus designed to test semantic search on keyword-free content:

| Corpus | Emails | Responsive | Non-Responsive | CPRA Request |
|--------|--------|------------|----------------|--------------|
| Primary | 339 | 155 (46%) | 184 (54%) | Lead contamination |
| Validation | 59 | 25 (42%) | 34 (58%) | PFAS contamination |

**Corpus Files:**
```
corpus/
├── primary/
│   ├── request.json      # CPRA request definition
│   ├── emails.json       # All emails with content
│   └── ground_truth.json # Labels with challenge types
└── validation/
    └── (same structure)
```

### Challenge Types

**Responsive categories:**
| Type | Count | Description |
|------|-------|-------------|
| DIRECT_MATCH | 30 | Explicit lead contamination discussion |
| AMBIGUOUS_TERMS | 30 | "Lead" (metal) with disambiguating context |
| INDIRECT_REFERENCE | 35 | Topic discussed without "lead" keyword |
| TECHNICAL_JARGON | 25 | Regulatory terms (LSL, CCT, ppb) |
| TEMPORAL_REFERENCE | 25 | Historical events or future planning |
| BURIED_IN_THREAD | 10 | Relevant content in thread context |

**Non-responsive categories:**
| Type | Count | Description |
|------|-------|-------------|
| KEYWORD_FALSE_POSITIVE | 55 | "Lead" as leadership/leading |
| ADJACENT_TOPIC | 45 | Related domain, not lead-specific |
| TRUE_NEGATIVE | 55 | Clearly unrelated content |

### Loading the Corpus

```python
from src.data.corpus import Corpus

corpus = Corpus.load("corpus/primary")
print(f"Emails: {len(corpus.emails)}")
print(f"Request: {corpus.request.title}")

# Check responsiveness
for email in corpus.emails:
    is_responsive = corpus.ground_truth.get(email.id, {}).get("responsive", False)
    challenge_type = corpus.ground_truth.get(email.id, {}).get("challenge_type")
```

### V1 Corpus (Archived)

The original LLM-generated v1 corpus (2,500 emails, 5 CPRA requests) is in `archive/`. It proved too easy — keyword baseline achieved 94% recall.

## Experiments

19 experiments completed. Full details in `EXPERIMENT_LOG.md`.

### Best Results (Meeting 94% Recall Requirement)

| # | Model | Recall | Precision | F1 | MAP |
|---|-------|--------|-----------|-----|-----|
| **006** | **all-mpnet-base-v2** | **98.71%** | **57.74%** | **72.86%** | **0.8923** |
| 003 | Jina v3 | 98.06% | 51.70% | 67.71% | 0.8592 |
| 008 | nomic-embed-text (Ollama) | 99.35% | 46.11% | 62.99% | 0.8158 |
| 007 | mxbai-embed-large (Ollama) | 98.71% | 51.17% | 67.40% | 0.8561 |
| 009 | BGE Large EN v1.5 | 99.35% | 47.24% | 64.03% | 0.8731 |
| 004 | BGE-M3 | 100.00% | 46.83% | 63.79% | 0.8607 |
| 005 | embeddinggemma (Ollama) | 100.00% | 49.36% | 66.10% | 0.8757 |

### Baselines

| # | Model | Recall | Precision | F1 | Notes |
|---|-------|--------|-----------|-----|-------|
| 001 | Keyword Search | 83.87% | 55.32% | 66.67% | Below 94% requirement |
| 002 | Snowflake Arctic L v2.0 | 81.29% | 70.39% | 75.45% | Best v1 performer |
| 010 | Qwen3 0.6B | 89.03% | 77.53% | 82.88% | Best F1, but misses recall |

### Cross-Encoder Experiments (011-019)

All cross-encoders underperformed bi-encoders on keyword-free corpus:
- MS-MARCO: Lexical bias (14% recall on INDIRECT_REFERENCE)
- NLI: No discrimination (scores everything as relevant)
- Best cross-encoder MAP (0.74) << best bi-encoder MAP (0.89)

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run an experiment
python -m src.run_experiment \
  --config configs/experiments/006_all_mpnet_base_v2.yaml \
  --corpus corpus/primary \
  --threshold 0.30

# Run keyword baseline
python -m src.run_experiment \
  --config configs/experiments/001_keyword_baseline.yaml \
  --corpus corpus/primary

# Compare results
python scripts/compare_results.py
```

## Requirements

- Python 3.12+
- See `requirements.txt` for dependencies
- Ollama (optional, for local models)
