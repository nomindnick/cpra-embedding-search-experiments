# CPRA Embedding Search Experiments

Experiments comparing different approaches for identifying responsive documents in CPRA (California Public Records Act) requests.

## Project Goal

Evaluate whether embedding-based semantic search and LLM-assisted classification can outperform traditional keyword search for public records responsiveness detection.

## Project Structure

```
cpra-embedding-search-experiments/
├── cpra-golden-emails/          # Synthetic email corpus generator
│   ├── data/generated/          # Generated test corpora
│   └── README.md                # Generator documentation
├── src/                         # Experiment code
│   ├── data/                    # Data loading utilities
│   ├── models/                  # Embedding/LLM model wrappers
│   ├── pipeline/                # Search pipeline implementations
│   └── evaluation/              # Metrics and evaluation
├── configs/                     # Experiment configurations
│   ├── models.yaml              # Model definitions
│   └── experiments/             # Per-experiment configs
├── docs/experiments/            # Experiment documentation
│   └── LOG.md                   # Experiment results log
└── requirements.txt             # Python dependencies
```

## Test Data

### Golden Email Corpus

The test data is a synthetic corpus of school district emails with ground truth labels for CPRA responsiveness.

#### Corpus Versions

**v1 (Original):** `cpra-golden-emails/data/generated/corpus_20251207_153555/`
- 2,500 emails, 15% responsive
- Keyword baseline achieves 94% recall (unrealistically high)
- Most "challenge" emails still contain searchable keywords

**v2 (Harder - In Progress):** Requires `--use-llm` flag
- 5,000 emails, 20% responsive
- **Keyword-free emails:** 40% of responsive emails contain NO request keywords
- Variable difficulty by request (Lead Testing hardest at 60% keyword-free)
- Expected keyword baseline recall: **~60-70%** (forcing semantic search to prove value)

#### Corpus Contents
| File | Description |
|------|-------------|
| `emails/` | Individual email files (.txt) |
| `ground_truth.json` | Complete responsiveness mapping (email → CPRA requests) |
| `cpra_requests.json` | 5 CPRA request definitions with keywords and concepts |
| `email_corpus.xlsx` | Excel workbook with all data and responsiveness matrix |
| `statistics.json` | Corpus statistics including keyword analysis |
| `district_context.json` | Generated school district context |
| `generation_summary.json` | Generation parameters and results |

#### Challenge Types
| Challenge Type | Description |
|----------------|-------------|
| Near Miss | Related but not quite responsive |
| Indirect Reference | Euphemisms, pronouns, oblique mentions |
| Temporal Mismatch | Right topic, wrong time period |
| Ambiguous Terms | e.g., "lead" as metal vs. leadership |
| Partial Match | Partially matches request criteria |
| **Keyword Free** (v2) | Responsive but contains zero request keywords |
| **Euphemism** (v2) | Uses indirect language to avoid keywords |
| **Buried in Thread** (v2) | Responsive content in earlier reply, benign surface |

#### Keyword-Free Rate by Request (v2)
| Request | Keyword-Free % | Rationale |
|---------|---------------|-----------|
| Lead Testing | 60% | Hardest - ambiguous terms, euphemisms common |
| COVID Relief | 40% | Moderate - bureaucratic language |
| Special Education | 30% | Moderate - specialized terminology |
| EdTech Vendor | 20% | Easiest - concrete business terms |
| Safety Incidents | 50% | Hard - sensitive topics use euphemisms |

### Loading the Test Data

```python
import json
from pathlib import Path

CORPUS_PATH = Path("cpra-golden-emails/data/generated/corpus_20251207_153555")

# Load ground truth
with open(CORPUS_PATH / "ground_truth.json") as f:
    ground_truth = json.load(f)

# Load CPRA requests
with open(CORPUS_PATH / "cpra_requests.json") as f:
    cpra_requests = json.load(f)

# Access email data
emails = ground_truth["emails"]
responsiveness_map = ground_truth["responsiveness_map"]

# Check if email is responsive to a request
def is_responsive(email_id: str, request_id: str) -> bool:
    responses = responsiveness_map.get(email_id, [])
    return any(r["cpra_request_id"] == request_id and r["is_responsive"]
               for r in responses)
```

### Generating New Corpora

To generate a new test corpus:

```bash
cd cpra-golden-emails
source ../.venv/bin/activate

# v1 style (template-based, easier)
python generate_corpus.py --num-emails 2500 --responsive-rate 0.15

# v2 style (LLM-generated keyword-free emails, harder)
export ANTHROPIC_API_KEY=your_key_here
python generate_corpus.py --num-emails 5000 --responsive-rate 0.20 --use-llm
```

The v2 corpus uses Claude Haiku to generate emails that avoid keywords entirely, creating a more realistic test where semantic search must prove its value.

See `cpra-golden-emails/README.md` for full generation options.

## Experiments

Experiments are tracked in `docs/experiments/LOG.md`.

| # | Name | Status | Description |
|---|------|--------|-------------|
| 001 | Baseline Keyword | Planned | Establish keyword search baseline |

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run baseline experiment (coming soon)
python -m src.pipeline.run_experiment --config configs/experiments/001_baseline_keyword.yaml
```

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies
