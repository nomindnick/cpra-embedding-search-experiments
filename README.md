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

The test data is a synthetic corpus of 2,500 school district emails with ground truth labels for CPRA responsiveness.

**Location:** `cpra-golden-emails/data/generated/corpus_20251207_153555/`

**Corpus Contents:**
| File | Description |
|------|-------------|
| `emails/` | 2,500 individual email files (.txt) |
| `ground_truth.json` | Complete responsiveness mapping (email → CPRA requests) |
| `cpra_requests.json` | 5 CPRA request definitions with keywords and concepts |
| `email_corpus.xlsx` | Excel workbook with all data and responsiveness matrix |
| `statistics.json` | Corpus statistics and distribution info |
| `district_context.json` | Generated school district context |
| `generation_summary.json` | Generation parameters and results |

**Corpus Statistics:**
- **Total emails:** 2,500
- **Responsive emails:** 375 (15%)
- **CPRA requests:** 5 (75 responsive emails each)
- **Challenge emails:** 110 (difficult edge cases)
- **Emails with attachments:** 37

**Challenge Types (for testing edge cases):**
| Challenge Type | Count | Description |
|----------------|-------|-------------|
| Near Miss | 50 | Related but not quite responsive |
| Indirect Reference | 35 | Euphemisms, pronouns, oblique mentions |
| Temporal Mismatch | 24 | Right topic, wrong time period |
| Ambiguous Terms | 18 | e.g., "lead" as metal vs. leadership |
| Partial Match | 12 | Partially matches request criteria |

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

To generate a new test corpus with different parameters:

```bash
cd cpra-golden-emails
python generate_corpus.py --num-emails 5000 --responsive-rate 0.20 --seed 123
```

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
