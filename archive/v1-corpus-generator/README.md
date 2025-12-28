# CPRA Golden Email Corpus Generator

A Python-based system for generating synthetic email corpora with ground truth labels for testing CPRA (California Public Records Act) document responsiveness detection algorithms.

## Overview

This tool generates a realistic corpus of school district emails with known ground truth about which emails are responsive to specific CPRA requests. It's designed to demonstrate the limitations of keyword-based search approaches and provide a testbed for evaluating more sophisticated NLP techniques like embeddings, query expansion, and LLM-based classification.

### Key Features

- **Realistic School District Context**: Generates a complete school district with staff, departments, and schools
- **Multiple CPRA Requests**: 5 diverse requests covering different topics and complexity levels
- **Challenge Patterns**: Intentionally difficult emails including:
  - Ambiguous terms (e.g., "lead" as metal vs. leadership)
  - Near-misses (related but not quite responsive)
  - Indirect references (euphemisms, pronouns)
- **Ground Truth Tracking**: Complete tracking of which emails are responsive to which requests
- **Multiple Export Formats**: JSON, Excel, and individual email files
- **Expandable Design**: Easy to add more emails, requests, or challenge patterns

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cpra-golden-emails
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up environment variables for LLM integration:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

### Basic Generation

Generate a default corpus of 2500 emails:
```bash
python generate_corpus.py
```

### Custom Parameters

```bash
# Generate 5000 emails with 20% responsive rate
python generate_corpus.py --num-emails 5000 --responsive-rate 0.20

# Use LLM for more realistic email generation
python generate_corpus.py --use-llm --llm-provider openai

# Set custom output directory
python generate_corpus.py --output-dir my_corpus

# Use a specific random seed for reproducibility
python generate_corpus.py --seed 123
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num-emails` | Number of emails to generate | 2500 |
| `--responsive-rate` | Percentage of responsive emails (0.0-1.0) | 0.15 |
| `--challenge-rate` | Percentage of responsive emails with challenges | 0.30 |
| `--use-llm` | Use LLM for email generation | False |
| `--llm-provider` | LLM provider (openai/anthropic) | openai |
| `--output-dir` | Output directory for generated files | data/generated |
| `--seed` | Random seed for reproducibility | 42 |
| `--config` | Path to custom configuration file | None |

### Using Configuration Files

Create a custom configuration file based on `config/generation_config.yaml`:

```yaml
generation:
  total_emails: 5000
  responsive_rate: 0.20
  challenge_email_rate: 0.40

email:
  min_length: 100
  max_length: 800
  attachment_probability: 0.15

llm:
  use_llm: true
  provider: "openai"
  model: "gpt-4"
```

Then run:
```bash
python generate_corpus.py --config my_config.yaml
```

## Output Structure

The generator creates a timestamped directory with the following structure:

```
data/generated/corpus_20241031_143022/
├── emails/                    # Individual email files
│   ├── a1b2c3d4.txt
│   ├── e5f6g7h8.txt
│   └── ...
├── ground_truth.json          # Complete ground truth mapping
├── email_corpus.xlsx          # Excel workbook with multiple sheets
├── cpra_requests.json         # CPRA request definitions
├── district_context.json      # School district information
├── statistics.json            # Corpus statistics
└── generation_summary.json    # Generation parameters and results
```

### File Descriptions

- **emails/**: Individual text files for each email in a format similar to .eml
- **ground_truth.json**: Complete mapping of which emails are responsive to which requests
- **email_corpus.xlsx**: Multi-sheet Excel file with:
  - Emails sheet: All emails with metadata
  - Responsiveness Matrix: Email × Request matrix
  - CPRA Requests: Request details
  - Statistics: Corpus statistics
  - Staff Directory: Generated staff members
- **cpra_requests.json**: Full CPRA request definitions with keywords and concepts
- **statistics.json**: Detailed statistics about the corpus

## CPRA Requests

The system generates 5 CPRA requests by default:

1. **Lead Testing in Water Systems** - Tests ambiguous term challenges
2. **COVID Relief Fund Allocation** - Tests indirect references
3. **Special Education Program Changes** - Tests near-misses and partial matches
4. **EdTech Vendor Contracts** - Tests temporal mismatches
5. **Student Safety Incidents** - Tests indirect references and near-misses

## Testing Your Detection Algorithms

### Loading the Corpus

```python
import json
from pathlib import Path

# Load ground truth
with open('data/generated/corpus_[timestamp]/ground_truth.json', 'r') as f:
    ground_truth = json.load(f)

# Load CPRA requests
with open('data/generated/corpus_[timestamp]/cpra_requests.json', 'r') as f:
    requests = json.load(f)

# Access emails
emails = ground_truth['emails']
responsiveness_map = ground_truth['responsiveness_map']

# Check if an email is responsive to a request
def is_responsive(email_id, request_id):
    responses = responsiveness_map.get(email_id, [])
    for resp in responses:
        if resp['cpra_request_id'] == request_id:
            return resp['is_responsive']
    return False
```

### Evaluating Your Approach

```python
# Example: Evaluate keyword search
def keyword_search(emails, keywords):
    results = []
    for email in emails:
        text = f"{email['subject']} {email['body']}".lower()
        if any(kw.lower() in text for kw in keywords):
            results.append(email['id'])
    return results

# Get results for a specific request
request = requests[0]  # First CPRA request
predicted = keyword_search(emails, request['primary_keywords'])

# Calculate metrics
true_positives = []
false_positives = []
false_negatives = []

for email in emails:
    email_id = email['id']
    is_predicted = email_id in predicted
    is_actual = is_responsive(email_id, request['id'])

    if is_predicted and is_actual:
        true_positives.append(email_id)
    elif is_predicted and not is_actual:
        false_positives.append(email_id)
    elif not is_predicted and is_actual:
        false_negatives.append(email_id)

precision = len(true_positives) / len(predicted) if predicted else 0
recall = len(true_positives) / (len(true_positives) + len(false_negatives))
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")
```

## Extending the System

### Adding New CPRA Requests

Edit `src/generators/cpra_requests.py` and add new requests to the `generate_requests` method:

```python
requests.append(CPRARequest(
    id="cpra_006",
    title="Your New Request",
    request_text="Full text of the CPRA request...",
    primary_keywords=["keyword1", "keyword2"],
    # ... other parameters
))
```

### Adding New Challenge Patterns

Modify `src/generators/email_generator.py` to add new challenge generation methods:

```python
def _generate_your_challenge_content(self, request, sender):
    # Your challenge generation logic
    subject = "..."
    body = "..."
    return subject, body
```

### Customizing the School District

Edit `src/generators/school_district.py` to modify:
- Number and types of schools
- Staff roles and departments
- District structure

## Performance Considerations

- **Without LLM**: Generation is fast (~1-2 minutes for 2500 emails)
- **With LLM**: Slower but more realistic (~10-20 minutes for 2500 emails depending on API)
- **Memory**: ~500MB for 2500 emails in memory
- **Disk Space**: ~50MB for exported 2500 email corpus

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure you're in the project root directory
2. **API Key Errors**: Check your .env file has valid API keys
3. **Memory Issues**: Reduce `--num-emails` or generate in batches
4. **LLM Rate Limits**: Add delays or use template-based generation

### Debug Mode

For verbose output during generation:
```python
# In generate_corpus.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Potential improvements for the system:

- [ ] Multi-language email generation
- [ ] Email threading and conversation chains
- [ ] More sophisticated attachment simulation
- [ ] Integration with real email formats (.eml, .msg)
- [ ] Web interface for interactive testing
- [ ] Benchmark suite for comparing algorithms
- [ ] Support for other document types (memos, reports)

## License

This project is designed for educational and research purposes to improve public records request processing.

## Contributing

Contributions are welcome! Areas where help would be appreciated:
- Additional challenge patterns
- More diverse CPRA request templates
- Improved LLM prompts for email generation
- Evaluation metrics and visualization tools

## Contact

For questions or suggestions about using this tool for CPRA responsiveness testing, please open an issue on the repository.