#!/bin/bash
# Run all experiments on the v2 corpus and generate comparison results
#
# Usage:
#   ./scripts/run_all_experiments.sh [corpus_path] [threshold]
#
# Arguments:
#   corpus_path: Path to corpus directory (default: corpus/primary)
#   threshold: Score threshold for binary classification (default: 0.50)
#
# Examples:
#   ./scripts/run_all_experiments.sh                          # Primary corpus, 0.50 threshold
#   ./scripts/run_all_experiments.sh corpus/validation 0.60   # Validation corpus, 0.60 threshold

set -e

CORPUS_PATH="${1:-corpus/primary}"
THRESHOLD="${2:-0.50}"

echo "=============================================="
echo "CPRA Embedding Search Experiments"
echo "=============================================="
echo "Corpus: $CORPUS_PATH"
echo "Default threshold: $THRESHOLD"
echo "=============================================="
echo ""

# Create results directory if it doesn't exist
mkdir -p results

# List of experiment configs to run
EXPERIMENTS=(
    "configs/experiments/001_keyword_baseline.yaml"
    "configs/experiments/002_snowflake_arctic_l_v2.yaml"
    "configs/experiments/003_jina_v3.yaml"
    "configs/experiments/004_bge_m3.yaml"
    "configs/experiments/005_embeddinggemma.yaml"
    "configs/experiments/006_all_mpnet_base_v2.yaml"
    "configs/experiments/007_mxbai_embed_large.yaml"
    "configs/experiments/008_nomic_embed_text.yaml"
    "configs/experiments/009_bge_large_en_v1.5.yaml"
)

# Track which experiments succeeded
declare -a COMPLETED=()
declare -a FAILED=()

for config in "${EXPERIMENTS[@]}"; do
    if [ ! -f "$config" ]; then
        echo "SKIP: Config not found: $config"
        continue
    fi

    name=$(basename "$config" .yaml)
    echo ""
    echo "----------------------------------------------"
    echo "Running: $name"
    echo "----------------------------------------------"

    if python -m src.run_experiment \
        --config "$config" \
        --corpus "$CORPUS_PATH" \
        --threshold "$THRESHOLD"; then
        COMPLETED+=("$name")
        echo "SUCCESS: $name"
    else
        FAILED+=("$name")
        echo "FAILED: $name"
    fi
done

echo ""
echo "=============================================="
echo "EXPERIMENT RUN COMPLETE"
echo "=============================================="
echo ""
echo "Completed: ${#COMPLETED[@]}"
for exp in "${COMPLETED[@]}"; do
    echo "  - $exp"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED[@]}"
    for exp in "${FAILED[@]}"; do
        echo "  - $exp"
    done
fi

echo ""
echo "Results saved to: results/"
echo ""
