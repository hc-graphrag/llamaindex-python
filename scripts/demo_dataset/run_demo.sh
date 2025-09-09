#!/bin/bash

# GraphRAG DRIFT Search Demo Script
# This script demonstrates the full workflow of DRIFT search with sample data

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEMO_DATA_DIR="$SCRIPT_DIR/documents"
OUTPUT_DIR="$PROJECT_ROOT/demo_output"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GraphRAG DRIFT Search Demo${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print step
print_step() {
    echo -e "${GREEN}[Step $1]${NC} $2"
}

# Function to print info
print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Function to print error
print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_info "Activating virtual environment..."
    if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    else
        print_error "Virtual environment not found. Please run 'pdm install' first."
        exit 1
    fi
fi

# Clean up previous demo output
if [ -d "$OUTPUT_DIR" ]; then
    print_info "Cleaning up previous demo output..."
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy demo documents to data directory
print_step 1 "Preparing demo documents..."
DATA_DIR="$PROJECT_ROOT/data"
mkdir -p "$DATA_DIR"
cp -r "$DEMO_DATA_DIR"/* "$DATA_DIR/"
echo "  ✓ Copied $(ls -1 "$DEMO_DATA_DIR" | wc -l) documents to data directory"

# Set environment variables
export GRAPHRAG_OUTPUT_DIR="$OUTPUT_DIR"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    print_error "ANTHROPIC_API_KEY is not set. Please set it before running the demo."
    echo "    export ANTHROPIC_API_KEY='your-api-key'"
    exit 1
fi

# Run document ingestion
print_step 2 "Ingesting documents and building knowledge graph..."
echo ""
cd "$PROJECT_ROOT"
python -m graphrag_anthropic_llamaindex.main add "$DATA_DIR" --output-dir "$OUTPUT_DIR"

# Wait for indexing to complete
sleep 2

# Display graph statistics
print_step 3 "Knowledge graph statistics:"
echo ""
if [ -f "$OUTPUT_DIR/entities.parquet" ]; then
    ENTITY_COUNT=$(python -c "import pandas as pd; print(len(pd.read_parquet('$OUTPUT_DIR/entities.parquet')))")
    echo "  • Entities extracted: $ENTITY_COUNT"
fi
if [ -f "$OUTPUT_DIR/relationships.parquet" ]; then
    REL_COUNT=$(python -c "import pandas as pd; print(len(pd.read_parquet('$OUTPUT_DIR/relationships.parquet')))")
    echo "  • Relationships found: $REL_COUNT"
fi
if [ -f "$OUTPUT_DIR/community_summaries.parquet" ]; then
    COMM_COUNT=$(python -c "import pandas as pd; print(len(pd.read_parquet('$OUTPUT_DIR/community_summaries.parquet')))")
    echo "  • Communities detected: $COMM_COUNT"
fi
echo ""

# Demo queries for different search modes
print_step 4 "Running demo searches..."
echo ""

# Define demo queries
declare -a queries=(
    "人工知能の最新技術について教えてください"
    "気候変動への対策は？"
    "量子コンピュータの応用分野"
    "CRISPRとゲノム編集"
    "再生可能エネルギーの種類"
)

declare -a modes=("local" "global" "drift")

# Function to run search
run_search() {
    local query="$1"
    local mode="$2"
    
    echo -e "${BLUE}Mode: ${mode^^}${NC}"
    echo "Query: \"$query\""
    echo "---"
    
    python -m graphrag_anthropic_llamaindex.main search "$query" \
        --mode "$mode" \
        --output-dir "$OUTPUT_DIR" 2>/dev/null | head -20
    
    echo ""
    echo "=========================================="
    echo ""
}

# Run DRIFT search for all queries
print_info "Testing DRIFT search (combines local and global search)..."
echo ""

for query in "${queries[@]:0:3}"; do
    run_search "$query" "drift"
done

# Comparison of different modes
print_step 5 "Comparing search modes for a sample query..."
echo ""

SAMPLE_QUERY="機械学習とディープラーニングの違いは？"
echo "Query: \"$SAMPLE_QUERY\""
echo ""

for mode in "${modes[@]}"; do
    echo -e "${BLUE}━━━ ${mode^^} Search ━━━${NC}"
    python -m graphrag_anthropic_llamaindex.main search "$SAMPLE_QUERY" \
        --mode "$mode" \
        --output-dir "$OUTPUT_DIR" 2>/dev/null | head -15
    echo ""
done

# Interactive mode
print_step 6 "Interactive DRIFT search"
echo ""
print_info "You can now try your own queries. Type 'exit' to quit."
echo ""

while true; do
    echo -n "Enter your query (or 'exit'): "
    read -r user_query
    
    if [ "$user_query" = "exit" ]; then
        break
    fi
    
    if [ -n "$user_query" ]; then
        echo ""
        python -m graphrag_anthropic_llamaindex.main search "$user_query" \
            --mode "drift" \
            --output-dir "$OUTPUT_DIR"
        echo ""
    fi
done

# Cleanup
print_step 7 "Demo completed!"
echo ""
print_info "Output files are saved in: $OUTPUT_DIR"
print_info "To run searches manually, use:"
echo "    python -m graphrag_anthropic_llamaindex.main search \"your query\" --mode drift --output-dir $OUTPUT_DIR"
echo ""

echo -e "${GREEN}✓ Demo completed successfully!${NC}"