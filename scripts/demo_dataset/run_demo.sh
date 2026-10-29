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
# OUTPUT_DIR will be determined from config.yaml, not hardcoded

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

# Get output directory from config after determining which config file to use
get_output_dir() {
    local config_file="$1"
    if [ -f "$config_file" ]; then
        # Extract output_dir from config, default to "./demo_output" relative to config location
        python -c "
import yaml
import os
from pathlib import Path

try:
    config_path = Path('$config_file').resolve()
    config_dir = config_path.parent

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = config.get('output_dir', './demo_output')

    # Resolve relative to config file location
    if not os.path.isabs(output_dir):
        resolved_path = config_dir / output_dir
        print(str(resolved_path.resolve()))
    else:
        print(output_dir)

except Exception as e:
    print('$SCRIPT_DIR/demo_output')
"
    else
        echo "$SCRIPT_DIR/demo_output"
    fi
}

# Copy demo documents to data directory
print_step 1 "Preparing demo documents..."
DATA_DIR="$PROJECT_ROOT/data"
mkdir -p "$DATA_DIR"
cp -r "$DEMO_DATA_DIR"/* "$DATA_DIR/"
echo "  ✓ Copied $(ls -1 "$DEMO_DATA_DIR" | wc -l) documents to data directory"

# Set environment variables
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# Use demo-specific config file
DEMO_CONFIG_FILE="$SCRIPT_DIR/config.yaml"
CONFIG_FILE="$PROJECT_ROOT/config/config.yaml"
LLM_PROVIDER="anthropic"  # Default provider

# Check demo config first, then fallback to main config
if [ -f "$DEMO_CONFIG_FILE" ]; then
    CONFIG_FILE="$DEMO_CONFIG_FILE"
    print_info "Using demo config: $DEMO_CONFIG_FILE"
    # Extract llm_provider from demo config
    if grep -q "llm_provider:" "$CONFIG_FILE"; then
        LLM_PROVIDER=$(grep "llm_provider:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"' | tr -d "'")
        print_info "Detected LLM provider: $LLM_PROVIDER"
    fi
elif [ -f "$CONFIG_FILE" ]; then
    print_info "Using main config: $CONFIG_FILE"
    # Extract llm_provider from main config (if specified)
    if grep -q "llm_provider:" "$CONFIG_FILE"; then
        LLM_PROVIDER=$(grep "llm_provider:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"' | tr -d "'")
        print_info "Detected LLM provider: $LLM_PROVIDER"
    fi
fi

# Determine output directory from config
OUTPUT_DIR=$(get_output_dir "$CONFIG_FILE")
print_info "Output directory resolved to: $OUTPUT_DIR"

# Check if demo output already exists
SKIP_INDEXING=false
if [ -d "$OUTPUT_DIR/alopexdb" ]; then
    print_info "Existing demo output found. Checking if indexing can be skipped..."

    # Check if the output contains valid data
    ENTITY_COUNT=$(python -c "from graphrag_anthropic_llamaindex.db_manager import load_entities_db; print(len(load_entities_db('$OUTPUT_DIR')))" 2>/dev/null || echo "0")
    REL_COUNT=$(python -c "from graphrag_anthropic_llamaindex.db_manager import load_relationships_db; print(len(load_relationships_db('$OUTPUT_DIR')))" 2>/dev/null || echo "0")

    if [ "$ENTITY_COUNT" -gt 0 ] && [ "$REL_COUNT" -gt 0 ]; then
        print_info "Valid index found with $ENTITY_COUNT entities and $REL_COUNT relationships"
        print_info "Skipping document ingestion and using existing index..."
        SKIP_INDEXING=true
    else
        print_info "Invalid or empty index found. Will rebuild..."
        rm -rf "$OUTPUT_DIR"
    fi
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Check API key only if using Anthropic provider directly
if [ "$LLM_PROVIDER" = "anthropic" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    print_error "ANTHROPIC_API_KEY is required when using Anthropic provider."
    echo "    Option 1: Set API key: export ANTHROPIC_API_KEY='your-api-key'"
    echo "    Option 2: Use AWS Bedrock by setting 'llm_provider: bedrock' in config/config.yaml"
    echo "             (Bedrock does NOT require ANTHROPIC_API_KEY)"
    exit 1
elif [ "$LLM_PROVIDER" = "bedrock" ]; then
    print_info "Using AWS Bedrock provider - ANTHROPIC_API_KEY not required"
fi

# Run document ingestion only if needed
if [ "$SKIP_INDEXING" = false ]; then
    print_step 2 "Ingesting documents and building knowledge graph..."
    echo ""
    cd "$PROJECT_ROOT"
    python -m graphrag_anthropic_llamaindex.main --config "$CONFIG_FILE" add "$DATA_DIR"

    # Wait for indexing to complete
    sleep 2
else
    print_step 2 "Using existing knowledge graph index..."
fi

# Display graph statistics
print_step 3 "Knowledge graph statistics:"
echo ""
if [ -d "$OUTPUT_DIR/alopexdb" ]; then
    ENTITY_COUNT=$(python -c "from graphrag_anthropic_llamaindex.db_manager import load_entities_db; print(len(load_entities_db('$OUTPUT_DIR')))")
    echo "  • Entities extracted: $ENTITY_COUNT"
    REL_COUNT=$(python -c "from graphrag_anthropic_llamaindex.db_manager import load_relationships_db; print(len(load_relationships_db('$OUTPUT_DIR')))")
    echo "  • Relationships found: $REL_COUNT"
    COMM_COUNT=$(python -c "from graphrag_anthropic_llamaindex.db_manager import load_community_summaries_db; print(len(load_community_summaries_db('$OUTPUT_DIR')))")
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
    
    python -m graphrag_anthropic_llamaindex.main --config "$CONFIG_FILE" search "$query" \
        --mode "$mode" 2>/dev/null | head -20
    
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
    python -m graphrag_anthropic_llamaindex.main --config "$CONFIG_FILE" search "$SAMPLE_QUERY" \
        --mode "$mode" 2>/dev/null | head -15
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
        python -m graphrag_anthropic_llamaindex.main --config "$CONFIG_FILE" search "$user_query" \
            --mode "drift"
        echo ""
    fi
done

# Cleanup
print_step 7 "Demo completed!"
echo ""
print_info "Output files are saved in: $OUTPUT_DIR"
print_info "To run searches manually, use:"
echo "    python -m graphrag_anthropic_llamaindex.main --config $CONFIG_FILE search \"your query\" --mode drift"
echo ""

echo -e "${GREEN}✓ Demo completed successfully!${NC}"
