# Global Search Usage Guide

## Introduction

Global Search is a powerful feature that enables comprehensive, high-level searches across your entire knowledge graph. Unlike local search which focuses on specific documents or entities, global search leverages community structures to provide broad, thematic answers by analyzing patterns and relationships across all your data.

## When to Use Global Search

### Best Use Cases

**Global Search** is ideal for:
- **Thematic Questions**: "What are the main themes in this dataset?"
- **Trend Analysis**: "What patterns emerge across all documents?"
- **Comprehensive Overviews**: "Summarize the key findings"
- **Strategic Insights**: "What are the strategic priorities?"
- **Cross-Document Synthesis**: "How do different topics relate to each other?"

**Local Search** is better for:
- Specific fact retrieval
- Detailed information about particular entities
- Document-specific queries
- Precise technical details

## Getting Started

### Prerequisites

Before using global search, ensure you have:

1. **Indexed Documents**: Documents must be processed and indexed
2. **Community Detection**: Leiden algorithm must have been run
3. **Community Reports**: Reports must be generated from detected communities
4. **Configuration**: Proper configuration in `config.yaml`

### Basic Setup

1. **Configure Global Search** in `config.yaml`:

```yaml
global_search:
  response_type: "Multiple Paragraphs"  # Output format
  max_tokens: 3000                      # Maximum response length
  temperature: 0.0                      # LLM temperature (0 = deterministic)
  top_p: 1.0                            # Top-p sampling parameter
  max_concurrent: 50                    # Parallel processing limit
  batch_size: 16                        # Reports per batch
  min_community_rank: 0                 # Minimum community level
  normalize_community_weight: true      # Weight normalization
```

2. **Ensure Community Detection** has been completed:

```bash
# Add documents with community detection
python -m graphrag_anthropic_llamaindex add

# This creates:
# - Entity graph
# - Community structure via Leiden algorithm
# - Community reports for global search
```

## Command-Line Usage

### Basic Global Search

```bash
# Perform a global search
python -m graphrag_anthropic_llamaindex search \
  "What are the main topics discussed?" \
  --mode global
```

### Specifying Response Format

```bash
# Get a concise single paragraph
python -m graphrag_anthropic_llamaindex search \
  "Summarize the key findings" \
  --mode global \
  --response-type "Single Paragraph"

# Get a bulleted list
python -m graphrag_anthropic_llamaindex search \
  "List the main themes" \
  --mode global \
  --response-type "List"

# Get structured JSON output
python -m graphrag_anthropic_llamaindex search \
  "Extract key insights" \
  --mode global \
  --response-type "JSON" \
  --output-format json
```

### Filtering by Community Level

```bash
# Only use high-level communities (strategic overview)
python -m graphrag_anthropic_llamaindex search \
  "What are the strategic priorities?" \
  --mode global \
  --min-community-rank 8

# Use all community levels (comprehensive)
python -m graphrag_anthropic_llamaindex search \
  "Explain everything about AI" \
  --mode global \
  --min-community-rank 0
```

### Output Format Options

```bash
# Markdown format (default)
python -m graphrag_anthropic_llamaindex search \
  "Describe the system architecture" \
  --mode global \
  --output-format markdown

# JSON format for programmatic use
python -m graphrag_anthropic_llamaindex search \
  "Extract technical specifications" \
  --mode global \
  --output-format json > results.json
```

## Programmatic Usage

### Basic Python Integration

```python
from graphrag_anthropic_llamaindex import create_global_search_engine

# Create search engine
engine = create_global_search_engine("config.yaml")

# Perform search
response = engine.query("What are the emerging trends?")
print(response.response)

# Access metadata
for source in response.source_nodes:
    print(f"Community: {source.node.metadata.get('community_id')}")
    print(f"Score: {source.score}")
```

### Advanced Integration

```python
from graphrag_anthropic_llamaindex.global_search import GlobalSearchRetriever
from graphrag_anthropic_llamaindex.vector_store_manager import VectorStoreManager
from llama_index.llms.anthropic import Anthropic

# Initialize components
llm = Anthropic(
    api_key="your-api-key",
    model="claude-3-sonnet-20240229"
)

vector_store = VectorStoreManager(config)
community_reports = vector_store.get_community_reports()

# Create retriever with custom settings
retriever = GlobalSearchRetriever(
    llm=llm,
    community_reports=community_reports,
    response_type="Multiple Paragraphs",
    min_community_rank=5,  # Focus on higher-level insights
    max_concurrent=100,     # Increase parallelism
    temperature=0.0         # Deterministic responses
)

# Use retriever
results = retriever.retrieve("What patterns emerge from the data?")

# Process results with traceability
for result in results:
    print(f"Answer: {result.node.text}")
    
    # Trace back to source documents
    traceability = result.node.metadata.get("traceability", {})
    doc_ids = traceability.get("document_ids", [])
    print(f"Based on {len(doc_ids)} documents")
```

### Async Operations

```python
import asyncio
from graphrag_anthropic_llamaindex.global_search import GlobalSearchRetriever

async def perform_searches():
    retriever = GlobalSearchRetriever(
        llm=llm,
        community_reports=community_reports
    )
    
    # Perform multiple searches concurrently
    queries = [
        "What are the main themes?",
        "What patterns are visible?",
        "What are the key relationships?"
    ]
    
    tasks = [retriever.aretrieve(q) for q in queries]
    results = await asyncio.gather(*tasks)
    
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        print(f"Answer: {result[0].node.text if result else 'No answer'}")
        print("---")

# Run async searches
asyncio.run(perform_searches())
```

## Understanding Response Types

### Multiple Paragraphs (Default)

Best for comprehensive answers with context and nuance:

```bash
python -m graphrag_anthropic_llamaindex search \
  "Explain the technological landscape" \
  --mode global \
  --response-type "Multiple Paragraphs"
```

**Output Example:**
```
The technological landscape reveals several interconnected themes...

First, artificial intelligence has emerged as a transformative force...

Additionally, cloud computing infrastructure continues to evolve...

Finally, the intersection of these technologies suggests...
```

### Single Paragraph

Best for concise summaries:

```bash
python -m graphrag_anthropic_llamaindex search \
  "Summarize the main findings" \
  --mode global \
  --response-type "Single Paragraph"
```

**Output Example:**
```
The analysis reveals three primary findings: widespread AI adoption across industries, increasing focus on sustainability metrics, and growing importance of data privacy regulations, all converging to reshape business strategies.
```

### List Format

Best for enumerated insights:

```bash
python -m graphrag_anthropic_llamaindex search \
  "What are the key challenges?" \
  --mode global \
  --response-type "List"
```

**Output Example:**
```
• Scalability limitations in current infrastructure
• Integration complexity between legacy and modern systems
• Skills gap in emerging technologies
• Regulatory compliance across jurisdictions
• Data security and privacy concerns
```

### JSON Format

Best for structured data extraction:

```bash
python -m graphrag_anthropic_llamaindex search \
  "Extract the main concepts" \
  --mode global \
  --response-type "JSON" \
  --output-format json
```

**Output Example:**
```json
{
  "summary": "Analysis of main concepts across the knowledge graph",
  "key_points": [
    {
      "concept": "Digital Transformation",
      "importance": 0.92,
      "related_topics": ["AI", "Cloud", "Automation"]
    },
    {
      "concept": "Sustainability",
      "importance": 0.87,
      "related_topics": ["ESG", "Carbon Neutral", "Green Tech"]
    }
  ],
  "confidence": 0.89
}
```

## Performance Optimization

### Adjusting Batch Size

For large knowledge graphs, optimize batch processing:

```yaml
global_search:
  batch_size: 8   # Smaller batches for very large graphs
  # or
  batch_size: 32  # Larger batches for smaller graphs
```

### Controlling Concurrency

Balance between speed and API limits:

```yaml
global_search:
  max_concurrent: 10   # Conservative for rate-limited APIs
  # or
  max_concurrent: 100  # Aggressive for high-throughput APIs
```

### Community Rank Filtering

Use `min_community_rank` to control scope:

- **0-3**: Include all details (slowest, most comprehensive)
- **4-7**: Balanced detail and performance
- **8+**: High-level overview only (fastest, most abstract)

```bash
# Quick strategic overview
python -m graphrag_anthropic_llamaindex search \
  "What should we focus on?" \
  --mode global \
  --min-community-rank 8

# Detailed analysis
python -m graphrag_anthropic_llamaindex search \
  "Explain all technical aspects" \
  --mode global \
  --min-community-rank 0
```

## Troubleshooting

### No Results Returned

**Problem**: Search returns empty results

**Solutions**:
1. Check if community reports exist:
   ```bash
   ls graphrag_output/communities/
   ```
2. Verify community detection was run during indexing
3. Lower `min_community_rank` to include more communities

### Poor Quality Answers

**Problem**: Answers are too generic or miss important points

**Solutions**:
1. Adjust `min_community_rank` to include more detailed communities
2. Use "Multiple Paragraphs" response type for more comprehensive answers
3. Ensure community weights are properly normalized in config

### Slow Performance

**Problem**: Searches take too long

**Solutions**:
1. Increase `min_community_rank` to process fewer communities
2. Reduce `batch_size` to decrease memory usage
3. Increase `max_concurrent` if API allows
4. Use caching for repeated queries

### API Rate Limiting

**Problem**: Getting rate limit errors

**Solutions**:
1. Reduce `max_concurrent` in configuration
2. Implement exponential backoff
3. Use caching to avoid repeated API calls

## Best Practices

### 1. Choose the Right Mode

```python
# Use auto mode to let the system decide
router = SearchModeRouter(config, mode="auto")

# Or explicitly choose based on query type
if "specific document" in query:
    mode = "local"
elif "overall themes" in query:
    mode = "global"
```

### 2. Optimize for Your Use Case

**For Research & Analysis**:
```yaml
global_search:
  response_type: "Multiple Paragraphs"
  min_community_rank: 0  # Include all details
  temperature: 0.0        # Consistent results
```

**For Quick Insights**:
```yaml
global_search:
  response_type: "Single Paragraph"
  min_community_rank: 7  # High-level only
  temperature: 0.3        # Some variation
```

**For Data Extraction**:
```yaml
global_search:
  response_type: "JSON"
  min_community_rank: 3
  temperature: 0.0
```

### 3. Monitor and Log

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

retriever = GlobalSearchRetriever(
    llm=llm,
    community_reports=reports
)

# Log search operations
logger.info(f"Searching: {query}")
results = retriever.retrieve(query)
logger.info(f"Found {len(results)} results")

# Track performance
import time
start = time.time()
results = retriever.retrieve(query)
elapsed = time.time() - start
logger.info(f"Search took {elapsed:.2f} seconds")
```

### 4. Implement Caching

```python
from functools import lru_cache
import hashlib

class CachedGlobalSearch:
    def __init__(self, retriever):
        self.retriever = retriever
        self.cache = {}
    
    def search(self, query, cache_ttl=3600):
        # Create cache key
        key = hashlib.md5(query.encode()).hexdigest()
        
        # Check cache
        if key in self.cache:
            cached_time, cached_result = self.cache[key]
            if time.time() - cached_time < cache_ttl:
                return cached_result
        
        # Perform search
        result = self.retriever.retrieve(query)
        
        # Cache result
        self.cache[key] = (time.time(), result)
        return result
```

## Examples by Domain

### Business Intelligence

```bash
# Strategic overview
python -m graphrag_anthropic_llamaindex search \
  "What are our competitive advantages?" \
  --mode global \
  --min-community-rank 7

# Market analysis
python -m graphrag_anthropic_llamaindex search \
  "What market trends are emerging?" \
  --mode global \
  --response-type "List"
```

### Research & Academia

```bash
# Literature review
python -m graphrag_anthropic_llamaindex search \
  "What are the main research themes?" \
  --mode global \
  --response-type "Multiple Paragraphs"

# Finding connections
python -m graphrag_anthropic_llamaindex search \
  "How do different research areas connect?" \
  --mode global
```

### Technical Documentation

```bash
# System overview
python -m graphrag_anthropic_llamaindex search \
  "Describe the system architecture" \
  --mode global \
  --response-type "Multiple Paragraphs"

# Component relationships
python -m graphrag_anthropic_llamaindex search \
  "How do system components interact?" \
  --mode global \
  --output-format json
```

## Conclusion

Global Search provides powerful capabilities for understanding large-scale patterns and themes in your knowledge graph. By properly configuring and using the various options available, you can extract meaningful insights that would be difficult to obtain through traditional search methods.

Remember to:
- Choose appropriate response types for your use case
- Adjust community rank filtering based on desired detail level
- Monitor performance and optimize settings as needed
- Use caching for frequently accessed queries
- Combine with local search for comprehensive information retrieval

For more technical details, see the [API Documentation](../api/global_search.md).