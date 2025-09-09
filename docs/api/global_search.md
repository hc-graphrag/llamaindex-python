# Global Search API Documentation

## Overview

The Global Search functionality provides comprehensive, high-level search capabilities across your entire knowledge graph using a Map-Reduce pattern inspired by Microsoft GraphRAG. This API enables you to perform community-based searches that leverage hierarchical graph structures for comprehensive answers.

## Core Components

### GlobalSearchRetriever

The main retriever class that implements LlamaIndex's `BaseRetriever` interface for global search operations.

```python
from graphrag_anthropic_llamaindex.global_search import GlobalSearchRetriever

retriever = GlobalSearchRetriever(
    llm=llm_instance,
    community_reports=community_reports_list,
    response_type="Multiple Paragraphs",
    max_tokens=3000,
    temperature=0.0,
    top_p=1.0,
    max_concurrent=50,
    min_community_rank=0
)
```

#### Parameters

- **llm** (`BaseLLM`): The LLM instance to use for Map-Reduce operations
- **community_reports** (`List[Dict]`): List of community report dictionaries
- **response_type** (`str`, optional): Output format type. Options:
  - `"Multiple Paragraphs"` (default): Detailed multi-paragraph response
  - `"Single Paragraph"`: Concise single paragraph
  - `"List"`: Bullet-point list format
  - `"JSON"`: Structured JSON response
- **max_tokens** (`int`, optional): Maximum tokens for LLM responses. Default: 3000
- **temperature** (`float`, optional): LLM temperature. Default: 0.0
- **top_p** (`float`, optional): LLM top-p sampling. Default: 1.0
- **max_concurrent** (`int`, optional): Maximum concurrent Map operations. Default: 50
- **min_community_rank** (`int`, optional): Minimum community rank to include. Default: 0

#### Methods

##### retrieve(query_str)

Synchronous retrieval method.

```python
results = retriever.retrieve("What are the main technological trends?")
for node_with_score in results:
    print(f"Score: {node_with_score.score}")
    print(f"Text: {node_with_score.node.text}")
    print(f"Traceability: {node_with_score.node.metadata.get('traceability')}")
```

##### aretrieve(query_str)

Asynchronous retrieval method.

```python
results = await retriever.aretrieve("What are the main technological trends?")
```

### SearchModeRouter

Unified router for handling different search modes (local, global, drift).

```python
from graphrag_anthropic_llamaindex.global_search import SearchModeRouter

router = SearchModeRouter(
    config=config_dict,
    mode="global",
    vector_store_main=main_store,
    vector_store_entity=entity_store,
    vector_store_community=community_store,
    response_type="Multiple Paragraphs",
    min_community_rank=0,
    output_format="markdown"
)
```

#### Parameters

- **config** (`Dict`): Configuration dictionary
- **mode** (`str`): Search mode - "local", "global", "drift", or "auto"
- **vector_store_main**: Main document vector store
- **vector_store_entity**: Entity vector store
- **vector_store_community**: Community vector store
- **response_type** (`str`, optional): Response format type
- **min_community_rank** (`int`, optional): Minimum community rank filter
- **output_format** (`str`, optional): Output format - "markdown" or "json"

#### Methods

##### search(query)

Execute search based on configured mode.

```python
results = router.search("Explain the key findings about AI development")
```

### CommunityContextBuilder

Manages community report context and batching for Map operations.

```python
from graphrag_anthropic_llamaindex.global_search.context_builder import CommunityContextBuilder

context_builder = CommunityContextBuilder(
    community_reports=reports,
    max_tokens=8192,
    response_type="Multiple Paragraphs"
)
```

#### Methods

##### build_context(min_community_rank=0)

Build context batches from community reports.

```python
context_text, context_data = context_builder.build_context(min_community_rank=5)
```

### MapProcessor

Handles parallel Map phase processing.

```python
from graphrag_anthropic_llamaindex.global_search.map_processor import MapProcessor

map_processor = MapProcessor(
    llm=llm_instance,
    response_type="Multiple Paragraphs",
    max_tokens=2000,
    temperature=0.0,
    max_concurrent=50
)
```

#### Methods

##### process_batch(query, contexts)

Process multiple contexts in parallel.

```python
map_results = await map_processor.process_batch(
    query="What are the trends?",
    contexts=context_list
)
```

### ReduceProcessor

Handles Reduce phase to combine Map results.

```python
from graphrag_anthropic_llamaindex.global_search.reduce_processor import ReduceProcessor

reduce_processor = ReduceProcessor(
    llm=llm_instance,
    response_type="Multiple Paragraphs",
    max_tokens=3000,
    temperature=0.0
)
```

#### Methods

##### reduce(query, map_results)

Combine Map results into final answer.

```python
final_answer = await reduce_processor.reduce(
    query="What are the trends?",
    map_results=map_results_list,
    output_format="markdown"
)
```

## Configuration

### YAML Configuration Example

```yaml
global_search:
  response_type: "Multiple Paragraphs"  # Output format
  max_tokens: 3000                      # Max tokens for responses
  temperature: 0.0                      # LLM temperature
  top_p: 1.0                            # Top-p sampling
  max_concurrent: 50                    # Max parallel Map operations
  batch_size: 16                        # Community reports per batch
  min_community_rank: 0                 # Min community rank filter
  normalize_community_weight: true      # Normalize community weights
```

### Programmatic Configuration

```python
config = {
    "global_search": {
        "response_type": "Multiple Paragraphs",
        "max_tokens": 3000,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_concurrent": 50,
        "batch_size": 16,
        "min_community_rank": 0,
        "normalize_community_weight": True
    }
}
```

## Usage Examples

### Basic Global Search

```python
from graphrag_anthropic_llamaindex.global_search import GlobalSearchRetriever
from llama_index.llms.anthropic import Anthropic

# Initialize LLM
llm = Anthropic(api_key="your-api-key", model="claude-3-sonnet-20240229")

# Get community reports from vector store
community_reports = vector_store.get_community_reports()

# Create retriever
retriever = GlobalSearchRetriever(
    llm=llm,
    community_reports=community_reports,
    response_type="Multiple Paragraphs"
)

# Perform search
results = retriever.retrieve("What are the main themes in the knowledge graph?")

# Process results
for result in results:
    print(f"Answer: {result.node.text}")
    print(f"Score: {result.score}")
    
    # Access traceability information
    traceability = result.node.metadata.get("traceability", {})
    print(f"Report IDs: {traceability.get('report_ids', [])}")
    print(f"Document IDs: {traceability.get('document_ids', [])}")
```

### Using SearchModeRouter

```python
from graphrag_anthropic_llamaindex.global_search import SearchModeRouter

# Create router with auto mode selection
router = SearchModeRouter(
    config=config,
    mode="auto",  # Automatically select best mode
    vector_store_main=main_store,
    vector_store_entity=entity_store,
    vector_store_community=community_store
)

# Search will automatically use the best mode
results = router.search("Explain the system architecture")
```

### Filtered Search with Min Community Rank

```python
# Only use high-level communities (rank >= 8)
retriever = GlobalSearchRetriever(
    llm=llm,
    community_reports=community_reports,
    min_community_rank=8,
    response_type="Single Paragraph"
)

results = retriever.retrieve("What are the strategic priorities?")
```

### JSON Output Format

```python
retriever = GlobalSearchRetriever(
    llm=llm,
    community_reports=community_reports,
    response_type="JSON"
)

results = retriever.retrieve("List the key findings")

# Parse JSON response
import json
for result in results:
    data = json.loads(result.node.text)
    print(f"Key Points: {data.get('key_points', [])}")
    print(f"Summary: {data.get('summary', '')}")
```

### Async Usage with LlamaIndex Query Engine

```python
from llama_index.core.query_engine import RetrieverQueryEngine

# Create query engine with global search retriever
query_engine = RetrieverQueryEngine(
    retriever=retriever
)

# Use in async context
async def search():
    response = await query_engine.aquery("What are the emerging patterns?")
    print(response.response)
    
    # Access source nodes
    for node in response.source_nodes:
        print(f"Source: {node.node.metadata}")

# Run async search
import asyncio
asyncio.run(search())
```

### Integration with LlamaIndex Chat Engine

```python
from llama_index.core.chat_engine import SimpleChatEngine

# Create chat engine with global search
chat_engine = SimpleChatEngine.from_defaults(
    retriever=retriever,
    llm=llm
)

# Interactive chat with global search context
response = chat_engine.chat("Tell me about the main topics in the graph")
print(response.response)

# Follow-up questions maintain context
follow_up = chat_engine.chat("Can you elaborate on the first topic?")
print(follow_up.response)
```

## Error Handling

```python
from graphrag_anthropic_llamaindex.global_search import GlobalSearchRetriever

try:
    retriever = GlobalSearchRetriever(
        llm=llm,
        community_reports=community_reports
    )
    results = retriever.retrieve("Your query")
    
except ValueError as e:
    if "No community reports" in str(e):
        print("No community reports available. Run indexing first.")
    elif "Community weights not set" in str(e):
        print("Community weights missing. Check Leiden detection.")
    else:
        raise
        
except Exception as e:
    print(f"Search failed: {e}")
```

## Performance Considerations

### Batching and Concurrency

- **Batch Size**: Controls how many community reports are processed per Map operation
- **Max Concurrent**: Limits parallel Map operations to prevent API rate limiting
- **Optimal Settings**:
  - Small graphs (<100 communities): `batch_size=16, max_concurrent=10`
  - Medium graphs (100-1000 communities): `batch_size=16, max_concurrent=50`
  - Large graphs (>1000 communities): `batch_size=8, max_concurrent=100`

### Token Management

- **Map Phase**: Each batch uses up to `max_tokens/2` tokens
- **Reduce Phase**: Final combination uses full `max_tokens`
- **Context Window**: Ensure batch context fits within LLM context limits

### Caching Strategies

```python
# Cache community reports in memory
class CachedGlobalSearchRetriever(GlobalSearchRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
    
    def retrieve(self, query_str):
        if query_str in self._cache:
            return self._cache[query_str]
        
        results = super().retrieve(query_str)
        self._cache[query_str] = results
        return results
```

## Traceability and Debugging

Each search result includes comprehensive traceability information:

```python
traceability = result.node.metadata.get("traceability", {})

# Community-level tracing
report_ids = traceability.get("report_ids", [])
print(f"Used {len(report_ids)} community reports")

# Document-level tracing
document_ids = traceability.get("document_ids", [])
print(f"Referenced {len(document_ids)} documents")

# Chunk-level tracing
chunk_ids = traceability.get("chunk_ids", [])
print(f"Based on {len(chunk_ids)} text chunks")

# Entity-level tracing
entity_ids = traceability.get("entity_ids", [])
print(f"Involved {len(entity_ids)} entities")
```

## Best Practices

1. **Community Weighting**: Always ensure communities have proper weights from Leiden detection
2. **Response Types**: Use "Multiple Paragraphs" for comprehensive answers, "JSON" for structured data
3. **Rank Filtering**: Use `min_community_rank` to focus on appropriate abstraction levels
4. **Error Handling**: Always handle empty community reports and missing weights
5. **Performance**: Adjust `max_concurrent` based on your API rate limits
6. **Caching**: Implement caching for frequently asked queries
7. **Monitoring**: Track Map-Reduce latencies and success rates