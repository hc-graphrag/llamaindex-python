"""Local search module for GraphRAG."""

from .models import (
    Entity,
    Relationship,
    TextUnit,
    ContextResult
)
from .entity_mapper import EntityMapper
from .context_builder import LocalContextBuilder
from .retriever import LocalSearchRetriever
from .prompts import (
    get_local_search_prompt,
    LOCAL_SEARCH_PROMPT,
    LOCAL_SEARCH_WITH_CITATIONS_PROMPT,
    LOCAL_SEARCH_ANALYTICAL_PROMPT,
    LOCAL_SEARCH_SUMMARY_PROMPT
)

__all__ = [
    # Models
    "Entity",
    "Relationship",
    "TextUnit",
    "ContextResult",
    
    # Core components
    "EntityMapper",
    "LocalContextBuilder",
    "LocalSearchRetriever",
    
    # Prompts
    "get_local_search_prompt",
    "LOCAL_SEARCH_PROMPT",
    "LOCAL_SEARCH_WITH_CITATIONS_PROMPT",
    "LOCAL_SEARCH_ANALYTICAL_PROMPT",
    "LOCAL_SEARCH_SUMMARY_PROMPT",
]

__version__ = "0.1.0"