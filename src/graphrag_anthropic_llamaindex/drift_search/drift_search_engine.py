"""DRIFT Search Engine - Main orchestrator for DRIFT search functionality."""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from llama_index.core import Settings
from llama_index.core.vector_stores.types import VectorStore

from .context_builder import ContextBuilder
from .global_searcher import GlobalSearcher
from .local_searcher import LocalSearcher
from .models import SearchContext
from .response_generator import ResponseGenerator

logger = logging.getLogger(__name__)


class DriftSearchEngine:
    """DRIFT Search Engine for hybrid local-global search."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        vector_stores: Dict[str, VectorStore],
        llm: Optional[Any] = None,
    ):
        """
        Initialize DRIFT Search Engine.
        
        Args:
            config: DRIFT search configuration
            vector_stores: Mapping of vector store names to instances
            llm: LLM instance to use (defaults to Settings.llm)
        """
        self.config = config.get("drift_search", {})
        self.vector_stores = vector_stores
        self.llm = llm or Settings.llm
        
        # Initialize components
        self.local_searcher = LocalSearcher(
            vector_stores=vector_stores,
            config=self.config.get("local_search", {}),
        )
        self.global_searcher = GlobalSearcher(
            vector_stores=vector_stores,
            config=self.config.get("global_search", {}),
        )
        self.context_builder = ContextBuilder(
            config=self.config.get("context", {}),
        )
        self.response_generator = ResponseGenerator(
            llm=self.llm,
            config=self.config.get("response", {}),
        )
        
        logger.info("DRIFT Search Engine initialized")
    
    async def search(
        self,
        query: str,
        streaming: bool = False,
        include_context: bool = True,
    ) -> Union[str, AsyncGenerator[str, None], Tuple[str, Dict[str, Any]]]:
        """
        Execute DRIFT search.
        
        Args:
            query: Search query
            streaming: Enable streaming response
            include_context: Include context data in response
            
        Returns:
            Search response (string, async generator, or tuple with context)
        """
        if streaming:
            return self._search_streaming(query, include_context)
        else:
            return await self._search_non_streaming(query, include_context)
    
    async def _search_non_streaming(
        self,
        query: str,
        include_context: bool,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Non-streaming search implementation."""
        try:
            logger.info(f"Starting DRIFT search for query: {query[:100]}...")
            
            # Execute local and global search in parallel
            local_task = asyncio.create_task(
                self.local_searcher.search_entities(query)
            )
            global_task = asyncio.create_task(
                self.global_searcher.search_communities(query)
            )
            
            # Wait for both searches to complete
            local_results, global_results = await asyncio.gather(
                local_task, global_task
            )
            
            # Expand local context if configured
            if self.config.get("local_search", {}).get("relationship_depth", 0) > 0:
                local_results = await self.local_searcher.expand_context(
                    local_results,
                    max_hops=self.config["local_search"]["relationship_depth"],
                )
            
            # Build search context
            context = self.context_builder.build_search_context(
                query=query,
                local_results=local_results,
                global_results=global_results,
            )
            
            # Prioritize and trim context
            max_tokens = self.config.get("context", {}).get("max_tokens", 8000)
            context = context.trim_to_token_limit(max_tokens)
            
            # Generate response
            response = await self.response_generator.generate_response(context)
            
            if include_context:
                return response, context.to_dict()
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error during DRIFT search: {e}", exc_info=True)
            error_msg = f"DRIFT search failed: {str(e)}"
            
            if include_context:
                return error_msg, {"error": str(e)}
            else:
                return error_msg
    
    async def _search_streaming(
        self,
        query: str,
        include_context: bool,
    ) -> AsyncGenerator[str, None]:
        """Streaming search implementation."""
        try:
            logger.info(f"Starting DRIFT search (streaming) for query: {query[:100]}...")
            
            # Execute local and global search in parallel
            local_task = asyncio.create_task(
                self.local_searcher.search_entities(query)
            )
            global_task = asyncio.create_task(
                self.global_searcher.search_communities(query)
            )
            
            # Wait for both searches to complete
            local_results, global_results = await asyncio.gather(
                local_task, global_task
            )
            
            # Expand local context if configured
            if self.config.get("local_search", {}).get("relationship_depth", 0) > 0:
                local_results = await self.local_searcher.expand_context(
                    local_results,
                    max_hops=self.config["local_search"]["relationship_depth"],
                )
            
            # Build search context
            context = self.context_builder.build_search_context(
                query=query,
                local_results=local_results,
                global_results=global_results,
            )
            
            # Prioritize and trim context
            max_tokens = self.config.get("context", {}).get("max_tokens", 8000)
            context = context.trim_to_token_limit(max_tokens)
            
            # Store context for later retrieval if needed
            if include_context:
                self._last_context = context.to_dict()
            
            # Stream response
            async for chunk in self.response_generator.stream_response(context):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error during DRIFT search: {e}", exc_info=True)
            yield f"DRIFT search failed: {str(e)}"
    
    def get_last_context(self) -> Optional[Dict[str, Any]]:
        """Get the context from the last streaming search."""
        return getattr(self, "_last_context", None)
    
    async def search_sync(
        self,
        query: str,
        include_context: bool = True,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Synchronous wrapper for search (for non-async contexts).
        
        Args:
            query: Search query
            include_context: Include context data in response
            
        Returns:
            Search response
        """
        return await self.search(query, streaming=False, include_context=include_context)
    
    def validate_configuration(self) -> bool:
        """
        Validate DRIFT search configuration.
        
        Returns:
            True if configuration is valid
        """
        required_stores = ["main", "entity", "community"]
        
        for store_name in required_stores:
            if store_name not in self.vector_stores:
                logger.warning(f"Missing required vector store: {store_name}")
                return False
        
        if not self.llm:
            logger.warning("No LLM configured for DRIFT search")
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get search engine statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "config": {
                "local_search": self.config.get("local_search", {}),
                "global_search": self.config.get("global_search", {}),
                "context": self.config.get("context", {}),
                "response": self.config.get("response", {}),
            },
            "vector_stores": list(self.vector_stores.keys()),
            "llm_configured": self.llm is not None,
        }
        
        return stats