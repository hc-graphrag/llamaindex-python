"""Local search retriever implementation."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic

from .entity_mapper import EntityMapper
from .context_builder import LocalContextBuilder
from .prompts import get_local_search_prompt
from .models import Entity, Relationship

logger = logging.getLogger(__name__)


class LocalSearchRetriever(BaseRetriever):
    """Local search retriever that uses entity mapping and context building."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        entity_mapper: Optional[EntityMapper] = None,
        context_builder: Optional[LocalContextBuilder] = None,
        llm: Optional[Any] = None,
        prompt_style: str = "default",
        top_k_entities: int = 10,
        max_context_tokens: int = 4000,
        **kwargs
    ):
        """
        Initialize the LocalSearchRetriever.
        
        Args:
            config: Configuration dictionary
            entity_mapper: EntityMapper instance (optional)
            context_builder: LocalContextBuilder instance (optional)
            llm: Language model instance (optional)
            prompt_style: Style of prompt to use
            top_k_entities: Number of entities to retrieve
            max_context_tokens: Maximum tokens for context
            **kwargs: Additional arguments for BaseRetriever
        """
        super().__init__(**kwargs)
        
        self.config = config
        self.prompt_style = prompt_style
        self.top_k_entities = top_k_entities
        
        # Initialize components
        self.entity_mapper = entity_mapper or EntityMapper(
            config=config,
            top_k=top_k_entities
        )
        
        self.context_builder = context_builder or LocalContextBuilder(
            max_context_tokens=max_context_tokens,
            include_entity_descriptions=True,
            include_relationship_descriptions=True,
            format_style="structured"
        )
        
        # Initialize LLM
        if llm is not None:
            self.llm = llm
        else:
            # Try to use Settings.llm but handle the case where it's not set
            try:
                if hasattr(Settings, '_llm') and Settings._llm is not None:
                    self.llm = Settings._llm
                else:
                    # Default to Anthropic if available
                    self.llm = Anthropic(
                        model=config.get("llm", {}).get("model", "claude-3-5-sonnet-20241022"),
                        temperature=config.get("llm", {}).get("temperature", 0.7)
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
                self.llm = None
        
        # Store relationships (would be loaded from storage in production)
        self.relationships: List[Relationship] = []
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Synchronously retrieve results for the query.
        
        Args:
            query_bundle: Query bundle containing the search query
            
        Returns:
            List of NodeWithScore objects containing the search results
        """
        # Run async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._aretrieve(query_bundle))
        finally:
            loop.close()
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Asynchronously retrieve results for the query.
        
        Args:
            query_bundle: Query bundle containing the search query
            
        Returns:
            List of NodeWithScore objects containing the search results
        """
        query = query_bundle.query_str
        logger.info(f"Local search for query: {query[:100]}...")
        
        try:
            # Step 1: Map query to entities
            entities = self.entity_mapper.map_query_to_entities(
                query=query,
                top_k=self.top_k_entities
            )
            
            if not entities:
                logger.warning("No entities found for query")
                return self._create_empty_response(query)
            
            # Step 2: Get relationships for entities (simplified for now)
            relationships = self._get_relationships_for_entities(entities)
            
            # Step 3: Build context
            context_result = self.context_builder.build_context(
                query=query,
                entities=entities,
                relationships=relationships,
                text_units=[]  # Text units would be loaded from storage
            )
            
            # Step 4: Generate response using LLM
            if self.llm is None:
                # If no LLM, return the context as is
                return self._create_context_only_response(context_result)
            
            prompt_template = get_local_search_prompt(self.prompt_style)
            prompt = prompt_template.format(
                context=context_result.context_text,
                query=query
            )
            
            # Get response from LLM
            response = await self._get_llm_response(prompt)
            
            # Create node with the response
            node = TextNode(
                text=response,
                metadata={
                    "search_type": "local",
                    "num_entities": len(entities),
                    "num_relationships": len(relationships),
                    "prompt_style": self.prompt_style,
                    "query": query
                }
            )
            
            return [NodeWithScore(node=node, score=1.0)]
            
        except Exception as e:
            logger.error(f"Error in local search: {e}")
            return self._create_error_response(query, str(e))
    
    def _get_relationships_for_entities(
        self,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Get relationships for the given entities.
        
        Args:
            entities: List of entities to get relationships for
            
        Returns:
            List of relevant relationships
        """
        # In a production implementation, this would query a relationship store
        # For now, return empty list or mock data
        entity_ids = {e.id for e in entities}
        
        # Filter stored relationships to those involving our entities
        relevant_relationships = [
            rel for rel in self.relationships
            if rel.source_id in entity_ids or rel.target_id in entity_ids
        ]
        
        return relevant_relationships
    
    async def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response
        """
        if hasattr(self.llm, 'acomplete'):
            # Async completion
            response = await self.llm.acomplete(prompt)
            return response.text
        elif hasattr(self.llm, 'complete'):
            # Sync completion (run in executor)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.llm.complete,
                prompt
            )
            return response.text
        else:
            # Fallback
            return "LLM response not available"
    
    def _create_empty_response(self, query: str) -> List[NodeWithScore]:
        """Create an empty response when no entities are found."""
        node = TextNode(
            text="No relevant information found for your query.",
            metadata={
                "search_type": "local",
                "query": query,
                "status": "no_entities_found"
            }
        )
        return [NodeWithScore(node=node, score=0.0)]
    
    def _create_context_only_response(
        self,
        context_result
    ) -> List[NodeWithScore]:
        """Create a response with just the context when no LLM is available."""
        node = TextNode(
            text=context_result.context_text,
            metadata={
                "search_type": "local",
                "query": context_result.query,
                "num_entities": len(context_result.entities),
                "num_relationships": len(context_result.relationships),
                "status": "context_only"
            }
        )
        return [NodeWithScore(node=node, score=0.8)]
    
    def _create_error_response(
        self,
        query: str,
        error: str
    ) -> List[NodeWithScore]:
        """Create an error response."""
        node = TextNode(
            text=f"An error occurred during local search: {error}",
            metadata={
                "search_type": "local",
                "query": query,
                "status": "error",
                "error": error
            }
        )
        return [NodeWithScore(node=node, score=0.0)]