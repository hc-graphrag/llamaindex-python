"""Entity mapper for local search - maps queries to relevant entities."""

from typing import List, Optional, Dict, Any
import logging
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle

from ..vector_store_manager import get_vector_store, get_index
from .models import Entity

logger = logging.getLogger(__name__)


class EntityMapper:
    """Maps queries to relevant entities using vector similarity search."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        entity_index: Optional[VectorStoreIndex] = None,
        top_k: int = 10
    ):
        """
        Initialize the EntityMapper.
        
        Args:
            config: Configuration dictionary
            entity_index: Pre-built entity vector index (optional)
            top_k: Number of top entities to retrieve
        """
        self.config = config
        self.top_k = top_k
        
        # Initialize entity index
        if entity_index is not None:
            self.entity_index = entity_index
        else:
            # Try to load from vector store
            entity_vector_store = get_vector_store(config, store_type="entity")
            if entity_vector_store:
                self.entity_index = get_index(
                    storage_dir=None,
                    vector_store=entity_vector_store,
                    index_type="entity"
                )
            else:
                logger.warning("No entity vector store configured, EntityMapper will return empty results")
                self.entity_index = None
    
    def map_query_to_entities(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Entity]:
        """
        Map a query to relevant entities using vector similarity search.
        
        Args:
            query: The search query
            top_k: Override for number of entities to retrieve
            
        Returns:
            List of relevant Entity objects
        """
        if self.entity_index is None:
            logger.warning("No entity index available, returning empty list")
            return []
        
        try:
            # Use the provided top_k or fall back to instance default
            k = top_k or self.top_k
            
            # Create a retriever from the index
            retriever = self.entity_index.as_retriever(similarity_top_k=k)
            
            # Perform similarity search
            query_bundle = QueryBundle(query_str=query)
            nodes = retriever.retrieve(query_bundle)
            
            # Convert nodes to Entity objects
            entities = []
            for node in nodes:
                # Extract entity information from node metadata
                metadata = node.metadata or {}
                
                entity = Entity(
                    id=metadata.get("id", node.node_id),
                    name=metadata.get("name", node.text[:100]),  # Use text preview as fallback
                    type=metadata.get("type"),
                    description=node.text,
                    properties=metadata.get("properties", {}),
                    embedding=None  # We don't expose embeddings in the result
                )
                entities.append(entity)
                
            logger.info(f"Retrieved {len(entities)} entities for query: {query[:50]}...")
            return entities
            
        except Exception as e:
            logger.error(f"Error mapping query to entities: {e}")
            return []
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """
        Retrieve a specific entity by its ID.
        
        Args:
            entity_id: The entity ID to retrieve
            
        Returns:
            Entity object if found, None otherwise
        """
        # This would require direct access to the entity storage
        # For now, this is a placeholder
        logger.warning(f"get_entity_by_id not fully implemented for ID: {entity_id}")
        return None