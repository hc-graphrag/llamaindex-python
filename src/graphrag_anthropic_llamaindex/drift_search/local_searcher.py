"""Local entity search for DRIFT Search."""

import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from llama_index.core.vector_stores.types import VectorStore, VectorStoreQuery, VectorStoreQueryMode

from ..db_manager import load_entities_db, load_relationships_db
from .models import Entity, TextUnit

logger = logging.getLogger(__name__)


class LocalSearcher:
    """Local entity search component for DRIFT search."""
    
    def __init__(
        self,
        vector_stores: Dict[str, VectorStore],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize local searcher.
        
        Args:
            vector_stores: Mapping of vector store names to instances
            config: Local search configuration
        """
        self.entity_store = vector_stores.get("entity")
        self.main_store = vector_stores.get("main")
        self.config = config or {}
        
        # Configuration
        self.entity_top_k = self.config.get("entity_top_k", 10)
        self.relationship_depth = self.config.get("relationship_depth", 2)
        self.include_text_units = self.config.get("include_text_units", True)
        self.text_unit_top_k = self.config.get("text_unit_top_k", 5)
        
        # Cache for entities and relationships
        self._entities_cache = None
        self._relationships_cache = None
        
        logger.info(f"LocalSearcher initialized with top_k={self.entity_top_k}")
    
    async def search_entities(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Entity]:
        """
        Search for relevant entities.
        
        Args:
            query: Search query
            top_k: Number of top entities to return
            
        Returns:
            List of relevant entities
        """
        if not self.entity_store:
            logger.warning("Entity vector store not available")
            return []
        
        top_k = top_k or self.entity_top_k
        
        try:
            # Create vector store query
            query_obj = VectorStoreQuery(
                query_str=query,
                mode=VectorStoreQueryMode.DEFAULT,
                similarity_top_k=top_k,
            )
            
            # Search entity store
            result = self.entity_store.query(query_obj)
            
            # Load entity data if not cached
            if self._entities_cache is None:
                # Use default output_dir from config or fallback
                output_dir = "graphrag_output"
                self._entities_cache = load_entities_db(output_dir)
            
            # Convert results to Entity objects
            entities = []
            for node in result.nodes:
                entity_id = node.node.node_id
                
                # Find entity in cache
                entity_data = self._entities_cache[
                    self._entities_cache["id"] == entity_id
                ].to_dict("records")
                
                if entity_data:
                    entity_dict = entity_data[0]
                    entity = Entity(
                        id=entity_dict.get("id", ""),
                        name=entity_dict.get("name", ""),
                        type=entity_dict.get("type", ""),
                        description=entity_dict.get("description", ""),
                        attributes=entity_dict.get("attributes", {}),
                        relationships=entity_dict.get("relationships", []),
                    )
                    entities.append(entity)
            
            logger.info(f"Found {len(entities)} entities for query")
            return entities
            
        except Exception as e:
            logger.error(f"Error searching entities: {e}", exc_info=True)
            return []
    
    async def expand_context(
        self,
        entities: List[Entity],
        max_hops: int = 2,
    ) -> List[Entity]:
        """
        Expand entity context by following relationships.
        
        Args:
            entities: Initial entities
            max_hops: Maximum relationship hops
            
        Returns:
            Expanded list of entities
        """
        if not entities or max_hops <= 0:
            return entities
        
        # Load relationships if not cached
        if self._relationships_cache is None:
            # Use default output_dir from config or fallback
            output_dir = "graphrag_output"
            self._relationships_cache = load_relationships_db(output_dir)
        
        # Track visited entities to avoid cycles
        visited = {e.id for e in entities}
        expanded = entities.copy()
        current_layer = entities.copy()
        
        for hop in range(max_hops):
            next_layer = []
            
            for entity in current_layer:
                # Find relationships for this entity
                related_df = self._relationships_cache[
                    (self._relationships_cache["source"] == entity.id) |
                    (self._relationships_cache["target"] == entity.id)
                ]
                
                for _, rel in related_df.iterrows():
                    # Get the other entity in the relationship
                    other_id = (
                        rel["target"] if rel["source"] == entity.id else rel["source"]
                    )
                    
                    if other_id not in visited:
                        visited.add(other_id)
                        
                        # Load entities if not cached
                        if self._entities_cache is None:
                            # Use default output_dir from config or fallback
                            output_dir = "graphrag_output"
                            self._entities_cache = load_entities_db(output_dir)
                        
                        # Find entity data
                        other_data = self._entities_cache[
                            self._entities_cache["id"] == other_id
                        ].to_dict("records")
                        
                        if other_data:
                            other_dict = other_data[0]
                            other_entity = Entity(
                                id=other_dict.get("id", ""),
                                name=other_dict.get("name", ""),
                                type=other_dict.get("type", ""),
                                description=other_dict.get("description", ""),
                                attributes=other_dict.get("attributes", {}),
                                relationships=other_dict.get("relationships", []),
                            )
                            next_layer.append(other_entity)
                            expanded.append(other_entity)
            
            current_layer = next_layer
            
            if not current_layer:
                break
        
        logger.info(f"Expanded from {len(entities)} to {len(expanded)} entities")
        return expanded
    
    async def get_text_units(
        self,
        entities: List[Entity],
        top_k: Optional[int] = None,
    ) -> List[TextUnit]:
        """
        Get text units associated with entities.
        
        Args:
            entities: Entities to get text units for
            top_k: Maximum number of text units
            
        Returns:
            List of text units
        """
        if not self.include_text_units or not self.main_store:
            return []
        
        top_k = top_k or self.text_unit_top_k
        text_units = []
        
        try:
            # Search for text units mentioning the entities
            entity_names = [e.name for e in entities[:5]]  # Limit to top 5 entities
            
            for name in entity_names:
                query_obj = VectorStoreQuery(
                    query_str=name,
                    mode=VectorStoreQueryMode.DEFAULT,
                    similarity_top_k=top_k // len(entity_names) + 1,
                )
                
                result = self.main_store.query(query_obj)
                
                for node in result.nodes:
                    text_unit = TextUnit(
                        id=node.node.node_id,
                        text=node.node.text,
                        chunk_id=node.node.metadata.get("chunk_id"),
                        document_id=node.node.metadata.get("document_id"),
                        entities=[name],
                    )
                    text_units.append(text_unit)
            
            # Deduplicate by ID
            seen_ids = set()
            unique_units = []
            for unit in text_units:
                if unit.id not in seen_ids:
                    seen_ids.add(unit.id)
                    unique_units.append(unit)
            
            # Limit to top_k
            unique_units = unique_units[:top_k]
            
            logger.info(f"Found {len(unique_units)} text units")
            return unique_units
            
        except Exception as e:
            logger.error(f"Error getting text units: {e}", exc_info=True)
            return []
    
    def clear_cache(self):
        """Clear cached entity and relationship data."""
        self._entities_cache = None
        self._relationships_cache = None
        logger.info("LocalSearcher cache cleared")