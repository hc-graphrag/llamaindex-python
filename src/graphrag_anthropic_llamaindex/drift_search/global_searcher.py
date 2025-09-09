"""Global community search for DRIFT Search."""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from llama_index.core.vector_stores.types import VectorStore, VectorStoreQuery, VectorStoreQueryMode

from ..db_manager import load_community_summaries_db
from .models import Community

logger = logging.getLogger(__name__)


class GlobalSearcher:
    """Global community search component for DRIFT search."""
    
    def __init__(
        self,
        vector_stores: Dict[str, VectorStore],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize global searcher.
        
        Args:
            vector_stores: Mapping of vector store names to instances
            config: Global search configuration
        """
        self.community_store = vector_stores.get("community")
        self.config = config or {}
        
        # Configuration
        self.community_top_k = self.config.get("community_top_k", 5)
        self.include_summaries = self.config.get("include_summaries", True)
        self.max_summary_length = self.config.get("max_summary_length", 500)
        
        # Cache for communities
        self._communities_cache = None
        
        logger.info(f"GlobalSearcher initialized with top_k={self.community_top_k}")
    
    async def search_communities(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Community]:
        """
        Search for relevant communities.
        
        Args:
            query: Search query
            top_k: Number of top communities to return
            
        Returns:
            List of relevant communities
        """
        if not self.community_store:
            logger.warning("Community vector store not available")
            return []
        
        top_k = top_k or self.community_top_k
        
        try:
            # Create vector store query
            query_obj = VectorStoreQuery(
                query_str=query,
                mode=VectorStoreQueryMode.DEFAULT,
                similarity_top_k=top_k,
            )
            
            # Search community store
            result = self.community_store.query(query_obj)
            
            # Load community data if not cached
            if self._communities_cache is None:
                import os
                output_dir = os.environ.get("GRAPHRAG_OUTPUT_DIR", "graphrag_output")
                self._communities_cache = load_community_summaries_db(output_dir)
            
            # Convert results to Community objects
            communities = []
            for node in result.nodes:
                # Extract community ID from node
                community_id = node.node.node_id
                
                # Find community in cache
                community_data = self._communities_cache[
                    self._communities_cache["id"] == community_id
                ].to_dict("records")
                
                if community_data:
                    community_dict = community_data[0]
                    
                    # Truncate summary if needed
                    summary = community_dict.get("summary", "")
                    if self.max_summary_length and len(summary) > self.max_summary_length:
                        summary = summary[:self.max_summary_length] + "..."
                    
                    community = Community(
                        id=community_dict.get("id", ""),
                        title=community_dict.get("title", ""),
                        summary=summary if self.include_summaries else "",
                        entities=community_dict.get("entities", []),
                        level=community_dict.get("level", 0),
                    )
                    communities.append(community)
            
            logger.info(f"Found {len(communities)} communities for query")
            return communities
            
        except Exception as e:
            logger.error(f"Error searching communities: {e}", exc_info=True)
            return []
    
    async def get_community_summaries(
        self,
        communities: List[Community],
    ) -> List[str]:
        """
        Get summaries for communities.
        
        Args:
            communities: Communities to get summaries for
            
        Returns:
            List of community summaries
        """
        summaries = []
        
        for community in communities:
            if community.summary:
                summary = f"Community: {community.title}\n{community.summary}"
                summaries.append(summary)
        
        return summaries
    
    async def get_hierarchical_communities(
        self,
        communities: List[Community],
    ) -> Dict[int, List[Community]]:
        """
        Organize communities by hierarchy level.
        
        Args:
            communities: Communities to organize
            
        Returns:
            Dictionary mapping level to communities
        """
        hierarchy = {}
        
        for community in communities:
            level = community.level
            if level not in hierarchy:
                hierarchy[level] = []
            hierarchy[level].append(community)
        
        # Sort by level
        sorted_hierarchy = dict(sorted(hierarchy.items()))
        
        logger.info(f"Organized {len(communities)} communities into {len(sorted_hierarchy)} levels")
        return sorted_hierarchy
    
    async def filter_by_entities(
        self,
        communities: List[Community],
        entity_ids: List[str],
    ) -> List[Community]:
        """
        Filter communities by entity membership.
        
        Args:
            communities: Communities to filter
            entity_ids: Entity IDs to filter by
            
        Returns:
            Filtered communities
        """
        entity_set = set(entity_ids)
        filtered = []
        
        for community in communities:
            # Check if any community entities match
            if any(e in entity_set for e in community.entities):
                filtered.append(community)
        
        logger.info(f"Filtered to {len(filtered)} communities containing specified entities")
        return filtered
    
    def clear_cache(self):
        """Clear cached community data."""
        self._communities_cache = None
        logger.info("GlobalSearcher cache cleared")