"""Context builder for DRIFT Search."""

import logging
from typing import Any, Dict, List, Optional

from .models import Community, Entity, SearchContext, TextUnit

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build and manage search context for DRIFT search."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize context builder.
        
        Args:
            config: Context configuration
        """
        self.config = config or {}
        
        # Configuration
        self.max_tokens = self.config.get("max_tokens", 8000)
        self.prioritization_strategy = self.config.get("prioritization_strategy", "relevance")
        self.include_metadata = self.config.get("include_metadata", True)
        
        logger.info(f"ContextBuilder initialized with max_tokens={self.max_tokens}")
    
    def build_search_context(
        self,
        query: str,
        local_results: List[Entity],
        global_results: List[Community],
        text_units: Optional[List[TextUnit]] = None,
    ) -> SearchContext:
        """
        Build search context from local and global results.
        
        Args:
            query: Original search query
            local_results: Entities from local search
            global_results: Communities from global search
            text_units: Optional text units
            
        Returns:
            SearchContext object
        """
        # Create metadata
        metadata = {}
        if self.include_metadata:
            metadata = {
                "query": query,
                "num_entities": len(local_results),
                "num_communities": len(global_results),
                "num_text_units": len(text_units) if text_units else 0,
                "prioritization_strategy": self.prioritization_strategy,
                "max_tokens": self.max_tokens,
            }
        
        # Create context
        context = SearchContext(
            query=query,
            entities=local_results,
            communities=global_results,
            text_units=text_units or [],
            metadata=metadata,
        )
        
        # Prioritize context based on strategy
        context = self.prioritize_context(context)
        
        logger.info(
            f"Built context with {len(context.entities)} entities, "
            f"{len(context.communities)} communities, "
            f"{len(context.text_units)} text units"
        )
        
        return context
    
    def prioritize_context(
        self,
        context: SearchContext,
    ) -> SearchContext:
        """
        Prioritize context elements based on strategy.
        
        Args:
            context: Search context to prioritize
            
        Returns:
            Prioritized search context
        """
        if self.prioritization_strategy == "relevance":
            # Keep original order (assumed to be by relevance score)
            return context
            
        elif self.prioritization_strategy == "recency":
            # Sort by recency if available in metadata
            # This would require timestamp metadata
            return context
            
        elif self.prioritization_strategy == "mixed":
            # Mixed strategy: interleave entities and communities
            return self._mixed_prioritization(context)
        
        else:
            logger.warning(f"Unknown prioritization strategy: {self.prioritization_strategy}")
            return context
    
    def _mixed_prioritization(self, context: SearchContext) -> SearchContext:
        """
        Apply mixed prioritization strategy.
        
        Args:
            context: Search context
            
        Returns:
            Prioritized context
        """
        # Interleave entities and communities
        max_entities = min(5, len(context.entities))
        max_communities = min(3, len(context.communities))
        
        prioritized = SearchContext(
            query=context.query,
            entities=context.entities[:max_entities],
            communities=context.communities[:max_communities],
            text_units=context.text_units[:5] if context.text_units else [],
            metadata=context.metadata,
        )
        
        return prioritized
    
    def merge_contexts(
        self,
        contexts: List[SearchContext],
    ) -> SearchContext:
        """
        Merge multiple search contexts.
        
        Args:
            contexts: List of search contexts to merge
            
        Returns:
            Merged search context
        """
        if not contexts:
            return SearchContext(query="")
        
        # Use first context as base
        merged = contexts[0]
        
        # Merge entities, communities, and text units
        for ctx in contexts[1:]:
            # Merge entities (avoid duplicates by ID)
            existing_entity_ids = {e.id for e in merged.entities}
            for entity in ctx.entities:
                if entity.id not in existing_entity_ids:
                    merged.entities.append(entity)
                    existing_entity_ids.add(entity.id)
            
            # Merge communities (avoid duplicates by ID)
            existing_community_ids = {c.id for c in merged.communities}
            for community in ctx.communities:
                if community.id not in existing_community_ids:
                    merged.communities.append(community)
                    existing_community_ids.add(community.id)
            
            # Merge text units (avoid duplicates by ID)
            existing_text_unit_ids = {t.id for t in merged.text_units}
            for text_unit in ctx.text_units:
                if text_unit.id not in existing_text_unit_ids:
                    merged.text_units.append(text_unit)
                    existing_text_unit_ids.add(text_unit.id)
            
            # Merge metadata
            merged.metadata.update(ctx.metadata)
        
        # Apply token limit
        merged = merged.trim_to_token_limit(self.max_tokens)
        
        logger.info(f"Merged {len(contexts)} contexts")
        return merged
    
    def format_context_for_prompt(
        self,
        context: SearchContext,
    ) -> str:
        """
        Format context for LLM prompt.
        
        Args:
            context: Search context
            
        Returns:
            Formatted context string
        """
        sections = []
        
        # Add entities section
        if context.entities:
            entity_texts = []
            for entity in context.entities[:10]:  # Limit to top 10
                entity_text = f"- {entity.name} ({entity.type}): {entity.description}"
                if entity.relationships:
                    rel_count = len(entity.relationships)
                    entity_text += f" [Имеет {rel_count} связей]"
                entity_texts.append(entity_text)
            
            sections.append("## Соответствующие сущности:\n" + "\n".join(entity_texts))
        
        # Add communities section
        if context.communities:
            community_texts = []
            for community in context.communities[:5]:  # Limit to top 5
                community_text = f"- {community.title}: {community.summary}"
                if community.entities:
                    entity_count = len(community.entities)
                    community_text += f" [Содержит {entity_count} сущностей]"
                community_texts.append(community_text)
            
            sections.append("## Сообщества знаний:\n" + "\n".join(community_texts))
        
        # Add text units section
        if context.text_units:
            text_unit_texts = []
            for unit in context.text_units[:5]:  # Limit to top 5
                text_unit_texts.append(f"- {unit.text[:200]}...")
            
            sections.append("## Текстовые фрагменты:\n" + "\n".join(text_unit_texts))
        
        return "\n\n".join(sections)
    
    def extract_key_information(
        self,
        context: SearchContext,
    ) -> Dict[str, Any]:
        """
        Extract key information from context.
        
        Args:
            context: Search context
            
        Returns:
            Dictionary with key information
        """
        key_info = {
            "main_entities": [],
            "main_communities": [],
            "key_relationships": [],
            "summary_points": [],
        }
        
        # Extract main entities
        for entity in context.entities[:5]:
            key_info["main_entities"].append({
                "name": entity.name,
                "type": entity.type,
                "description": entity.description[:100],
            })
        
        # Extract main communities  
        for community in context.communities[:3]:
            key_info["main_communities"].append({
                "title": community.title,
                "summary": community.summary[:200],
                "size": len(community.entities),
            })
        
        # Extract key relationships
        for entity in context.entities[:3]:
            if entity.relationships:
                for rel in entity.relationships[:2]:
                    key_info["key_relationships"].append(rel)
        
        # Generate summary points
        if context.entities:
            key_info["summary_points"].append(
                f"Найдено {len(context.entities)} связанных сущностей"
            )
        if context.communities:
            key_info["summary_points"].append(
                f"Идентифицировано {len(context.communities)} сообществ знаний"
            )
        
        return key_info