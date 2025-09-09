"""Context builder for local search - constructs search context from entities and relationships."""

from typing import List, Optional, Dict, Any
import logging
from .models import Entity, Relationship, TextUnit, ContextResult

logger = logging.getLogger(__name__)


class LocalContextBuilder:
    """Builds context for local search from entities and relationships."""
    
    def __init__(
        self,
        max_context_tokens: int = 4000,
        include_entity_descriptions: bool = True,
        include_relationship_descriptions: bool = True,
        format_style: str = "structured"  # "structured" or "narrative"
    ):
        """
        Initialize the LocalContextBuilder.
        
        Args:
            max_context_tokens: Maximum tokens for context (approximate)
            include_entity_descriptions: Whether to include entity descriptions
            include_relationship_descriptions: Whether to include relationship descriptions
            format_style: Format style for context ("structured" or "narrative")
        """
        self.max_context_tokens = max_context_tokens
        self.include_entity_descriptions = include_entity_descriptions
        self.include_relationship_descriptions = include_relationship_descriptions
        self.format_style = format_style
        
        # Rough token estimation: 1 token ≈ 4 characters
        self.max_context_chars = max_context_tokens * 4
    
    def build_context(
        self,
        query: str,
        entities: List[Entity],
        relationships: List[Relationship],
        text_units: Optional[List[TextUnit]] = None
    ) -> ContextResult:
        """
        Build context from entities and relationships.
        
        Args:
            query: The original search query
            entities: List of relevant entities
            relationships: List of relevant relationships
            text_units: Optional list of text units
            
        Returns:
            ContextResult containing the formatted context
        """
        text_units = text_units or []
        
        if self.format_style == "structured":
            context_text = self._build_structured_context(entities, relationships, text_units)
        else:
            context_text = self._build_narrative_context(entities, relationships, text_units)
        
        # Truncate if necessary
        if len(context_text) > self.max_context_chars:
            context_text = self._truncate_context(context_text)
        
        return ContextResult(
            query=query,
            entities=entities,
            relationships=relationships,
            text_units=text_units,
            context_text=context_text,
            metadata={
                "num_entities": len(entities),
                "num_relationships": len(relationships),
                "num_text_units": len(text_units),
                "format_style": self.format_style
            }
        )
    
    def _build_structured_context(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        text_units: List[TextUnit]
    ) -> str:
        """Build context in structured format."""
        sections = []
        
        # Add entities section
        if entities:
            entity_lines = ["## Entities:"]
            for entity in entities:
                entity_line = f"- **{entity.name}**"
                if entity.type:
                    entity_line += f" ({entity.type})"
                if self.include_entity_descriptions and entity.description:
                    entity_line += f": {entity.description[:200]}"
                entity_lines.append(entity_line)
            sections.append("\n".join(entity_lines))
        
        # Add relationships section
        if relationships:
            rel_lines = ["## Relationships:"]
            for rel in relationships:
                rel_line = f"- {rel.source_id} --[{rel.type}]--> {rel.target_id}"
                if self.include_relationship_descriptions and rel.description:
                    rel_line += f": {rel.description[:150]}"
                rel_lines.append(rel_line)
            sections.append("\n".join(rel_lines))
        
        # Add text units section if available
        if text_units:
            text_lines = ["## Related Text:"]
            for unit in text_units[:5]:  # Limit to first 5 text units
                text_preview = unit.text[:300] + "..." if len(unit.text) > 300 else unit.text
                text_lines.append(f"- {text_preview}")
            sections.append("\n".join(text_lines))
        
        return "\n\n".join(sections)
    
    def _build_narrative_context(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        text_units: List[TextUnit]
    ) -> str:
        """Build context in narrative format."""
        paragraphs = []
        
        # Create entity map for easier lookup
        entity_map = {e.id: e for e in entities}
        
        # Group relationships by source entity
        rel_by_source: Dict[str, List[Relationship]] = {}
        for rel in relationships:
            if rel.source_id not in rel_by_source:
                rel_by_source[rel.source_id] = []
            rel_by_source[rel.source_id].append(rel)
        
        # Build narrative paragraphs
        for entity in entities:
            para_parts = [f"{entity.name}"]
            
            if entity.type:
                para_parts[0] += f" (a {entity.type})"
            
            if self.include_entity_descriptions and entity.description:
                para_parts.append(entity.description[:200])
            
            # Add relationships for this entity
            if entity.id in rel_by_source:
                rel_descriptions = []
                for rel in rel_by_source[entity.id][:3]:  # Limit to 3 relationships per entity
                    target_name = entity_map.get(rel.target_id, {}).get("name", rel.target_id)
                    rel_desc = f"{rel.type} {target_name}"
                    if self.include_relationship_descriptions and rel.description:
                        rel_desc += f" ({rel.description[:50]})"
                    rel_descriptions.append(rel_desc)
                
                if rel_descriptions:
                    para_parts.append(f"This entity {', '.join(rel_descriptions)}.")
            
            paragraphs.append(" ".join(para_parts))
        
        # Add text units as additional context
        if text_units:
            paragraphs.append("\nAdditional context:")
            for unit in text_units[:3]:  # Limit to first 3 text units
                text_preview = unit.text[:200] + "..." if len(unit.text) > 200 else unit.text
                paragraphs.append(text_preview)
        
        return "\n\n".join(paragraphs)
    
    def _truncate_context(self, context: str) -> str:
        """Truncate context to fit within token limits."""
        if len(context) <= self.max_context_chars:
            return context
        
        # Truncate and add ellipsis
        truncated = context[:self.max_context_chars - 20]
        
        # Try to truncate at a sentence boundary
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        truncate_point = max(last_period, last_newline)
        if truncate_point > self.max_context_chars * 0.8:  # If we found a good breaking point
            truncated = truncated[:truncate_point + 1]
        
        return truncated + "\n\n[Context truncated...]"