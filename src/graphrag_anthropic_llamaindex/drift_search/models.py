"""Data models for DRIFT Search."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Entity:
    """Entity data model."""
    
    id: str
    name: str
    type: str
    description: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "attributes": self.attributes,
            "relationships": self.relationships,
        }


@dataclass
class Community:
    """Community data model."""
    
    id: str
    title: str
    summary: str
    entities: List[str] = field(default_factory=list)
    level: int = 0
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert community to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "entities": self.entities,
            "level": self.level,
        }


@dataclass
class TextUnit:
    """Text unit data model."""
    
    id: str
    text: str
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert text unit to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "entities": self.entities,
            "relationships": self.relationships,
        }


@dataclass
class SearchContext:
    """Search context data model."""
    
    query: str
    entities: List[Entity] = field(default_factory=list)
    communities: List[Community] = field(default_factory=list)
    text_units: List[TextUnit] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search context to dictionary."""
        return {
            "query": self.query,
            "entities": [e.to_dict() for e in self.entities],
            "communities": [c.to_dict() for c in self.communities],
            "text_units": [t.to_dict() for t in self.text_units],
            "metadata": self.metadata,
        }
    
    def get_token_count(self) -> int:
        """Estimate token count for the context."""
        # Rough estimation: 1 token per 4 characters
        total_chars = len(self.query)
        
        for entity in self.entities:
            total_chars += len(entity.name) + len(entity.description)
            
        for community in self.communities:
            total_chars += len(community.title) + len(community.summary)
            
        for text_unit in self.text_units:
            total_chars += len(text_unit.text)
            
        return total_chars // 4
    
    def trim_to_token_limit(self, max_tokens: int = 8000) -> "SearchContext":
        """Trim context to fit within token limit."""
        current_tokens = self.get_token_count()
        
        if current_tokens <= max_tokens:
            return self
        
        # Progressively remove items to fit within limit
        # Priority: text_units < entities < communities
        trimmed = SearchContext(
            query=self.query,
            entities=self.entities.copy(),
            communities=self.communities.copy(),
            text_units=self.text_units.copy(),
            metadata=self.metadata.copy(),
        )
        
        # Remove text units first
        while trimmed.text_units and trimmed.get_token_count() > max_tokens:
            trimmed.text_units.pop()
        
        # Then remove entities
        while trimmed.entities and trimmed.get_token_count() > max_tokens:
            trimmed.entities.pop()
        
        # Finally remove communities if needed
        while trimmed.communities and trimmed.get_token_count() > max_tokens:
            trimmed.communities.pop()
        
        return trimmed