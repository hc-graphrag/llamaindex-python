"""Data models for local search functionality."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    
    id: str
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __str__(self) -> str:
        """String representation of the entity."""
        return f"{self.name} ({self.type or 'Unknown'})"


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    
    id: str
    source_id: str
    target_id: str
    type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    def __str__(self) -> str:
        """String representation of the relationship."""
        return f"{self.source_id} --[{self.type}]--> {self.target_id}"


@dataclass
class TextUnit:
    """Represents a text unit containing entities."""
    
    id: str
    text: str
    entity_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __str__(self) -> str:
        """String representation of the text unit."""
        preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
        return f"TextUnit({self.id}): {preview}"


@dataclass
class ContextResult:
    """Result from context building for local search."""
    
    query: str
    entities: List[Entity]
    relationships: List[Relationship]
    text_units: List[TextUnit] = field(default_factory=list)
    context_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        """String representation of the context result."""
        return (f"ContextResult(entities={len(self.entities)}, "
                f"relationships={len(self.relationships)}, "
                f"text_units={len(self.text_units)})")