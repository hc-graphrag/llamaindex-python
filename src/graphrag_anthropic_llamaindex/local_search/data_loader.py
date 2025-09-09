"""Data loader for local search - loads entities and relationships from storage."""

import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from ..db_manager import load_entities_db, load_relationships_db
from .models import Entity, Relationship, TextUnit

logger = logging.getLogger(__name__)


def load_entities_from_parquet(output_dir: str) -> List[Entity]:
    """
    Load entities from Parquet file and convert to Entity objects.
    
    Args:
        output_dir: Directory containing the Parquet files
        
    Returns:
        List of Entity objects
    """
    try:
        # Load DataFrame from Parquet
        df = load_entities_db(output_dir)
        
        if df.empty:
            logger.warning(f"No entities found in {output_dir}")
            return []
        
        entities = []
        for _, row in df.iterrows():
            # Create Entity object from DataFrame row
            entity = Entity(
                id=str(row.get('id', row.name)) if 'id' in df.columns else str(row.name),
                name=row.get('name', ''),
                type=row.get('type') if pd.notna(row.get('type')) else None,
                description=row.get('description') if pd.notna(row.get('description')) else None,
                properties={}
            )
            
            # Add any additional columns as properties
            for col in df.columns:
                if col not in ['id', 'name', 'type', 'description']:
                    value = row.get(col)
                    if pd.notna(value):
                        entity.properties[col] = value
            
            entities.append(entity)
        
        logger.info(f"Loaded {len(entities)} entities from {output_dir}")
        return entities
        
    except Exception as e:
        logger.error(f"Error loading entities from {output_dir}: {e}")
        return []


def load_relationships_from_parquet(output_dir: str) -> List[Relationship]:
    """
    Load relationships from Parquet file and convert to Relationship objects.
    
    Args:
        output_dir: Directory containing the Parquet files
        
    Returns:
        List of Relationship objects
    """
    try:
        # Load DataFrame from Parquet
        df = load_relationships_db(output_dir)
        
        if df.empty:
            logger.warning(f"No relationships found in {output_dir}")
            return []
        
        relationships = []
        for idx, row in df.iterrows():
            # Create Relationship object from DataFrame row
            relationship = Relationship(
                id=str(row.get('id', idx)) if 'id' in df.columns else str(idx),
                source_id=str(row.get('source', '')),
                target_id=str(row.get('target', '')),
                type=str(row.get('type', 'RELATED')),
                description=row.get('description') if pd.notna(row.get('description')) else None,
                properties={},
                weight=float(row.get('weight', 1.0)) if 'weight' in df.columns else 1.0
            )
            
            # Add any additional columns as properties
            for col in df.columns:
                if col not in ['id', 'source', 'target', 'type', 'description', 'weight']:
                    value = row.get(col)
                    if pd.notna(value):
                        relationship.properties[col] = value
            
            relationships.append(relationship)
        
        logger.info(f"Loaded {len(relationships)} relationships from {output_dir}")
        return relationships
        
    except Exception as e:
        logger.error(f"Error loading relationships from {output_dir}: {e}")
        return []


def load_text_units_from_parquet(output_dir: str) -> List[TextUnit]:
    """
    Load text units from Parquet file if available.
    
    Args:
        output_dir: Directory containing the Parquet files
        
    Returns:
        List of TextUnit objects
    """
    try:
        # Check if text units file exists
        text_units_path = os.path.join(output_dir, 'text_units.parquet')
        if not os.path.exists(text_units_path):
            logger.info(f"No text units file found at {text_units_path}")
            return []
        
        # Load DataFrame from Parquet
        df = pd.read_parquet(text_units_path)
        
        if df.empty:
            logger.warning(f"No text units found in {output_dir}")
            return []
        
        text_units = []
        for idx, row in df.iterrows():
            # Create TextUnit object from DataFrame row
            text_unit = TextUnit(
                id=str(row.get('id', idx)) if 'id' in df.columns else str(idx),
                text=str(row.get('text', '')),
                entity_ids=[],
                metadata={}
            )
            
            # Parse entity IDs if available
            if 'entity_ids' in df.columns and pd.notna(row.get('entity_ids')):
                entity_ids = row.get('entity_ids')
                if isinstance(entity_ids, str):
                    # Assume comma-separated or similar format
                    text_unit.entity_ids = [e.strip() for e in entity_ids.split(',')]
                elif isinstance(entity_ids, list):
                    text_unit.entity_ids = entity_ids
            
            # Add any additional columns as metadata
            for col in df.columns:
                if col not in ['id', 'text', 'entity_ids']:
                    value = row.get(col)
                    if pd.notna(value):
                        text_unit.metadata[col] = value
            
            text_units.append(text_unit)
        
        logger.info(f"Loaded {len(text_units)} text units from {output_dir}")
        return text_units
        
    except Exception as e:
        logger.error(f"Error loading text units from {output_dir}: {e}")
        return []


def load_all_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load all data (entities, relationships, text units) from storage.
    
    Args:
        config: Configuration dictionary containing output_dir
        
    Returns:
        Dictionary containing loaded data
    """
    output_dir = config.get('output_dir', '.')
    
    data = {
        'entities': load_entities_from_parquet(output_dir),
        'relationships': load_relationships_from_parquet(output_dir),
        'text_units': load_text_units_from_parquet(output_dir)
    }
    
    logger.info(
        f"Loaded data summary: "
        f"{len(data['entities'])} entities, "
        f"{len(data['relationships'])} relationships, "
        f"{len(data['text_units'])} text units"
    )
    
    return data