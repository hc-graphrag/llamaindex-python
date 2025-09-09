"""Tests for local search functionality."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from llama_index.core.schema import QueryBundle, NodeWithScore

from src.graphrag_anthropic_llamaindex.local_search.models import (
    Entity, Relationship, TextUnit, ContextResult
)
from src.graphrag_anthropic_llamaindex.local_search.entity_mapper import EntityMapper
from src.graphrag_anthropic_llamaindex.local_search.context_builder import LocalContextBuilder
from src.graphrag_anthropic_llamaindex.local_search.retriever import LocalSearchRetriever
from src.graphrag_anthropic_llamaindex.local_search.data_loader import (
    load_entities_from_parquet,
    load_relationships_from_parquet
)


class TestDataModels:
    """Test data model classes."""
    
    def test_entity_creation(self):
        """Test Entity model creation."""
        entity = Entity(
            id="1",
            name="Test Entity",
            type="Person",
            description="A test entity",
            properties={"age": 30}
        )
        
        assert entity.id == "1"
        assert entity.name == "Test Entity"
        assert entity.type == "Person"
        assert entity.description == "A test entity"
        assert entity.properties["age"] == 30
        assert str(entity) == "Test Entity (Person)"
    
    def test_relationship_creation(self):
        """Test Relationship model creation."""
        rel = Relationship(
            id="r1",
            source_id="1",
            target_id="2",
            type="KNOWS",
            description="Person 1 knows Person 2",
            weight=0.8
        )
        
        assert rel.id == "r1"
        assert rel.source_id == "1"
        assert rel.target_id == "2"
        assert rel.type == "KNOWS"
        assert rel.weight == 0.8
        assert str(rel) == "1 --[KNOWS]--> 2"
    
    def test_context_result_creation(self):
        """Test ContextResult model creation."""
        entities = [Entity(id="1", name="Entity1")]
        relationships = [Relationship(id="r1", source_id="1", target_id="2", type="REL")]
        
        result = ContextResult(
            query="test query",
            entities=entities,
            relationships=relationships,
            context_text="Test context"
        )
        
        assert result.query == "test query"
        assert len(result.entities) == 1
        assert len(result.relationships) == 1
        assert result.context_text == "Test context"


class TestEntityMapper:
    """Test EntityMapper class."""
    
    def test_entity_mapper_initialization(self):
        """Test EntityMapper initialization."""
        config = {"output_dir": "."}
        mapper = EntityMapper(config=config, top_k=5)
        
        assert mapper.config == config
        assert mapper.top_k == 5
        assert mapper.entity_index is None  # No vector store configured
    
    @patch('src.graphrag_anthropic_llamaindex.local_search.entity_mapper.get_vector_store')
    @patch('src.graphrag_anthropic_llamaindex.local_search.entity_mapper.get_index')
    def test_map_query_to_entities(self, mock_get_index, mock_get_vector_store):
        """Test mapping query to entities."""
        # Setup mocks
        mock_vector_store = Mock()
        mock_get_vector_store.return_value = mock_vector_store
        
        mock_index = Mock()
        mock_retriever = Mock()
        mock_node = Mock()
        mock_node.node_id = "1"
        mock_node.text = "Entity description"
        mock_node.metadata = {
            "id": "1",
            "name": "Test Entity",
            "type": "Person"
        }
        
        mock_retriever.retrieve.return_value = [mock_node]
        mock_index.as_retriever.return_value = mock_retriever
        mock_get_index.return_value = mock_index
        
        # Test
        config = {"output_dir": "."}
        mapper = EntityMapper(config=config)
        
        entities = mapper.map_query_to_entities("test query", top_k=1)
        
        assert len(entities) == 1
        assert entities[0].id == "1"
        assert entities[0].name == "Test Entity"
        assert entities[0].type == "Person"
    
    def test_map_query_to_entities_no_index(self):
        """Test mapping with no index returns empty list."""
        config = {"output_dir": "."}
        mapper = EntityMapper(config=config)
        mapper.entity_index = None
        
        entities = mapper.map_query_to_entities("test query")
        
        assert entities == []


class TestLocalContextBuilder:
    """Test LocalContextBuilder class."""
    
    def test_context_builder_initialization(self):
        """Test LocalContextBuilder initialization."""
        builder = LocalContextBuilder(
            max_context_tokens=2000,
            include_entity_descriptions=True,
            format_style="structured"
        )
        
        assert builder.max_context_tokens == 2000
        assert builder.include_entity_descriptions is True
        assert builder.format_style == "structured"
    
    def test_build_structured_context(self):
        """Test building structured context."""
        builder = LocalContextBuilder(format_style="structured")
        
        entities = [
            Entity(id="1", name="Alice", type="Person", description="A person"),
            Entity(id="2", name="Bob", type="Person", description="Another person")
        ]
        
        relationships = [
            Relationship(id="r1", source_id="1", target_id="2", type="KNOWS")
        ]
        
        result = builder.build_context(
            query="Who knows whom?",
            entities=entities,
            relationships=relationships
        )
        
        assert result.query == "Who knows whom?"
        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert "## Entities:" in result.context_text
        assert "Alice" in result.context_text
        assert "Bob" in result.context_text
        assert "## Relationships:" in result.context_text
        assert "KNOWS" in result.context_text
    
    def test_build_narrative_context(self):
        """Test building narrative context."""
        builder = LocalContextBuilder(format_style="narrative")
        
        entities = [
            Entity(id="1", name="Alice", type="Person", description="A software engineer")
        ]
        
        relationships = [
            Relationship(id="r1", source_id="1", target_id="2", type="WORKS_WITH")
        ]
        
        result = builder.build_context(
            query="Tell me about Alice",
            entities=entities,
            relationships=relationships
        )
        
        assert "Alice (a Person)" in result.context_text
        assert "software engineer" in result.context_text
    
    def test_context_truncation(self):
        """Test context truncation when exceeding max tokens."""
        builder = LocalContextBuilder(max_context_tokens=10)  # Very small limit
        
        entities = [
            Entity(id=str(i), name=f"Entity{i}", description="Long description " * 100)
            for i in range(10)
        ]
        
        result = builder.build_context(
            query="test",
            entities=entities,
            relationships=[]
        )
        
        # Context should be truncated (10 tokens * 4 chars = 40 chars max)
        assert len(result.context_text) <= 50  # Some buffer for truncation message
        assert "[Context truncated...]" in result.context_text


class TestLocalSearchRetriever:
    """Test LocalSearchRetriever class."""
    
    def test_retriever_initialization(self):
        """Test LocalSearchRetriever initialization."""
        config = {"output_dir": ".", "llm": {"model": "test"}}
        retriever = LocalSearchRetriever(
            config=config,
            prompt_style="default",
            top_k_entities=5
        )
        
        assert retriever.config == config
        assert retriever.prompt_style == "default"
        assert retriever.top_k_entities == 5
    
    @patch('src.graphrag_anthropic_llamaindex.local_search.retriever.EntityMapper')
    @patch('src.graphrag_anthropic_llamaindex.local_search.retriever.LocalContextBuilder')
    def test_retrieve_no_entities(self, mock_builder_class, mock_mapper_class):
        """Test retrieval when no entities are found."""
        # Setup mocks
        mock_mapper = Mock()
        mock_mapper.map_query_to_entities.return_value = []
        mock_mapper_class.return_value = mock_mapper
        
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        
        # Test
        config = {"output_dir": "."}
        retriever = LocalSearchRetriever(config=config)
        
        query_bundle = QueryBundle(query_str="test query")
        results = retriever._retrieve(query_bundle)
        
        assert len(results) == 1
        assert results[0].score == 0.0
        assert "No relevant information found" in results[0].node.text
    
    @patch('src.graphrag_anthropic_llamaindex.local_search.retriever.EntityMapper')
    @patch('src.graphrag_anthropic_llamaindex.local_search.retriever.LocalContextBuilder')
    def test_retrieve_with_entities_no_llm(self, mock_builder_class, mock_mapper_class):
        """Test retrieval with entities but no LLM."""
        # Setup mocks
        mock_mapper = Mock()
        mock_mapper.map_query_to_entities.return_value = [
            Entity(id="1", name="Test Entity")
        ]
        mock_mapper_class.return_value = mock_mapper
        
        mock_builder = Mock()
        mock_context_result = Mock()
        mock_context_result.context_text = "Test context"
        mock_context_result.query = "test query"
        mock_context_result.entities = [Entity(id="1", name="Test Entity")]
        mock_context_result.relationships = []
        mock_builder.build_context.return_value = mock_context_result
        mock_builder_class.return_value = mock_builder
        
        # Test
        config = {"output_dir": "."}
        retriever = LocalSearchRetriever(config=config, llm=None)
        
        query_bundle = QueryBundle(query_str="test query")
        results = retriever._retrieve(query_bundle)
        
        assert len(results) == 1
        assert results[0].score == 0.8
        assert results[0].node.text == "Test context"


class TestDataLoader:
    """Test data loader functions."""
    
    @patch('src.graphrag_anthropic_llamaindex.local_search.data_loader.load_entities_db')
    def test_load_entities_from_parquet(self, mock_load_db):
        """Test loading entities from Parquet."""
        import pandas as pd
        
        # Mock DataFrame
        df = pd.DataFrame({
            'id': ['1', '2'],
            'name': ['Alice', 'Bob'],
            'type': ['Person', 'Person'],
            'description': ['Desc1', 'Desc2']
        })
        mock_load_db.return_value = df
        
        entities = load_entities_from_parquet(".")
        
        assert len(entities) == 2
        assert entities[0].id == '1'
        assert entities[0].name == 'Alice'
        assert entities[1].id == '2'
        assert entities[1].name == 'Bob'
    
    @patch('src.graphrag_anthropic_llamaindex.local_search.data_loader.load_relationships_db')
    def test_load_relationships_from_parquet(self, mock_load_db):
        """Test loading relationships from Parquet."""
        import pandas as pd
        
        # Mock DataFrame
        df = pd.DataFrame({
            'id': ['r1', 'r2'],
            'source': ['1', '2'],
            'target': ['2', '3'],
            'type': ['KNOWS', 'WORKS_WITH'],
            'description': ['Desc1', 'Desc2']
        })
        mock_load_db.return_value = df
        
        relationships = load_relationships_from_parquet(".")
        
        assert len(relationships) == 2
        assert relationships[0].id == 'r1'
        assert relationships[0].source_id == '1'
        assert relationships[0].target_id == '2'
        assert relationships[0].type == 'KNOWS'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])