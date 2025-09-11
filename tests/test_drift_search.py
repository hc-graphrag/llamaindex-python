"""Tests for DRIFT Search functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from graphrag_anthropic_llamaindex.drift_search import (
    DriftSearchEngine,
    LocalSearcher,
    GlobalSearcher,
    ContextBuilder,
    ResponseGenerator,
    Entity,
    Community,
    SearchContext,
    TextUnit,
)


class TestDriftSearchEngine:
    """Test DriftSearchEngine class."""
    
    @pytest.fixture
    def mock_vector_stores(self):
        """Create mock vector stores."""
        return {
            "main": MagicMock(),
            "entity": MagicMock(),
            "community": MagicMock(),
        }
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "drift_search": {
                "local_search": {
                    "entity_top_k": 5,
                    "relationship_depth": 1,
                },
                "global_search": {
                    "community_top_k": 3,
                },
                "context": {
                    "max_tokens": 1000,
                },
                "response": {
                    "max_tokens": 500,
                },
            }
        }
    
    @pytest.mark.asyncio
    async def test_drift_search_basic(self, mock_vector_stores, mock_config):
        """Test basic DRIFT search functionality."""
        # Create engine
        engine = DriftSearchEngine(
            config=mock_config,
            vector_stores=mock_vector_stores,
            llm=MagicMock(),
        )
        
        # Mock searcher results
        with patch.object(engine.local_searcher, "search_entities") as mock_local:
            mock_local.return_value = [
                Entity("1", "Entity1", "Type1", "Description1"),
                Entity("2", "Entity2", "Type2", "Description2"),
            ]
            
            with patch.object(engine.global_searcher, "search_communities") as mock_global:
                mock_global.return_value = [
                    Community("c1", "Community1", "Summary1"),
                ]
                
                with patch.object(engine.response_generator, "generate_response") as mock_gen:
                    mock_gen.return_value = "Test response"
                    
                    # Execute search
                    result = await engine.search("test query", streaming=False, include_context=False)
                    
                    assert result == "Test response"
                    mock_local.assert_called_once()
                    mock_global.assert_called_once()
                    mock_gen.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_drift_search_with_context(self, mock_vector_stores, mock_config):
        """Test DRIFT search with context."""
        engine = DriftSearchEngine(
            config=mock_config,
            vector_stores=mock_vector_stores,
            llm=MagicMock(),
        )
        
        # Mock results
        with patch.object(engine.local_searcher, "search_entities") as mock_local:
            mock_local.return_value = [Entity("1", "Entity1", "Type1", "Description1")]
            
            with patch.object(engine.global_searcher, "search_communities") as mock_global:
                mock_global.return_value = [Community("c1", "Community1", "Summary1")]
                
                with patch.object(engine.response_generator, "generate_response") as mock_gen:
                    mock_gen.return_value = "Test response with context"
                    
                    # Execute search with context
                    result, context = await engine.search(
                        "test query", 
                        streaming=False,
                        include_context=True
                    )
                    
                    assert result == "Test response with context"
                    assert isinstance(context, dict)
                    assert "entities" in context
                    assert "communities" in context


class TestLocalSearcher:
    """Test LocalSearcher class."""
    
    @pytest.fixture
    def mock_vector_stores(self):
        """Create mock vector stores."""
        return {
            "entity": MagicMock(),
            "main": MagicMock(),
        }
    
    @pytest.mark.asyncio
    async def test_search_entities(self, mock_vector_stores):
        """Test entity search."""
        # No longer using GRAPHRAG_OUTPUT_DIR environment variable
        
        searcher = LocalSearcher(mock_vector_stores)
        
        # Mock vector store query result
        mock_result = MagicMock()
        mock_node = MagicMock()
        mock_node.node.node_id = "entity_1"
        mock_result.nodes = [mock_node]
        
        mock_vector_stores["entity"].query.return_value = mock_result
        
        # Mock entity data
        with patch("graphrag_anthropic_llamaindex.drift_search.local_searcher.load_entities_db") as mock_load:
            import pandas as pd
            mock_load.return_value = pd.DataFrame([
                {
                    "id": "entity_1",
                    "name": "Test Entity",
                    "type": "TestType",
                    "description": "Test Description",
                    "attributes": {},
                    "relationships": [],
                }
            ])
            
            # Execute search
            results = await searcher.search_entities("test query")
            
            assert len(results) == 1
            assert results[0].name == "Test Entity"
            assert results[0].type == "TestType"
    
    @pytest.mark.asyncio
    async def test_expand_context(self, mock_vector_stores):
        """Test context expansion."""
        # No longer using GRAPHRAG_OUTPUT_DIR environment variable
        
        searcher = LocalSearcher(mock_vector_stores)
        
        # Initial entities
        entities = [
            Entity("1", "Entity1", "Type1", "Description1"),
        ]
        
        # Mock relationships
        with patch("graphrag_anthropic_llamaindex.drift_search.local_searcher.load_relationships_db") as mock_rel:
            import pandas as pd
            mock_rel.return_value = pd.DataFrame([
                {"source": "1", "target": "2", "type": "related"},
            ])
            
            # Mock entities
            with patch("graphrag_anthropic_llamaindex.drift_search.local_searcher.load_entities_db") as mock_ent:
                mock_ent.return_value = pd.DataFrame([
                    {
                        "id": "2",
                        "name": "Entity2",
                        "type": "Type2",
                        "description": "Description2",
                        "attributes": {},
                        "relationships": [],
                    }
                ])
                
                # Expand context
                expanded = await searcher.expand_context(entities, max_hops=1)
                
                assert len(expanded) == 2
                assert any(e.id == "2" for e in expanded)


class TestGlobalSearcher:
    """Test GlobalSearcher class."""
    
    @pytest.fixture
    def mock_vector_stores(self):
        """Create mock vector stores."""
        return {
            "community": MagicMock(),
        }
    
    @pytest.mark.asyncio
    async def test_search_communities(self, mock_vector_stores):
        """Test community search."""
        # No longer using GRAPHRAG_OUTPUT_DIR environment variable
        
        searcher = GlobalSearcher(mock_vector_stores)
        
        # Mock vector store query result
        mock_result = MagicMock()
        mock_node = MagicMock()
        mock_node.node.node_id = "comm_1"
        mock_result.nodes = [mock_node]
        
        mock_vector_stores["community"].query.return_value = mock_result
        
        # Mock community data
        with patch("graphrag_anthropic_llamaindex.drift_search.global_searcher.load_community_summaries_db") as mock_load:
            import pandas as pd
            mock_load.return_value = pd.DataFrame([
                {
                    "id": "comm_1",
                    "title": "Test Community",
                    "summary": "Test Summary",
                    "entities": ["e1", "e2"],
                    "level": 0,
                }
            ])
            
            # Execute search
            results = await searcher.search_communities("test query")
            
            assert len(results) == 1
            assert results[0].title == "Test Community"
            assert results[0].summary == "Test Summary"
            assert len(results[0].entities) == 2


class TestContextBuilder:
    """Test ContextBuilder class."""
    
    def test_build_search_context(self):
        """Test context building."""
        builder = ContextBuilder()
        
        # Create test data
        entities = [
            Entity("1", "Entity1", "Type1", "Description1"),
            Entity("2", "Entity2", "Type2", "Description2"),
        ]
        communities = [
            Community("c1", "Community1", "Summary1"),
        ]
        
        # Build context
        context = builder.build_search_context(
            query="test query",
            local_results=entities,
            global_results=communities,
        )
        
        assert context.query == "test query"
        assert len(context.entities) == 2
        assert len(context.communities) == 1
        assert isinstance(context.metadata, dict)
    
    def test_format_context_for_prompt(self):
        """Test context formatting."""
        builder = ContextBuilder()
        
        # Create context
        context = SearchContext(
            query="test query",
            entities=[Entity("1", "Entity1", "Type1", "Description1")],
            communities=[Community("c1", "Community1", "Summary1")],
        )
        
        # Format context
        formatted = builder.format_context_for_prompt(context)
        
        assert "Entity1" in formatted
        assert "Community1" in formatted
        assert "Summary1" in formatted
    
    def test_context_trimming(self):
        """Test context trimming to token limit."""
        # Create large context
        entities = [
            Entity(f"e{i}", f"Entity{i}", "Type", "Long description " * 100)
            for i in range(10)
        ]
        
        context = SearchContext(
            query="test",
            entities=entities,
        )
        
        # Check initial token count is large
        initial_tokens = context.get_token_count()
        assert initial_tokens > 1000
        
        # Trim context
        trimmed = context.trim_to_token_limit(max_tokens=500)
        
        # Check trimmed token count
        final_tokens = trimmed.get_token_count()
        assert final_tokens <= 500
        assert len(trimmed.entities) < len(entities)


class TestResponseGenerator:
    """Test ResponseGenerator class."""
    
    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test response generation."""
        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = "Generated response"
        mock_llm.achat.return_value = mock_response
        
        generator = ResponseGenerator(llm=mock_llm)
        
        # Create context
        context = SearchContext(
            query="test query",
            entities=[Entity("1", "Entity1", "Type1", "Description1")],
            communities=[Community("c1", "Community1", "Summary1")],
        )
        
        # Generate response
        response = await generator.generate_response(context)
        
        assert response == "Generated response"
        mock_llm.achat.assert_called_once()
    
    def test_create_summary_response(self):
        """Test summary response creation."""
        # Mock LLM to avoid API key requirement
        generator = ResponseGenerator(llm=MagicMock())
        
        # Create context
        context = SearchContext(
            query="test query",
            entities=[Entity("1", "Entity1", "Type1", "Description1")],
            communities=[Community("c1", "Community1", "Summary1", entities=["e1", "e2"])],
        )
        
        # Create summary
        summary = generator.create_summary_response(context)
        
        assert "test query" in summary
        assert "Entity1" in summary
        assert "Community1" in summary
        assert "2 сущностей" in summary  # Check entity count is included
    
    def test_validate_response(self):
        """Test response validation."""
        # Mock LLM to avoid API key requirement
        generator = ResponseGenerator(llm=MagicMock())
        
        context = SearchContext(query="test query about entities")
        
        # Valid response
        valid_response = "This is a detailed response about test entities with sufficient length."
        assert generator.validate_response(valid_response, context) is True
        
        # Too short response
        short_response = "Short"
        assert generator.validate_response(short_response, context) is False
        
        # Response not addressing query
        unrelated_response = "This is completely unrelated content " * 10
        assert generator.validate_response(unrelated_response, context) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])