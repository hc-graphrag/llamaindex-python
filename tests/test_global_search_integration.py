"""
Integration tests for the Global Search functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

from graphrag_anthropic_llamaindex.global_search import (
    GlobalSearchRetriever,
    SearchModeRouter,
    GlobalSearchResult,
    MapResult,
    KeyPoint,
    TraceabilityInfo
)
from graphrag_anthropic_llamaindex.global_search.context_builder import CommunityContextBuilder
from graphrag_anthropic_llamaindex.global_search.map_processor import MapProcessor
from graphrag_anthropic_llamaindex.global_search.reduce_processor import ReduceProcessor

from llama_index.core.schema import QueryBundle, NodeWithScore


class TestGlobalSearchRetriever:
    """Test GlobalSearchRetriever class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        return {
            "llm": {
                "model": "test-model"
            },
            "global_search": {
                "max_context_tokens": 8000,
                "include_community_weight": True,
                "include_key_points": False,
                "min_community_rank": 0,
                "max_concurrent": 5
            },
            "entity_extraction": {
                "enabled": True
            }
        }
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store"""
        mock_store = Mock()
        return mock_store
    
    def test_retriever_initialization(self, mock_config, mock_vector_store):
        """Test that GlobalSearchRetriever initializes correctly"""
        with patch('graphrag_anthropic_llamaindex.global_search.retriever.CommunityContextBuilder'):
            with patch('graphrag_anthropic_llamaindex.global_search.retriever.MapProcessor'):
                with patch('graphrag_anthropic_llamaindex.global_search.retriever.ReduceProcessor'):
                    retriever = GlobalSearchRetriever(
                        config=mock_config,
                        vector_store=mock_vector_store,
                        response_type="multiple paragraphs",
                        min_community_rank=0
                    )
                    
                    assert retriever.config == mock_config
                    assert retriever.response_type == "multiple paragraphs"
                    assert retriever.min_community_rank == 0
                    assert retriever.output_format == "markdown"
    
    def test_create_nodes(self, mock_config):
        """Test node creation from GlobalSearchResult"""
        # Create mock result
        map_result = MapResult(
            batch_id=0,
            key_points=[
                KeyPoint(
                    description="Test point 1",
                    score=90,
                    report_ids=["r1", "r2"],
                    source_metadata={"document_ids": ["d1"]}
                )
            ],
            context_tokens=1000,
            processing_time=1.5
        )
        
        traceability = TraceabilityInfo(
            report_ids=["r1", "r2"],
            document_ids=["d1", "d2"],
            chunk_ids=["c1"],
            entity_ids=["e1"]
        )
        
        global_result = GlobalSearchResult(
            response="Test response",
            response_type="multiple paragraphs",
            map_results=[map_result],
            traceability=traceability,
            total_tokens=1000,
            processing_time=2.0
        )
        
        with patch('graphrag_anthropic_llamaindex.global_search.retriever.CommunityContextBuilder'):
            with patch('graphrag_anthropic_llamaindex.global_search.retriever.MapProcessor'):
                with patch('graphrag_anthropic_llamaindex.global_search.retriever.ReduceProcessor'):
                    retriever = GlobalSearchRetriever(
                        config=mock_config,
                        response_type="multiple paragraphs"
                    )
                    
                    nodes = retriever._create_nodes(global_result)
                    
                    assert len(nodes) >= 1
                    assert isinstance(nodes[0], NodeWithScore)
                    assert nodes[0].node.text == "Test response"
                    assert nodes[0].node.metadata["search_type"] == "global"
                    assert nodes[0].score == 1.0


class TestMapProcessor:
    """Test MapProcessor class"""
    
    def test_extract_key_points_from_text(self):
        """Test key point extraction from text response"""
        llm_config = {"model": "test"}
        processor = MapProcessor(llm_config)
        
        llm_response = """
        Here are the key findings:
        
        1. First important point about the data
        
        2. Second critical insight from the analysis
        
        3. Third observation regarding the patterns
        """
        
        report_ids = ["r1", "r2", "r3"]
        records = [
            {"id": "r1", "metadata": {"document_id": "d1"}},
            {"id": "r2", "metadata": {"document_id": "d2"}}
        ]
        
        key_points = processor.extract_key_points(llm_response, report_ids, records)
        
        assert len(key_points) > 0
        assert all(isinstance(kp, KeyPoint) for kp in key_points)
        assert all(kp.score >= 50 for kp in key_points)
    
    def test_extract_key_points_from_json(self):
        """Test key point extraction from JSON response"""
        llm_config = {"model": "test"}
        processor = MapProcessor(llm_config)
        
        llm_response = """
        ```json
        {
            "key_points": [
                {
                    "description": "First key point",
                    "score": 95,
                    "report_ids": ["r1"]
                },
                {
                    "description": "Second key point",
                    "score": 85,
                    "report_ids": ["r2", "r3"]
                }
            ]
        }
        ```
        """
        
        report_ids = ["r1", "r2", "r3"]
        records = []
        
        key_points = processor.extract_key_points(llm_response, report_ids, records)
        
        assert len(key_points) == 2
        assert key_points[0].description == "First key point"
        assert key_points[0].score == 95
        assert key_points[1].description == "Second key point"
        assert key_points[1].score == 85


class TestReduceProcessor:
    """Test ReduceProcessor class"""
    
    def test_build_traceability(self):
        """Test traceability information building"""
        llm_config = {"model": "test"}
        processor = ReduceProcessor(llm_config)
        
        key_points = [
            KeyPoint(
                description="Point 1",
                score=90,
                report_ids=["r1", "r2"],
                source_metadata={
                    "document_ids": ["d1", "d2"],
                    "chunk_ids": ["c1"],
                    "entity_ids": ["e1", "e2"]
                }
            ),
            KeyPoint(
                description="Point 2",
                score=80,
                report_ids=["r3"],
                source_metadata={
                    "document_ids": ["d3"],
                    "chunk_ids": ["c2", "c3"],
                    "entity_ids": ["e3"]
                }
            )
        ]
        
        traceability = processor._build_traceability(key_points)
        
        assert isinstance(traceability, TraceabilityInfo)
        assert set(traceability.report_ids) == {"r1", "r2", "r3"}
        assert set(traceability.document_ids) == {"d1", "d2", "d3"}
        assert set(traceability.chunk_ids) == {"c1", "c2", "c3"}
        assert set(traceability.entity_ids) == {"e1", "e2", "e3"}


class TestSearchModeRouter:
    """Test SearchModeRouter class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        return {
            "llm": {"model": "test"},
            "global_search": {"include_community_weight": True}
        }
    
    def test_router_initialization(self, mock_config):
        """Test SearchModeRouter initialization"""
        router = SearchModeRouter(
            config=mock_config,
            mode="global"
        )
        
        assert router.mode.value == "global"
        assert router.config == mock_config
    
    def test_auto_mode_selection(self, mock_config):
        """Test automatic mode selection based on query"""
        router = SearchModeRouter(
            config=mock_config,
            mode="auto"
        )
        
        # Mock retrievers
        router.global_retriever = Mock()
        router.local_retriever = Mock()
        
        # Test global keywords
        global_mode = router._auto_select_mode("Give me an overall summary")
        assert global_mode.value == "global"
        
        # Test local keywords
        local_mode = router._auto_select_mode("Show me specific details about X")
        assert local_mode.value == "local"
    
    def test_get_available_modes(self, mock_config):
        """Test getting available search modes"""
        router = SearchModeRouter(
            config=mock_config,
            mode="global"
        )
        
        # Initially no modes available
        modes = router.get_available_modes()
        assert len(modes) == 0
        
        # Add a global retriever
        router.global_retriever = Mock()
        modes = router.get_available_modes()
        assert len(modes) == 1
        assert modes[0].value == "global"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])