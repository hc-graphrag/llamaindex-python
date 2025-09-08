#!/usr/bin/env python
"""
Simple test script for Global Search functionality
"""

import sys
import os
from unittest.mock import MagicMock, Mock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create a mock LLM class
from llama_index.core.llms import LLM
from llama_index.core.base.llms.types import CompletionResponse, ChatMessage

class MockLLM(LLM):
    @property
    def metadata(self):
        return {"model_name": "mock"}
    
    def complete(self, prompt, **kwargs):
        return CompletionResponse(text="Mock response")
    
    def chat(self, messages, **kwargs):
        return CompletionResponse(text="Mock chat response")
    
    def stream_complete(self, prompt, **kwargs):
        yield CompletionResponse(text="Mock stream response")
    
    def stream_chat(self, messages, **kwargs):
        yield CompletionResponse(text="Mock stream chat response")
    
    async def acomplete(self, prompt, **kwargs):
        return CompletionResponse(text="Mock async response")
    
    async def achat(self, messages, **kwargs):
        return CompletionResponse(text="Mock async chat response")
    
    async def astream_complete(self, prompt, **kwargs):
        yield CompletionResponse(text="Mock async stream response")
    
    async def astream_chat(self, messages, **kwargs):
        yield CompletionResponse(text="Mock async stream chat response")

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


def test_imports():
    """Test that all modules can be imported"""
    print("✓ All imports successful")
    return True


def test_data_models():
    """Test data model creation and serialization"""
    print("\n=== Testing Data Models ===")
    
    # Test KeyPoint
    kp = KeyPoint(
        description="Test key point",
        score=90,
        report_ids=["r1", "r2"],
        source_metadata={"document_ids": ["d1"]}
    )
    assert kp.description == "Test key point"
    assert kp.score == 90
    kp_dict = kp.to_dict()
    assert kp_dict["description"] == "Test key point"
    print("✓ KeyPoint model works")
    
    # Test MapResult
    mr = MapResult(
        batch_id=0,
        key_points=[kp],
        context_tokens=1000,
        processing_time=1.5
    )
    assert mr.batch_id == 0
    assert len(mr.key_points) == 1
    mr_dict = mr.to_dict()
    assert mr_dict["batch_id"] == 0
    print("✓ MapResult model works")
    
    # Test TraceabilityInfo
    ti = TraceabilityInfo(
        report_ids=["r1", "r2"],
        document_ids=["d1", "d2"],
        chunk_ids=["c1"],
        entity_ids=["e1"]
    )
    assert len(ti.report_ids) == 2
    ti_dict = ti.to_dict()
    assert len(ti_dict["document_ids"]) == 2
    print("✓ TraceabilityInfo model works")
    
    # Test GlobalSearchResult
    gsr = GlobalSearchResult(
        response="Test response",
        response_type="multiple paragraphs",
        map_results=[mr],
        traceability=ti,
        total_tokens=1000,
        processing_time=2.0
    )
    assert gsr.response == "Test response"
    
    # Test markdown formatting
    markdown = gsr.format_output("markdown")
    assert "# GLOBAL Search Result" in markdown
    assert "Test response" in markdown
    print("✓ GlobalSearchResult model works")
    
    # Test JSON formatting
    json_output = gsr.format_output("json")
    assert json_output["response"] == "Test response"
    assert json_output["total_tokens"] == 1000
    print("✓ JSON formatting works")
    
    return True


def test_key_point_extraction():
    """Test key point extraction from text"""
    print("\n=== Testing Key Point Extraction ===")
    
    # Mock the LLM in Settings to avoid OpenAI initialization
    from llama_index.core import Settings
    Settings.llm = MockLLM()
    
    llm_config = {"model": "test"}
    processor = MapProcessor(llm_config)
    
    # Test text extraction
    text_response = """
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
    
    key_points = processor.extract_key_points(text_response, report_ids, records)
    
    assert len(key_points) > 0, "Should extract key points from text"
    assert all(isinstance(kp, KeyPoint) for kp in key_points), "All items should be KeyPoint instances"
    assert all(kp.score >= 50 for kp in key_points), "All scores should be >= 50"
    print(f"✓ Extracted {len(key_points)} key points from text")
    
    # Test JSON extraction
    json_response = """
    ```json
    {
        "key_points": [
            {
                "description": "First key point from JSON",
                "score": 95,
                "report_ids": ["r1"]
            },
            {
                "description": "Second key point from JSON",
                "score": 85,
                "report_ids": ["r2", "r3"]
            }
        ]
    }
    ```
    """
    
    key_points_json = processor.extract_key_points(json_response, report_ids, records)
    
    assert len(key_points_json) == 2, "Should extract 2 key points from JSON"
    assert key_points_json[0].description == "First key point from JSON"
    assert key_points_json[0].score == 95
    assert key_points_json[1].description == "Second key point from JSON"
    assert key_points_json[1].score == 85
    print("✓ JSON key point extraction works")
    
    return True


def test_traceability_building():
    """Test traceability information building"""
    print("\n=== Testing Traceability Building ===")
    
    # Mock the LLM in Settings to avoid OpenAI initialization
    from llama_index.core import Settings
    Settings.llm = MockLLM()
    
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
    print("✓ Traceability building works correctly")
    
    return True


def test_context_builder_validation():
    """Test community context builder validation"""
    print("\n=== Testing Context Builder Validation ===")
    
    # Test configuration without community weighting (should fail)
    config_invalid = {
        "global_search": {
            "include_community_weight": False  # This should cause an error
        }
    }
    
    try:
        builder = CommunityContextBuilder(config_invalid)
        print("✗ Should have raised ValueError for missing community weight")
        return False
    except ValueError as e:
        assert "コミュニティ重み付けは必須です" in str(e)
        print("✓ Correctly validates community weight requirement")
    
    # Test valid configuration
    config_valid = {
        "global_search": {
            "include_community_weight": True,
            "max_context_tokens": 8000
        },
        "entity_extraction": {
            "enabled": True
        }
    }
    
    try:
        builder = CommunityContextBuilder(config_valid)
        print("✓ Valid configuration accepted")
    except Exception as e:
        print(f"✗ Valid configuration failed: {e}")
        return False
    
    return True


def test_search_mode_router():
    """Test search mode routing"""
    print("\n=== Testing Search Mode Router ===")
    
    from graphrag_anthropic_llamaindex.global_search.router import SearchMode
    
    config = {
        "llm": {"model": "test"},
        "global_search": {"include_community_weight": True}
    }
    
    # Test mode initialization
    router = SearchModeRouter(config=config, mode="global")
    assert router.mode == SearchMode.GLOBAL
    print("✓ Router initializes with correct mode")
    
    # Test auto mode selection
    router_auto = SearchModeRouter(config=config, mode="auto")
    router_auto.global_retriever = True  # Mock retriever existence
    router_auto.local_retriever = True  # Mock retriever existence
    
    # Test global keywords
    mode = router_auto._auto_select_mode("Give me an overall summary")
    assert mode == SearchMode.GLOBAL
    print("✓ Auto-selects global mode for summary queries")
    
    # Test local keywords
    mode = router_auto._auto_select_mode("Show me specific details about X")
    assert mode == SearchMode.LOCAL
    print("✓ Auto-selects local mode for detail queries")
    
    # Test available modes
    router_test = SearchModeRouter(config=config, mode="global")
    modes = router_test.get_available_modes()
    assert len(modes) == 0  # No retrievers initialized
    
    router_test.global_retriever = True  # Mock
    modes = router_test.get_available_modes()
    assert len(modes) == 1
    assert SearchMode.GLOBAL in modes
    print("✓ Available modes detection works")
    
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 50)
    print("GLOBAL SEARCH FUNCTIONALITY TESTS")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Models", test_data_models),
        ("Key Point Extraction", test_key_point_extraction),
        ("Traceability Building", test_traceability_building),
        ("Context Builder Validation", test_context_builder_validation),
        ("Search Mode Router", test_search_mode_router)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            import traceback
            print(f"\n✗ {test_name} failed with error: {e}")
            print("Traceback:")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)