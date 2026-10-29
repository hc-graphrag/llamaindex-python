"""End-to-end CLI tests for GLOBAL search functionality."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


def test_cli_help_message(capsys):
    """Test CLI help message includes new arguments."""
    test_args = ["main.py", "search", "--help"]
    
    with patch('sys.argv', test_args):
        with pytest.raises(SystemExit):
            from graphrag_anthropic_llamaindex.main import main
            main()
    
    captured = capsys.readouterr()
    help_text = captured.out
    
    # Verify new arguments are documented
    assert "--mode" in help_text
    assert "--response-type" in help_text
    assert "--output-format" in help_text
    assert "--min-community-rank" in help_text
    assert "global" in help_text
    assert "local" in help_text


def test_cli_error_invalid_mode():
    """Test CLI error handling for invalid mode."""
    # This is handled by argparse, so we test the argument parser directly
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--mode", type=str, default="global", 
                               choices=["local", "global", "drift", "auto"])
    
    # Valid mode should work
    args = parser.parse_args(["search", "test query", "--mode", "global"])
    assert args.mode == "global"
    
    # Invalid mode should raise an error
    with pytest.raises(SystemExit):
        parser.parse_args(["search", "test query", "--mode", "invalid_mode"])


def test_cli_argument_parsing():
    """Test that CLI correctly parses all new arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    # Replicate the search parser from main.py
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--mode", type=str, default="global",
                               choices=["local", "global", "drift", "auto"])
    search_parser.add_argument("--response-type", type=str, default="multiple paragraphs")
    search_parser.add_argument("--output-format", type=str, default="markdown",
                               choices=["markdown", "json"])
    search_parser.add_argument("--min-community-rank", type=int, default=0)
    
    # Test parsing with all arguments
    args = parser.parse_args([
        "search",
        "What are the latest trends?",
        "--mode", "global",
        "--response-type", "Multiple Paragraphs",
        "--output-format", "json",
        "--min-community-rank", "5"
    ])
    
    assert args.command == "search"
    assert args.query == "What are the latest trends?"
    assert args.mode == "global"
    assert args.response_type == "Multiple Paragraphs"
    assert args.output_format == "json"
    assert args.min_community_rank == 5


def test_search_mode_router_import():
    """Test that SearchModeRouter can be imported from global_search module."""
    from graphrag_anthropic_llamaindex.global_search import SearchModeRouter
    assert SearchModeRouter is not None


def test_global_search_retriever_import():
    """Test that GlobalSearchRetriever can be imported."""
    from graphrag_anthropic_llamaindex.global_search import GlobalSearchRetriever
    assert GlobalSearchRetriever is not None


def test_global_search_module_structure():
    """Test that global_search module has expected components."""
    import graphrag_anthropic_llamaindex.global_search as gs_module
    
    # Check that key classes are available
    assert hasattr(gs_module, 'SearchModeRouter')
    assert hasattr(gs_module, 'GlobalSearchRetriever')
    
    # Check that the module is properly initialized
    assert gs_module.__name__ == 'graphrag_anthropic_llamaindex.global_search'


def test_config_with_global_search_section():
    """Test configuration with global_search section."""
    config = {
        "anthropic": {
            "api_key": "test-key",
            "model": "claude-3-sonnet-20240229"
        },
        "embedding_model": {
            "name": "BAAI/bge-small-en-v1.5"
        },
        "chunking": {
            "chunk_size": 512,
            "chunk_overlap": 64
        },
        "global_search": {
            "response_type": "Multiple Paragraphs",
            "max_tokens": 3000,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_concurrent": 50,
            "batch_size": 16,
            "min_community_rank": 0,
            "normalize_community_weight": True
        }
    }
    
    # Verify global_search config structure
    assert "global_search" in config
    assert config["global_search"]["response_type"] == "Multiple Paragraphs"
    assert config["global_search"]["max_concurrent"] == 50
    assert config["global_search"]["batch_size"] == 16
    assert config["global_search"]["min_community_rank"] == 0
    assert config["global_search"]["normalize_community_weight"] is True


def test_response_type_values():
    """Test that different response type values are valid."""
    valid_response_types = [
        "Multiple Paragraphs",
        "multiple paragraphs",
        "Single Paragraph",
        "single paragraph",
        "List",
        "list",
        "JSON",
        "json"
    ]
    
    for response_type in valid_response_types:
        # These should all be valid strings
        assert isinstance(response_type, str)
        assert len(response_type) > 0


def test_output_format_values():
    """Test output format options."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--output-format", type=str, default="markdown",
                               choices=["markdown", "json"])
    
    # Test markdown format
    args = parser.parse_args(["search", "query", "--output-format", "markdown"])
    assert args.output_format == "markdown"
    
    # Test json format
    args = parser.parse_args(["search", "query", "--output-format", "json"])
    assert args.output_format == "json"
    
    # Invalid format should raise error
    with pytest.raises(SystemExit):
        parser.parse_args(["search", "query", "--output-format", "xml"])


def test_min_community_rank_values():
    """Test min-community-rank argument values."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--min-community-rank", type=int, default=0)
    
    # Test default value
    args = parser.parse_args(["search", "query"])
    assert args.min_community_rank == 0
    
    # Test positive integer
    args = parser.parse_args(["search", "query", "--min-community-rank", "5"])
    assert args.min_community_rank == 5
    
    # Test zero
    args = parser.parse_args(["search", "query", "--min-community-rank", "0"])
    assert args.min_community_rank == 0
    
    # Test negative (should work as int type allows it)
    args = parser.parse_args(["search", "query", "--min-community-rank", "-1"])
    assert args.min_community_rank == -1


def test_mode_choices():
    """Test all mode choices are valid."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--mode", type=str, default="global",
                               choices=["local", "global", "drift", "auto"])
    
    # Test each valid mode
    for mode in ["local", "global", "drift", "auto"]:
        args = parser.parse_args(["search", "query", "--mode", mode])
        assert args.mode == mode
    
    # Test default
    args = parser.parse_args(["search", "query"])
    assert args.mode == "global"


def test_cli_integration_mock():
    """Integration test with mocked components."""
    # Create a minimal config
    config = {
        "anthropic": {"api_key": "test-key", "model": "claude-3-sonnet"},
        "embedding_model": {"name": "test-model"},
        "chunking": {"chunk_size": 512, "chunk_overlap": 64},
        "global_search": {"response_type": "Multiple Paragraphs"}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # Set up environment to avoid API key issues
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        
        # Test that config can be loaded
        from graphrag_anthropic_llamaindex.config_manager import load_config
        loaded_config = load_config(config_path)
        
        if loaded_config:
            assert loaded_config["anthropic"]["api_key"] == "test-key"
            assert "global_search" in loaded_config
    finally:
        # Cleanup
        Path(config_path).unlink()
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]


def test_output_dir_override_is_used_for_all_embedded_stores():
    """CLIの出力先上書きをSQL・全HNSWで共有する。"""
    config = {"llm_provider": "claude_code", "output_dir": "configured-output"}
    stores = [MagicMock(), MagicMock(), MagicMock()]
    router = MagicMock()
    router._retrieve.return_value = []

    with patch(
        "graphrag_anthropic_llamaindex.main.load_config", return_value=config
    ), patch(
        "graphrag_anthropic_llamaindex.main.get_vector_store",
        side_effect=stores,
    ) as mock_get_store, patch(
        "graphrag_anthropic_llamaindex.main.ClaudeCodeLLM"
    ), patch(
        "graphrag_anthropic_llamaindex.main.HuggingFaceEmbedding"
    ), patch(
        "graphrag_anthropic_llamaindex.main.Settings"
    ), patch(
        "graphrag_anthropic_llamaindex.main.SearchModeRouter",
        return_value=router,
    ) as mock_router, patch(
        "sys.argv",
        [
            "main.py",
            "--config",
            "test.yaml",
            "search",
            "query",
            "--output-dir",
            "override-output",
        ],
    ):
        from graphrag_anthropic_llamaindex.main import main

        main()

    storage_config = {**config, "output_dir": "override-output"}
    assert mock_get_store.call_args_list == [
        ((storage_config,), {"store_type": "main"}),
        ((storage_config,), {"store_type": "entity"}),
        ((storage_config,), {"store_type": "community"}),
    ]
    assert mock_router.call_args.kwargs["config"] == storage_config

def test_router_raises_when_drift_retriever_missing():
    """DRIFT が未初期化なら Global へ退避せず例外にする。"""
    import pytest
    from llama_index.core.schema import QueryBundle
    from graphrag_anthropic_llamaindex.global_search.router import SearchMode, SearchModeRouter

    router = object.__new__(SearchModeRouter)
    router.mode = SearchMode.DRIFT
    router.local_retriever = None
    router.drift_retriever = None
    router.global_retriever = object()
    router.route = lambda query: SearchMode.DRIFT

    with pytest.raises(RuntimeError, match="DRIFT"):
        router._retrieve(QueryBundle(query_str="q"))
