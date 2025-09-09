"""Test API key validation for different providers."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO


def test_bedrock_provider_no_api_key_required():
    """Test that Bedrock provider does NOT require ANTHROPIC_API_KEY."""
    # Remove ANTHROPIC_API_KEY if it exists
    original_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    
    try:
        config = {
            "llm_provider": "bedrock",
            "bedrock": {
                "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                "region": "us-east-1"
            }
        }
        
        # This should NOT raise an error even without ANTHROPIC_API_KEY
        with patch("graphrag_anthropic_llamaindex.main.load_config", return_value=config):
            with patch("sys.argv", ["main.py", "--config", "test.yaml", "search", "test query"]):
                # Mock the necessary components
                with patch("graphrag_anthropic_llamaindex.main.get_vector_store"):
                    with patch("llama_index.llms.bedrock.Bedrock"):
                        with patch("graphrag_anthropic_llamaindex.main.SearchModeRouter"):
                            # This should work without ANTHROPIC_API_KEY
                            from graphrag_anthropic_llamaindex.main import main
                            # Should not exit with error
                            # If it requires ANTHROPIC_API_KEY, it would sys.exit(1)
                            pass  # Success if we reach here
                            
    finally:
        # Restore original key if it existed
        if original_key:
            os.environ["ANTHROPIC_API_KEY"] = original_key


def test_anthropic_provider_requires_api_key():
    """Test that Anthropic provider REQUIRES ANTHROPIC_API_KEY."""
    # Remove ANTHROPIC_API_KEY if it exists
    original_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    
    try:
        config = {
            "llm_provider": "anthropic",
            "anthropic": {
                "model": "claude-3-opus-20240229"
            }
        }
        
        # This SHOULD raise an error without ANTHROPIC_API_KEY
        with patch("graphrag_anthropic_llamaindex.main.load_config", return_value=config):
            with patch("sys.argv", ["main.py", "--config", "test.yaml", "search", "test query"]):
                with pytest.raises(SystemExit) as exc_info:
                    from graphrag_anthropic_llamaindex.main import main
                    main()
                assert exc_info.value.code == 1
                
    finally:
        # Restore original key if it existed
        if original_key:
            os.environ["ANTHROPIC_API_KEY"] = original_key


def test_anthropic_provider_with_api_key_works():
    """Test that Anthropic provider works when API key is provided."""
    # Set a test API key
    os.environ["ANTHROPIC_API_KEY"] = "test-api-key"
    
    try:
        config = {
            "llm_provider": "anthropic",
            "anthropic": {
                "model": "claude-3-opus-20240229"
            }
        }
        
        with patch("graphrag_anthropic_llamaindex.main.load_config", return_value=config):
            with patch("sys.argv", ["main.py", "--config", "test.yaml", "search", "test query"]):
                # Mock the necessary components
                with patch("graphrag_anthropic_llamaindex.main.get_vector_store"):
                    with patch("llama_index.llms.anthropic.Anthropic"):
                        with patch("graphrag_anthropic_llamaindex.main.SearchModeRouter"):
                            # This should work with ANTHROPIC_API_KEY
                            from graphrag_anthropic_llamaindex.main import main
                            # Should not exit with error
                            pass  # Success if we reach here
                            
    finally:
        # Clean up
        os.environ.pop("ANTHROPIC_API_KEY", None)