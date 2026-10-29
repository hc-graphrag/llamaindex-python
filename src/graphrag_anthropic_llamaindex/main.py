import os
import sys
import argparse
from dotenv import load_dotenv

from llama_index.llms.anthropic import Anthropic
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

from graphrag_anthropic_llamaindex.claude_code_llm import ClaudeCodeLLM
from graphrag_anthropic_llamaindex.config_manager import load_config
from graphrag_anthropic_llamaindex.vector_store_manager import get_vector_store
from graphrag_anthropic_llamaindex.document_processor import add_documents
from graphrag_anthropic_llamaindex.file_filter import FileFilter
from graphrag_anthropic_llamaindex.global_search import SearchModeRouter, GlobalSearchRetriever

def main():
    """Main function to run the GraphRAG CLI."""
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="A CLI for GraphRAG with LlamaIndex and Anthropic.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the configuration file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'add' command
    add_parser = subparsers.add_parser("add", help="Add documents to the index.")
    add_parser.add_argument("input_dir", nargs="?", help="Input directory containing documents (optional, uses config if not specified)")
    add_parser.add_argument("--output-dir", help="Output directory for processed data (optional, uses config if not specified)")

    # 'search' command
    search_parser = subparsers.add_parser("search", help="Search the index.")
    search_parser.add_argument("query", type=str, help="The search query.")
    search_parser.add_argument("--mode", type=str, default="global", choices=["local", "global", "drift", "auto"], 
                               help="Search mode: 'local' (detailed), 'global' (comprehensive), 'drift' (future), or 'auto' (automatic selection).")
    search_parser.add_argument("--response-type", type=str, default="multiple paragraphs",
                               help="Response type for global search (e.g., 'multiple paragraphs', 'single paragraph', 'list').")
    search_parser.add_argument("--output-format", type=str, default="markdown", choices=["markdown", "json"],
                               help="Output format: 'markdown' or 'json'.")
    search_parser.add_argument("--min-community-rank", type=int, default=0,
                               help="Minimum community rank to include (0 = all levels).")
    search_parser.add_argument("--output-dir", help="Output directory for search index (optional, uses config if not specified)")

    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        return

    # LLMプロバイダーの設定を取得
    llm_provider = config.get("llm_provider", "anthropic")  # デフォルトはanthropic
    api_base_url = None  # 初期化
    
    if llm_provider == "claude_code":
        # Claude Code (claude CLI) のサブスクリプション認証 - APIキー不要
        model_name = config.get("claude_code", {}).get("model", "sonnet")
    elif llm_provider == "bedrock":
        # AWS Bedrock設定 - ANTHROPIC_API_KEYは不要
        bedrock_config = config.get("bedrock", {})
        model_name = bedrock_config.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
        region_name = bedrock_config.get("region", "us-east-1")
        # AWS Profile support (環境変数からのみ取得)
        aws_profile_name = os.environ.get("AWS_PROFILE_NAME")
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
    else:
        # Anthropic直接設定 - API_KEYが必要
        anthropic_config = config.get("anthropic", {})
        # 環境変数からのみ取得
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found in environment variables")
            print("Please set it using: export ANTHROPIC_API_KEY='your-api-key'")
            print("Or use the Claude subscription login via llm_provider: 'claude_code', or AWS Bedrock via 'bedrock' in config/config.yaml")
            sys.exit(1)
        model_name = anthropic_config.get("model", "claude-sonnet-5-5")
        api_base_url = anthropic_config.get("api_base_url")

    # Override input/output directories from command line if provided
    input_dir = args.input_dir if hasattr(args, 'input_dir') and args.input_dir else config.get("input_dir", "./data")
    if hasattr(args, 'output_dir') and args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config.get("output_dir", "./graphrag_output")

    storage_config = {**config, "output_dir": output_dir}
    main_vector_store = get_vector_store(storage_config, store_type="main")
    entity_vector_store = get_vector_store(storage_config, store_type="entity")
    community_vector_store = get_vector_store(storage_config, store_type="community")

    llm_params = {"model": model_name}
    if api_base_url:
        llm_params["api_base_url"] = api_base_url

    # Configure embedding model
    embedding_config = config.get("embedding_model", {})
    embed_model_name = embedding_config.get("name", "intfloat/multilingual-e5-small")
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # Configure chunking
    chunking_config = config.get("chunking", {})
    chunk_size = chunking_config.get("chunk_size", 1024)
    chunk_overlap = chunking_config.get("chunk_overlap", 20)
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Configure Settings based on provider
    if llm_provider == "claude_code":
        Settings.llm = ClaudeCodeLLM(model=model_name)
        print(f"Using Claude Code (subscription login) with model: {model_name}")
    elif llm_provider == "bedrock":
        # AWS Bedrock LLM設定
        bedrock_params = {
            "model": model_name,
            "region_name": region_name,
        }
        # AWS Profileを優先的に使用
        if aws_profile_name:
            bedrock_params["profile_name"] = aws_profile_name
        elif aws_access_key_id and aws_secret_access_key:
            # Profileが指定されていない場合のみ、明示的な認証情報を使用
            bedrock_params["aws_access_key_id"] = aws_access_key_id
            bedrock_params["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                bedrock_params["aws_session_token"] = aws_session_token
        
        llm = BedrockConverse(**bedrock_params)
        Settings.llm = llm
        print(f"Using AWS Bedrock with model: {model_name}")
    else:
        # Anthropic直接LLM設定
        llm = Anthropic(**llm_params)
        Settings.llm = llm
        print(f"Using Anthropic API with model: {model_name}")
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    community_detection_config = config.get("community_detection", {})
    ignore_patterns = config.get("ignore_patterns", [])
    
    # Create file filter with ignore patterns
    file_filter = FileFilter(ignore_patterns)

    if args.command == "add":
        add_documents(input_dir, output_dir, main_vector_store,
                      entity_vector_store,
                      community_vector_store, community_detection_config,
                      use_archive_reader=True, file_filter=file_filter)
    elif args.command == "search":
        # config の search.mode が CLI の --mode より優先される
        config_mode = config.get("search", {}).get("mode", args.mode)
        mode = config_mode if config_mode else args.mode
        print(f"Using search mode: {mode} (from {'config' if config_mode != args.mode else 'command line'})")

        # Use the new SearchModeRouter for unified search interface
        router = SearchModeRouter(
            config=storage_config,
            mode=mode,
            vector_store_main=main_vector_store,
            vector_store_entity=entity_vector_store,
            vector_store_community=community_vector_store,
            response_type=args.response_type,
            min_community_rank=args.min_community_rank,
            output_format=args.output_format
        )
        
        # Execute search
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=args.query)
        results = router._retrieve(query_bundle)
        
        # Display results
        if results:
            for i, node_with_score in enumerate(results):
                if i == 0:  # Main result
                    if args.output_format == "json":
                        import json
                        print(json.dumps(node_with_score.node.metadata, ensure_ascii=False, indent=2))
                    else:
                        print(node_with_score.node.text)
                else:
                    # Additional nodes (key points, etc.) if included
                    if args.output_format == "json":
                        print(f"\n--- Key Point {i} (Score: {node_with_score.score:.2f}) ---")
                        print(node_with_score.node.text)
        else:
            print("No results found.")

if __name__ == "__main__":
    main()
