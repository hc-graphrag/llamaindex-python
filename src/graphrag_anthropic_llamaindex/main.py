import os
import argparse

from llama_index.llms.anthropic import Anthropic
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

from graphrag_anthropic_llamaindex.config_manager import load_config
from graphrag_anthropic_llamaindex.vector_store_manager import get_vector_store
from graphrag_anthropic_llamaindex.document_processor import add_documents
from graphrag_anthropic_llamaindex.search_processor import search_index
from graphrag_anthropic_llamaindex.file_filter import FileFilter

def main():
    """Main function to run the GraphRAG CLI."""
    parser = argparse.ArgumentParser(description="A CLI for GraphRAG with LlamaIndex and Anthropic.")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'add' command
    add_parser = subparsers.add_parser("add", help="Add documents to the index.")

    # 'search' command
    search_parser = subparsers.add_parser("search", help="Search the index.")
    search_parser.add_argument("query", type=str, help="The search query.")
    search_parser.add_argument("--target-index", type=str, default="both", choices=["main", "entity", "community", "both"], help="Specify which index to search: 'main', 'entity', 'community', or 'both'.")

    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        return

    # LLMプロバイダーの設定を取得
    llm_provider = config.get("llm_provider", "anthropic")  # デフォルトはanthropic
    api_base_url = None  # 初期化
    
    if llm_provider == "bedrock":
        # AWS Bedrock設定
        bedrock_config = config.get("bedrock", {})
        model_name = bedrock_config.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
        region_name = bedrock_config.get("region", "us-east-1")
        aws_access_key_id = bedrock_config.get("aws_access_key_id")
        aws_secret_access_key = bedrock_config.get("aws_secret_access_key")
        aws_session_token = bedrock_config.get("aws_session_token")
    else:
        # Anthropic直接設定（従来の動作）
        anthropic_config = config.get("anthropic", {})
        os.environ["ANTHROPIC_API_KEY"] = anthropic_config.get("api_key")
        model_name = anthropic_config.get("model", "claude-3-opus-20240229")
        api_base_url = anthropic_config.get("api_base_url")

    input_dir = config.get("input_dir", "./data")
    output_dir = config.get("output_dir", "./graphrag_output")

    main_vector_store = get_vector_store(config, store_type="main")
    entity_vector_store = get_vector_store(config, store_type="entity")
    community_vector_store = get_vector_store(config, store_type="community")

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
    if llm_provider == "bedrock":
        # AWS Bedrock LLM設定
        bedrock_params = {
            "model": model_name,
            "region_name": region_name,
        }
        if aws_access_key_id:
            bedrock_params["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            bedrock_params["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            bedrock_params["aws_session_token"] = aws_session_token
        
        llm = Bedrock(**bedrock_params)
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
        search_index(args.query, output_dir, llm_params, main_vector_store,
                     entity_vector_store, community_vector_store, args.target_index)

if __name__ == "__main__":
    main()
