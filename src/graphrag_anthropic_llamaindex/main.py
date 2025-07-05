import os
import argparse

from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, ServiceContext
from llama_index.core.node_parser import SentenceSplitter

from graphrag_anthropic_llamaindex.config_manager import load_config
from graphrag_anthropic_llamaindex.vector_store_manager import get_vector_store
from graphrag_anthropic_llamaindex.document_processor import add_documents
from graphrag_anthropic_llamaindex.search_processor import search_index

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

    anthropic_config = config.get("anthropic", {})
    os.environ["ANTHROPIC_API_KEY"] = anthropic_config.get("api_key")
    model_name = anthropic_config.get("model", "claude-3-opus-20240229")
    api_base_url = anthropic_config.get("api_base_url")

    data_dir = config.get("data", {}).get("directory", "data")
    storage_dir = config.get("storage", {}).get("directory", "storage")
    data_storage_root_dir = config.get("data_storage_root", "./graphrag_data")

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

    # Create ServiceContext
    llm = Anthropic(**llm_params)
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )

    # Set tokenizer for LlamaIndex Settings
    Settings.tokenizer = llm.tokenizer

    community_detection_config = config.get("community_detection", {})

    if args.command == "add":
        add_documents(data_dir, storage_dir, main_vector_store, data_storage_root_dir,
                      entity_vector_store,
                      community_vector_store, service_context, community_detection_config)
    elif args.command == "search":
        search_index(args.query, storage_dir, llm_params, main_vector_store,
                     entity_vector_store, community_vector_store, service_context, args.target_index)

if __name__ == "__main__":
    main()
