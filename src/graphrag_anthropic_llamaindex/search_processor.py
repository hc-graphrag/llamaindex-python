import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

from graphrag_anthropic_llamaindex.vector_store_manager import get_index

def search_index(query, storage_dir, llm_params, vector_store=None, entity_vector_store=None, community_vector_store=None, service_context=None, target_index="both", data_storage_root_dir=None):
    """Searches the main text index and optionally the entity index with a given query."""
    main_index = None
    entity_index = None
    community_index = None

    # Load main text index if requested or if entity index is not requested
    if target_index in ["main", "both"]:
        if vector_store:
            main_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
        else:
            main_index = get_index(storage_dir, service_context=service_context, index_type="main")

        if main_index is None:
            print("Main text index not found. Please add documents first using the 'add' command.")
            if target_index == "main": # If only main was requested and not found, return
                return

    # Load entity index if requested
    if target_index in ["entity", "both"]:
        if entity_vector_store:
            entity_index = VectorStoreIndex.from_vector_store(entity_vector_store, service_context=service_context)
        else:
            entity_index_dir = os.path.join(storage_dir, "entities_index")
            if os.path.exists(entity_index_dir):
                entity_storage_context = StorageContext.from_defaults(persist_dir=entity_index_dir)
                entity_index = load_index_from_storage(entity_storage_context, service_context=service_context)

        if entity_index is None:
            print("Entity index not found or not configured.")
            if target_index == "entity": # If only entity was requested and not found, return
                return

    # Load community summary index if requested
    if target_index in ["community", "both"]:
        if community_vector_store:
            community_index = VectorStoreIndex.from_vector_store(community_vector_store, service_context=service_context)
        else:
            community_index_dir = os.path.join(storage_dir, "community_summaries_index")
            if os.path.exists(community_index_dir):
                community_storage_context = StorageContext.from_defaults(persist_dir=community_index_dir)
                community_index = load_index_from_storage(community_storage_context, service_context=service_context)
        
        if community_index is None:
            print("Community summary index not found or not configured.")
            if target_index == "community":
                return

    if main_index and target_index in ["main", "both"]:
        print("Searching main text index...")
        main_query_engine = main_index.as_query_engine(llm=service_context.llm)
        main_response = main_query_engine.query(query)
        print("Main Text Response:", main_response)

    if entity_index and target_index in ["entity", "both"]:
        print("\nSearching entity index...")
        entity_query_engine = entity_index.as_query_engine(llm=service_context.llm)
        entity_response = entity_query_engine.query(query)
        print("Entity Response:", entity_response)
    
    if community_index and target_index in ["community", "both"]:
        print("\nSearching community summary index...")
        community_query_engine = community_index.as_query_engine(llm=service_context.llm)
        community_response = community_query_engine.query(query)
        print("Community Summary Response:", community_response)

    if not main_index and not entity_index and not community_index:
        print("No index found to search based on the target_index setting.")
