import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings

from graphrag_anthropic_llamaindex.vector_store_manager import get_index
import asyncio
import logging

logger = logging.getLogger(__name__)

def search_index(query, output_dir, llm_params, vector_store=None, entity_vector_store=None, community_vector_store=None, target_index="both", mode="auto"):
    """Searches the main text index and optionally the entity index with a given query."""
    main_index = None
    entity_index = None
    community_index = None

    # Load main text index if requested or if entity index is not requested
    if target_index in ["main", "both"]:
        if vector_store:
            main_index = VectorStoreIndex.from_vector_store(vector_store)
        else:
            main_index = get_index(os.path.join(output_dir, "storage"), index_type="main")

        if main_index is None:
            print("Main text index not found. Please add documents first using the 'add' command.")
            if target_index == "main": # If only main was requested and not found, return
                return

    # Load entity index if requested
    if target_index in ["entity", "both"]:
        if entity_vector_store:
            entity_index = VectorStoreIndex.from_vector_store(entity_vector_store)
        else:
            entity_index_dir = os.path.join(output_dir, "entities_index")
            if os.path.exists(entity_index_dir):
                entity_storage_context = StorageContext.from_defaults(persist_dir=entity_index_dir)
                entity_index = load_index_from_storage(entity_storage_context)

        if entity_index is None:
            print("Entity index not found or not configured.")
            if target_index == "entity": # If only entity was requested and not found, return
                return

    # Load community summary index if requested
    if target_index in ["community", "both"]:
        if community_vector_store:
            community_index = VectorStoreIndex.from_vector_store(community_vector_store)
        else:
            community_index_dir = os.path.join(output_dir, "community_summaries_index")
            if os.path.exists(community_index_dir):
                community_storage_context = StorageContext.from_defaults(persist_dir=community_index_dir)
                community_index = load_index_from_storage(community_storage_context)
        
        if community_index is None:
            print("Community summary index not found or not configured.")
            if target_index == "community":
                return

    if main_index and target_index in ["main", "both"]:
        print("Searching main text index...")
        try:
            main_query_engine = main_index.as_query_engine(llm=Settings.llm)
            main_response = main_query_engine.query(query)
            print("Main Text Response:", main_response)
        except Exception as e:
            print(f"Error searching main index: {e}")

    if entity_index and target_index in ["entity", "both"]:
        print("\nSearching entity index...")
        try:
            entity_query_engine = entity_index.as_query_engine(llm=Settings.llm)
            entity_response = entity_query_engine.query(query)
            print("Entity Response:", entity_response)
        except Exception as e:
            print(f"Error searching entity index: {e}")
    
    if community_index and target_index in ["community", "both"]:
        print("\nSearching community summary index...")
        try:
            community_query_engine = community_index.as_query_engine(llm=Settings.llm)
            community_response = community_query_engine.query(query)
            print("Community Summary Response:", community_response)
        except Exception as e:
            print(f"Error searching community index: {e}")

    # DRIFT検索モードの処理
    if mode == "drift" and vector_store and entity_vector_store and community_vector_store:
        print("Executing DRIFT search...")
        try:
            from .drift_search import DriftSearchEngine
            
            # ベクターストアを収集
            vector_stores = {
                "main": vector_store,
                "entity": entity_vector_store,
                "community": community_vector_store,
            }
            
            # DRIFT検索エンジンを初期化
            drift_engine = DriftSearchEngine(
                config=llm_params.get("config", {}),
                vector_stores=vector_stores,
                llm=Settings.llm
            )
            
            # DRIFT検索を実行（非同期）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                drift_engine.search(query, streaming=False, include_context=True)
            )
            loop.close()
            
            if isinstance(response, tuple):
                drift_response, context = response
                print("DRIFT Search Response:", drift_response)
                print("\n--- DRIFT Search Context ---")
                print(f"Entities found: {len(context.get('entities', []))}")
                print(f"Communities found: {len(context.get('communities', []))}")
                print(f"Text units found: {len(context.get('text_units', []))}")
            else:
                print("DRIFT Search Response:", response)
                
        except Exception as e:
            print(f"Error executing DRIFT search: {e}")
            logger.error(f"DRIFT search error: {e}", exc_info=True)
    
    if not main_index and not entity_index and not community_index and mode != "drift":
        print("No index found to search based on the target_index setting.")