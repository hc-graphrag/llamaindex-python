import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings

from graphrag_anthropic_llamaindex.vector_store_manager import get_index
from graphrag_anthropic_llamaindex.llm_utils import get_configured_llm
import asyncio
import logging

logger = logging.getLogger(__name__)

def search_index(query, output_dir, llm_params, vector_store=None, entity_vector_store=None, community_vector_store=None, target_index="both", mode="auto"):
    """Searches the main text index and optionally the entity index with a given query."""
    print(f"\n=== Search Index Debug ===")
    print(f"Query: '{query}'")
    print(f"Output dir: {output_dir}")
    print(f"Target index: {target_index}")
    print(f"Mode: {mode}")
    print(f"Vector stores provided: main={vector_store is not None}, entity={entity_vector_store is not None}, community={community_vector_store is not None}")

    main_index = None
    entity_index = None
    community_index = None

    # Load main text index if requested or if entity index is not requested
    if target_index in ["main", "both"]:
        print(f"\n--- Loading Main Index ---")
        if vector_store:
            print(f"Using provided vector store: {type(vector_store).__name__}")
            main_index = VectorStoreIndex.from_vector_store(vector_store)
            print("✓ Main index loaded from vector store")
        else:
            storage_path = os.path.join(output_dir, "storage")
            print(f"Loading from storage path: {storage_path}")
            print(f"Storage path exists: {os.path.exists(storage_path)}")
            main_index = get_index(storage_path, index_type="main")
            if main_index:
                print("✓ Main index loaded from storage")
            else:
                print("❌ Failed to load main index from storage")

        if main_index is None:
            print("Main text index not found. Please add documents first using the 'add' command.")
            if target_index == "main": # If only main was requested and not found, return
                return

    # Load entity index if requested
    if target_index in ["entity", "both"]:
        print(f"\n--- Loading Entity Index ---")
        if entity_vector_store:
            print(f"Using provided entity vector store: {type(entity_vector_store).__name__}")
            entity_index = VectorStoreIndex.from_vector_store(entity_vector_store)
            print("✓ Entity index loaded from vector store")
        else:
            entity_index_dir = os.path.join(output_dir, "entities_index")
            print(f"Entity index directory: {entity_index_dir}")
            print(f"Entity index directory exists: {os.path.exists(entity_index_dir)}")
            if os.path.exists(entity_index_dir):
                entity_storage_context = StorageContext.from_defaults(persist_dir=entity_index_dir)
                entity_index = load_index_from_storage(entity_storage_context)
                if entity_index:
                    print("✓ Entity index loaded from storage")
                else:
                    print("❌ Failed to load entity index from storage")

        if entity_index is None:
            print("Entity index not found or not configured.")
            if target_index == "entity": # If only entity was requested and not found, return
                return

    # Load community summary index if requested
    if target_index in ["community", "both"]:
        print(f"\n--- Loading Community Index ---")
        if community_vector_store:
            print(f"Using provided community vector store: {type(community_vector_store).__name__}")
            community_index = VectorStoreIndex.from_vector_store(community_vector_store)
            print("✓ Community index loaded from vector store")
        else:
            community_index_dir = os.path.join(output_dir, "community_summaries_index")
            print(f"Community index directory: {community_index_dir}")
            print(f"Community index directory exists: {os.path.exists(community_index_dir)}")
            if os.path.exists(community_index_dir):
                community_storage_context = StorageContext.from_defaults(persist_dir=community_index_dir)
                community_index = load_index_from_storage(community_storage_context)
                if community_index:
                    print("✓ Community index loaded from storage")
                else:
                    print("❌ Failed to load community index from storage")

        if community_index is None:
            print("Community summary index not found or not configured.")
            if target_index == "community":
                return

    print(f"\n--- Index Loading Summary ---")
    print(f"Main index loaded: {main_index is not None}")
    print(f"Entity index loaded: {entity_index is not None}")
    print(f"Community index loaded: {community_index is not None}")

    if main_index and target_index in ["main", "both"]:
        print("\n=== Main Text Index Search ===")
        print(f"Query: '{query}'")
        print(f"Index type: {type(main_index).__name__}")
        try:
            main_query_engine = main_index.as_query_engine(llm=Settings.llm)
            print("✓ Query engine created")
            main_response = main_query_engine.query(query)
            print(f"✓ Query executed, response length: {len(str(main_response))}")
            print("Main Text Response:", main_response)

            # Debug response metadata if available
            if hasattr(main_response, 'source_nodes'):
                print(f"Source nodes found: {len(main_response.source_nodes)}")
        except Exception as e:
            print(f"❌ Error searching main index: {e}")
            logger.error(f"Main index search error: {e}", exc_info=True)

    if entity_index and target_index in ["entity", "both"]:
        print("\n=== Entity Index Search ===")
        print(f"Query: '{query}'")
        print(f"Index type: {type(entity_index).__name__}")
        try:
            entity_query_engine = entity_index.as_query_engine(llm=Settings.llm)
            print("✓ Entity query engine created")
            entity_response = entity_query_engine.query(query)
            print(f"✓ Entity query executed, response length: {len(str(entity_response))}")
            print("Entity Response:", entity_response)

            # Debug response metadata if available
            if hasattr(entity_response, 'source_nodes'):
                print(f"Entity source nodes found: {len(entity_response.source_nodes)}")
        except Exception as e:
            print(f"❌ Error searching entity index: {e}")
            logger.error(f"Entity index search error: {e}", exc_info=True)

    if community_index and target_index in ["community", "both"]:
        print("\n=== Community Summary Index Search ===")
        print(f"Query: '{query}'")
        print(f"Index type: {type(community_index).__name__}")
        try:
            community_query_engine = community_index.as_query_engine(llm=Settings.llm)
            print("✓ Community query engine created")
            community_response = community_query_engine.query(query)
            print(f"✓ Community query executed, response length: {len(str(community_response))}")
            print("Community Summary Response:", community_response)

            # Debug response metadata if available
            if hasattr(community_response, 'source_nodes'):
                print(f"Community source nodes found: {len(community_response.source_nodes)}")
        except Exception as e:
            print(f"❌ Error searching community index: {e}")
            logger.error(f"Community index search error: {e}", exc_info=True)

    # DRIFT検索モードの処理
    if mode == "drift" and vector_store and entity_vector_store and community_vector_store:
        print("=== Starting DRIFT Search Debug ===")
        print(f"Query: '{query}'")

        # Vector stores availability check
        print("\n--- Vector Store Debug ---")
        print(f"Main vector store: {type(vector_store).__name__ if vector_store else 'None'}")
        print(f"Entity vector store: {type(entity_vector_store).__name__ if entity_vector_store else 'None'}")
        print(f"Community vector store: {type(community_vector_store).__name__ if community_vector_store else 'None'}")

        # Config validation
        print("\n--- Configuration Debug ---")
        config = llm_params.get("config", {})
        print(f"Config keys: {list(config.keys())}")
        configured_llm = get_configured_llm()
        print(f"LLM settings: {type(configured_llm).__name__ if configured_llm else 'None'}")

        try:
            from .drift_search import DriftSearchEngine
            print("✓ DriftSearchEngine imported successfully")

            # ベクターストアを収集
            vector_stores = {
                "main": vector_store,
                "entity": entity_vector_store,
                "community": community_vector_store,
            }
            print(f"✓ Vector stores collected: {list(vector_stores.keys())}")

            # DRIFT検索エンジンを初期化
            print("\n--- Initializing DRIFT Engine ---")
            drift_engine = DriftSearchEngine(
                config=config,
                vector_stores=vector_stores,
                llm=Settings.llm
            )
            print("✓ DRIFT engine initialized successfully")

            # DRIFT検索を実行（非同期）
            print("\n--- Executing DRIFT Search ---")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            print("Starting async search...")
            response = loop.run_until_complete(
                drift_engine.search(query, streaming=False, include_context=True)
            )
            loop.close()
            print("✓ Async search completed")

            print("\n--- Processing Search Results ---")
            if isinstance(response, tuple):
                drift_response, context = response
                print(f"Response type: tuple (response + context)")
                print(f"Response length: {len(str(drift_response))}")
                print("DRIFT Search Response:", drift_response)

                print("\n--- DRIFT Search Context Analysis ---")
                if context:
                    entities = context.get('entities', [])
                    communities = context.get('communities', [])
                    text_units = context.get('text_units', [])

                    print(f"Context keys: {list(context.keys())}")
                    print(f"Entities found: {len(entities)}")
                    print(f"Communities found: {len(communities)}")
                    print(f"Text units found: {len(text_units)}")

                    # Sample entities for debugging
                    if entities:
                        print(f"Sample entities (first 3): {[e.get('name', 'no_name') for e in entities[:3]]}")

                    # Sample communities for debugging
                    if communities:
                        print(f"Sample communities (first 3): {[c.get('id', 'no_id') for c in communities[:3]]}")

                    # Check for empty context
                    total_items = len(entities) + len(communities) + len(text_units)
                    if total_items == 0:
                        print("⚠️  WARNING: All context arrays are empty!")
                    else:
                        print(f"✓ Total context items: {total_items}")
                else:
                    print("⚠️  WARNING: Context is None or empty!")

            else:
                print(f"Response type: {type(response)}")
                print("DRIFT Search Response:", response)
                print("⚠️  WARNING: No context returned (single response)")

            print("=== DRIFT Search Debug Complete ===\n")

        except ImportError as e:
            print(f"❌ Failed to import DRIFT search: {e}")
            logger.error(f"DRIFT import error: {e}", exc_info=True)
        except Exception as e:
            print(f"❌ Error executing DRIFT search: {e}")
            print(f"Error type: {type(e).__name__}")
            logger.error(f"DRIFT search error: {e}", exc_info=True)
    
    if not main_index and not entity_index and not community_index and mode != "drift":
        print("No index found to search based on the target_index setting.")