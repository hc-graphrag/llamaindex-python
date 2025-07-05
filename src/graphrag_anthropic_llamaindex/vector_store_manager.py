import os
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.lancedb import LanceDBVectorStore

def get_vector_store(config, store_type="main"):
    """Initializes the vector store based on the configuration."""
    if store_type == "main":
        vs_config = config.get("vector_store", {})
    elif store_type == "entity":
        vs_config = config.get("entity_vector_store", {})
    elif store_type == "community":
        vs_config = config.get("community_vector_store", {})
    else:
        return None

    if vs_config.get("type") == "lancedb":
        lancedb_config = vs_config.get("lancedb", {})
        return LanceDBVectorStore(
            uri=lancedb_config.get("uri", "./lancedb"),
            table_name=lancedb_config.get("table_name", "vectors"),
            mode="overwrite", # Consider changing to "append" if you want to add to existing table
        )
    return None # Falls back to default in-memory

def get_index(storage_dir, vector_store=None, service_context=None, index_type="main"):
    """Loads the index from storage if it exists, otherwise creates a new one."""
    if index_type == "main" and vector_store is None and os.path.exists(storage_dir):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context, service_context=service_context)
    elif vector_store is not None:
        return VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
    return None
