import os
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.lancedb import LanceDBVectorStore

# Constants for vector store table names
VECTOR_STORE_TABLE_NAMES = {
    "main": "vectors",           # Main document vectors
    "entity": "entities_vectors", # Entity vectors
    "community": "community_vectors"  # Community vectors
}

def get_vector_store(config, store_type="main"):
    """Initializes the vector store based on the configuration.
    
    All vector stores are consolidated into a single LanceDB database with
    different tables for each store type.
    """
    # Get the table name for the specified store type
    table_name = VECTOR_STORE_TABLE_NAMES.get(store_type)
    if table_name is None:
        return None

    # Get the main vector store configuration
    vs_config = config.get("vector_store", {})
    
    if vs_config.get("type") == "lancedb":
        lancedb_config = vs_config.get("lancedb", {})
        
        # パス設定の修正: 相対パス/絶対パスを適切に処理
        # All stores use the same URI (consolidated storage)
        uri_from_config = lancedb_config.get("uri", "lancedb_default")
        
        if os.path.isabs(uri_from_config):
            # 絶対パスの場合はそのまま使用
            uri = uri_from_config
        elif uri_from_config.startswith("./"):
            # 相対パス（./で始まる）の場合は、output_dirを基準にする
            output_dir = config.get("output_dir", ".")
            uri = os.path.join(output_dir, uri_from_config[2:])  # "./"を取り除く
        else:
            # その他の場合はoutput_dirと結合
            output_dir = config.get("output_dir", ".")
            uri = os.path.join(output_dir, uri_from_config)
        
        # Check if table already exists and use appropriate mode
        table_path = os.path.join(uri, f"{table_name}.lance")

        # Always use "create" mode to avoid schema issues
        # LanceDB will handle existing tables appropriately
        try:
            return LanceDBVectorStore(
                uri=uri,
                table_name=table_name,  # Use table name from constants
                mode="create",  # Always create to ensure schema consistency
            )
        except Exception as e:
            # If creation fails due to existing table with different schema,
            # try to remove the problematic table and recreate
            if "schema" in str(e).lower() or "field" in str(e).lower():
                print(f"Schema mismatch detected for table {table_name}. Recreating table...")
                import shutil
                if os.path.exists(table_path):
                    shutil.rmtree(table_path)
                return LanceDBVectorStore(
                    uri=uri,
                    table_name=table_name,
                    mode="create",
                )
            else:
                raise e
    return None # Falls back to default in-memory # Falls back to default in-memory # Falls back to default in-memory # Falls back to default in-memory # Falls back to default in-memory

def get_index(storage_dir, vector_store=None, index_type="main"):
    """Loads the index from storage if it exists, otherwise creates a new one."""
    if index_type == "main" and vector_store is None and os.path.exists(storage_dir):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context)
    elif vector_store is not None:
        return VectorStoreIndex.from_vector_store(vector_store)
    return None