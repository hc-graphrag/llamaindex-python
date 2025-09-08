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
        
        # パス設定の修正: 相対パス/絶対パスを適切に処理
        uri_from_config = lancedb_config.get("uri", "lancedb_default")
        
        if os.path.isabs(uri_from_config):
            # 絶対パスの場合はそのまま使用
            uri = uri_from_config
        elif uri_from_config.startswith("./"):
            # 相対パス（./で始まる）の場合は、output_dirを基準にする
            output_dir = config.get("output_dir", ".")
            uri = os.path.join(output_dir, uri_from_config[2:])  # "./を取り除く
        else:
            # その他の場合はoutput_dirと結合
            output_dir = config.get("output_dir", ".")
            uri = os.path.join(output_dir, uri_from_config)
        
        return LanceDBVectorStore(
            uri=uri,
            table_name=lancedb_config.get("table_name", "vectors"),
            mode="overwrite", # Consider changing to "append" if you want to add to existing table
        )
    return None # Falls back to default in-memory # Falls back to default in-memory

def get_index(storage_dir, vector_store=None, index_type="main"):
    """Loads the index from storage if it exists, otherwise creates a new one."""
    if index_type == "main" and vector_store is None and os.path.exists(storage_dir):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context)
    elif vector_store is not None:
        return VectorStoreIndex.from_vector_store(vector_store)
    return None