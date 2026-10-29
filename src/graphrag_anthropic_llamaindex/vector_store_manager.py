"""AlopexDB vector store construction."""

from __future__ import annotations

from llama_index.core import VectorStoreIndex

from .alopex_db import get_database, is_server_target
from .alopex_server_vector_store import AlopexServerVectorStore
from .alopex_vector_store import AlopexVectorStore


VECTOR_STORE_INDEX_NAMES = {
    "main": "main_vectors",
    "entity": "entity_vectors",
    "community": "community_vectors",
}


def get_vector_store(
    config: dict, store_type: str = "main"
) -> AlopexVectorStore | AlopexServerVectorStore | None:
    """Return one logical HNSW index from embedded or server AlopexDB."""
    index_name = VECTOR_STORE_INDEX_NAMES.get(store_type)
    if index_name is None:
        return None
    target = config.get("output_dir", "./graphrag_output")
    if is_server_target(target):
        return AlopexServerVectorStore(
            database=get_database(target),
            index_name=index_name,
        )
    return AlopexVectorStore(
        output_dir=target,
        index_name=index_name,
    )


def get_index(storage_dir=None, vector_store=None, index_type: str = "main"):
    """Build a LlamaIndex facade over an AlopexDB vector store."""
    if vector_store is None:
        return None
    return VectorStoreIndex.from_vector_store(vector_store)
