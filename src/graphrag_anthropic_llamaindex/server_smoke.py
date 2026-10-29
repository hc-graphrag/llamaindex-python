"""Live SQL VECTOR/HNSW smoke test from LlamaIndex container to AlopexDB."""

from __future__ import annotations

import json
import os
import uuid

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from .alopex_db import close_database, get_database
from .alopex_server_vector_store import AlopexServerVectorStore


def main() -> None:
    target = os.environ.get("ALOPEX_SERVER_URL", "http://alopexdb:8080")
    index_name = f"smoke_{uuid.uuid4().hex}"
    database = get_database(target)
    store = AlopexServerVectorStore(database, index_name)
    table_name = f"graphrag_server_{index_name}"
    try:
        store.add(
            [
                TextNode(id_="alpha", text="alpha", embedding=[1.0, 0.0]),
                TextNode(id_="beta", text="beta", embedding=[0.0, 1.0]),
            ]
        )
        result = store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0], similarity_top_k=2)
        )
        if result.ids != ["alpha", "beta"]:
            raise RuntimeError(f"Unexpected vector result: {result.ids!r}")
        print(json.dumps({"status": "ok", "ids": result.ids}))
    finally:
        database.execute_sql(f"DROP TABLE IF EXISTS {table_name}")
        close_database(target)


if __name__ == "__main__":
    main()
