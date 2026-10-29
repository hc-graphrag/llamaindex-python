"""LlamaIndex vector store backed by embedded AlopexDB HNSW."""

from __future__ import annotations

import json
from threading import RLock
from typing import Any, Sequence

import numpy as np
from alopex import AlopexError, HnswConfig, Metric, TxnMode
from llama_index.core import Settings
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pydantic import PrivateAttr

from .alopex_db import get_database


_index_lock = RLock()


class AlopexVectorStore(BasePydanticVectorStore):
    """Store LlamaIndex nodes in SQL and embeddings in AlopexDB HNSW."""

    stores_text: bool = True
    output_dir: str
    index_name: str
    _database: Any = PrivateAttr()

    def __init__(self, output_dir: str, index_name: str) -> None:
        super().__init__(output_dir=output_dir, index_name=index_name)
        self._database = get_database(output_dir)

    @property
    def client(self) -> Any:
        return self._database

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        if not nodes:
            return []

        embeddings = np.asarray(
            [node.get_embedding() for node in nodes], dtype=np.float32
        )
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise ValueError("AlopexDB requires non-empty embeddings of one dimension")
        self._ensure_index(embeddings.shape[1])

        node_ids = [node.node_id for node in nodes]
        sql_rows = []
        for node in nodes:
            payload = json.dumps(
                node_to_metadata_dict(node, remove_text=False),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            sql_rows.append(
                [
                    self._storage_key(node.node_id),
                    self.index_name,
                    node.node_id,
                    node.ref_doc_id,
                    payload,
                ]
            )

        with self._database.begin(TxnMode.READ_WRITE) as transaction:
            transaction.upsert_to_hnsw_batch(
                self.index_name,
                [node_id.encode() for node_id in node_ids],
                embeddings,
            )
            transaction.execute_many(
                """
                INSERT INTO graphrag_vector_nodes (
                    storage_key, index_name, node_id, ref_doc_id, payload
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET
                    index_name = excluded.index_name,
                    node_id = excluded.node_id,
                    ref_doc_id = excluded.ref_doc_id,
                    payload = excluded.payload
                """,
                sql_rows,
            )
            transaction.commit()
        return node_ids

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        if filters is not None:
            raise NotImplementedError("AlopexDB metadata filters are not implemented")
        sql = "SELECT payload FROM graphrag_vector_nodes WHERE index_name = ?"
        params: list[Any] = [self.index_name]
        if node_ids:
            sql += f" AND node_id IN ({','.join('?' for _ in node_ids)})"
            params.extend(node_ids)
        return [
            metadata_dict_to_node(json.loads(row["payload"]))
            for row in self._database.execute_sql(sql, params)
        ]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        rows = self._database.execute_sql(
            """
            SELECT node_id FROM graphrag_vector_nodes
            WHERE index_name = ? AND ref_doc_id = ?
            """,
            [self.index_name, ref_doc_id],
        )
        self.delete_nodes([row["node_id"] for row in rows])

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        if filters is not None:
            raise NotImplementedError("AlopexDB metadata filters are not implemented")
        if not node_ids:
            return
        with self._database.begin(TxnMode.READ_WRITE) as transaction:
            for node_id in node_ids:
                transaction.delete_from_hnsw(self.index_name, node_id.encode())
            placeholders = ",".join("?" for _ in node_ids)
            transaction.execute_sql(
                f"""
                DELETE FROM graphrag_vector_nodes
                WHERE index_name = ? AND node_id IN ({placeholders})
                """,
                [self.index_name, *node_ids],
            )
            transaction.commit()

    def clear(self) -> None:
        dimension = self._dimension()
        if dimension is None:
            return
        self._database.drop_hnsw_index(self.index_name)
        self._database.create_hnsw_index(
            self.index_name, HnswConfig(dimension, metric=Metric.COSINE)
        )
        self._database.execute_sql(
            "DELETE FROM graphrag_vector_nodes WHERE index_name = ?",
            [self.index_name],
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(f"AlopexDB does not support query mode {query.mode}")
        if query.filters is not None:
            raise NotImplementedError("AlopexDB metadata filters are not implemented")
        dimension = self._dimension()
        if dimension is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        embedding = query.query_embedding
        if embedding is None and query.query_str:
            embedding = Settings.embed_model.get_query_embedding(query.query_str)
        if embedding is None:
            raise ValueError("AlopexDB vector queries require an embedding or query text")
        if len(embedding) != dimension:
            raise ValueError(
                f"Embedding dimension {len(embedding)} does not match {dimension}"
            )

        constrained = query.node_ids is not None or query.doc_ids is not None
        limit = self._database.get_hnsw_stats(self.index_name).node_count if constrained else query.similarity_top_k
        if limit == 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        results, _ = self._database.search_hnsw(
            self.index_name,
            np.asarray(embedding, dtype=np.float32),
            limit,
        )

        ids = [result.key.decode() for result in results]
        rows = self._node_rows(ids)
        allowed_nodes = set(query.node_ids or ids)
        allowed_docs = set(query.doc_ids or [])
        selected = [
            (result, rows[result.key.decode()])
            for result in results
            if result.key.decode() in rows
            and result.key.decode() in allowed_nodes
            and (not allowed_docs or rows[result.key.decode()]["ref_doc_id"] in allowed_docs)
        ][: query.similarity_top_k]

        return VectorStoreQueryResult(
            nodes=[metadata_dict_to_node(json.loads(row["payload"])) for _, row in selected],
            similarities=[1.0 - float(result.score) for result, _ in selected],
            ids=[result.key.decode() for result, _ in selected],
        )

    def _node_rows(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not node_ids:
            return {}
        placeholders = ",".join("?" for _ in node_ids)
        rows = self._database.execute_sql(
            f"""
            SELECT node_id, ref_doc_id, payload FROM graphrag_vector_nodes
            WHERE index_name = ? AND node_id IN ({placeholders})
            """,
            [self.index_name, *node_ids],
        )
        return {row["node_id"]: row for row in rows}

    def _dimension(self) -> int | None:
        rows = self._database.execute_sql(
            "SELECT dimension FROM graphrag_vector_indexes WHERE index_name = ?",
            [self.index_name],
        )
        return int(rows[0]["dimension"]) if rows else None

    def _ensure_index(self, dimension: int) -> None:
        with _index_lock:
            existing = self._dimension()
            if existing is not None and existing != dimension:
                raise ValueError(
                    f"AlopexDB index {self.index_name} uses dimension {existing}, not {dimension}"
                )
            if existing is None:
                self._database.execute_sql(
                    """
                    INSERT INTO graphrag_vector_indexes VALUES (?, ?)
                    ON CONFLICT DO UPDATE SET dimension = excluded.dimension
                    """,
                    [self.index_name, dimension],
                )
            try:
                self._database.get_hnsw_stats(self.index_name)
            except AlopexError as error:
                if "index not found" not in str(error):
                    raise
                self._database.create_hnsw_index(
                    self.index_name,
                    HnswConfig(dimension, metric=Metric.COSINE),
                )

    def _storage_key(self, node_id: str) -> str:
        return json.dumps([self.index_name, node_id], separators=(",", ":"))
