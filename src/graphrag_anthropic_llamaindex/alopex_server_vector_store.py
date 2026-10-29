"""LlamaIndex vector store backed by an AlopexDB server over SQL."""

from __future__ import annotations

import json
import re
from threading import RLock
from typing import Any, Sequence

from alopex import DatabaseLike, TxnMode
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


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_VECTOR_TYPE = re.compile(r"^VECTOR\((\d+),", re.IGNORECASE)
_schema_lock = RLock()


class AlopexServerVectorStore(BasePydanticVectorStore):
    """Store LlamaIndex nodes in an AlopexDB server SQL vector table."""

    stores_text: bool = True
    index_name: str
    _database: DatabaseLike = PrivateAttr()
    _table_name: str = PrivateAttr()
    _hnsw_name: str = PrivateAttr()

    def __init__(self, database: DatabaseLike, index_name: str) -> None:
        if not _IDENTIFIER.fullmatch(index_name):
            raise ValueError(f"Invalid AlopexDB index name: {index_name!r}")
        super().__init__(index_name=index_name)
        self._database = database
        self._table_name = f"graphrag_server_{index_name}"
        self._hnsw_name = f"{self._table_name}_embedding_hnsw"

    @property
    def client(self) -> DatabaseLike:
        return self._database

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        if not nodes:
            return []
        embeddings = [node.get_embedding() for node in nodes]
        dimension = len(embeddings[0])
        if dimension == 0 or any(len(item) != dimension for item in embeddings):
            raise ValueError("AlopexDB requires non-empty embeddings of one dimension")
        self._ensure_table(dimension)

        with self._database.begin(TxnMode.READ_WRITE) as transaction:
            for node, embedding in zip(nodes, embeddings):
                payload = json.dumps(
                    node_to_metadata_dict(node, remove_text=False),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                transaction.execute_sql(
                    f"""
                    INSERT INTO {self._table_name} (
                        node_id, ref_doc_id, payload, embedding
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT DO UPDATE SET
                        ref_doc_id = excluded.ref_doc_id,
                        payload = excluded.payload,
                        embedding = excluded.embedding
                    """,
                    [node.node_id, node.ref_doc_id, payload, embedding],
                )
            transaction.commit()
        return [node.node_id for node in nodes]

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        if filters is not None:
            raise NotImplementedError("AlopexDB metadata filters are not implemented")
        if self._dimension() is None:
            return []
        sql = f"SELECT payload FROM {self._table_name}"
        params: list[Any] = []
        if node_ids:
            sql += f" WHERE node_id IN ({','.join('?' for _ in node_ids)})"
            params.extend(node_ids)
        return [
            metadata_dict_to_node(json.loads(row["payload"]))
            for row in self._database.execute_sql(sql, params)
        ]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        if self._dimension() is not None:
            self._database.execute_sql(
                f"DELETE FROM {self._table_name} WHERE ref_doc_id = ?",
                [ref_doc_id],
            )

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        if filters is not None:
            raise NotImplementedError("AlopexDB metadata filters are not implemented")
        if not node_ids or self._dimension() is None:
            return
        placeholders = ",".join("?" for _ in node_ids)
        self._database.execute_sql(
            f"DELETE FROM {self._table_name} WHERE node_id IN ({placeholders})",
            node_ids,
        )

    def clear(self) -> None:
        if self._dimension() is not None:
            self._database.execute_sql(f"DELETE FROM {self._table_name}")

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

        conditions: list[str] = []
        condition_params: list[Any] = []
        if query.node_ids:
            conditions.append(f"node_id IN ({','.join('?' for _ in query.node_ids)})")
            condition_params.extend(query.node_ids)
        if query.doc_ids:
            conditions.append(f"ref_doc_id IN ({','.join('?' for _ in query.doc_ids)})")
            condition_params.extend(query.doc_ids)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._database.execute_sql(
            f"""
            SELECT node_id, payload,
                   vector_distance(embedding, ?, 'cosine') AS distance
            FROM {self._table_name}
            {where}
            ORDER BY vector_distance(embedding, ?, 'cosine') ASC
            LIMIT ?
            """,
            [embedding, *condition_params, embedding, query.similarity_top_k],
        )
        return VectorStoreQueryResult(
            nodes=[metadata_dict_to_node(json.loads(row["payload"])) for row in rows],
            similarities=[1.0 - float(row["distance"]) for row in rows],
            ids=[row["node_id"] for row in rows],
        )

    def _dimension(self) -> int | None:
        rows = self._database.execute_sql(
            """
            SELECT data_type FROM information_schema.columns
            WHERE table_name = ? AND column_name = 'embedding'
            """,
            [self._table_name],
        )
        if not rows:
            return None
        match = _VECTOR_TYPE.match(rows[0]["data_type"])
        if match is None:
            raise RuntimeError(f"Unexpected vector type: {rows[0]['data_type']}")
        return int(match.group(1))

    def _ensure_table(self, dimension: int) -> None:
        with _schema_lock:
            existing = self._dimension()
            if existing is not None and existing != dimension:
                raise ValueError(
                    f"AlopexDB index {self.index_name} uses dimension "
                    f"{existing}, not {dimension}"
                )
            if existing is None:
                self._database.execute_sql(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        node_id TEXT PRIMARY KEY,
                        ref_doc_id TEXT,
                        payload TEXT NOT NULL,
                        embedding VECTOR({dimension}, COSINE) NOT NULL
                    )
                    """
                )
                self._database.execute_sql(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self._hnsw_name}
                    ON {self._table_name} (embedding) USING HNSW
                    """
                )
