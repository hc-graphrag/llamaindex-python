"""Shared embedded and server AlopexDB connections and schema."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from threading import RLock
from typing import Any

from alopex import Database, DatabaseLike, connect


DATABASE_DIRNAME = "alopexdb"

_databases: dict[str, DatabaseLike] = {}
_lock = RLock()


def is_server_target(target: str) -> bool:
    """Return whether a storage target is an AlopexDB HTTP server."""
    return target.startswith(("http://", "https://"))


def get_database_path(output_dir: str) -> str:
    """Return the single embedded database path for a GraphRAG output directory."""
    return str((Path(output_dir).resolve() / DATABASE_DIRNAME))


def get_database(output_dir: str) -> DatabaseLike:
    """Open or reuse an embedded path or AlopexDB server URL."""
    path = output_dir if is_server_target(output_dir) else get_database_path(output_dir)
    with _lock:
        if path not in _databases:
            if is_server_target(output_dir):
                database = connect(
                    output_dir,
                    api_key=os.environ.get("ALOPEX_API_KEY"),
                    insecure=os.environ.get("ALOPEX_SERVER_INSECURE", "false").lower()
                    in {"1", "true", "yes"},
                )
            else:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                database = Database.open(path)
            _ensure_schema(database)
            _databases[path] = database
        return _databases[path]


def close_database(output_dir: str) -> None:
    """Close one cached database so the directory can be reopened safely."""
    path = output_dir if is_server_target(output_dir) else get_database_path(output_dir)
    with _lock:
        database = _databases.pop(path, None)
        if database is not None:
            database.close()


def close_all_databases() -> None:
    """Close all embedded handles held by this process."""
    with _lock:
        databases = list(_databases.values())
        _databases.clear()
    for database in databases:
        database.close()


def _ensure_schema(database: DatabaseLike) -> None:
    database.execute_sql(
        """
        CREATE TABLE IF NOT EXISTS graphrag_records (
            row_key TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            position BIGINT NOT NULL,
            record_id TEXT,
            name TEXT,
            record_type TEXT,
            source TEXT,
            target TEXT,
            community_id TEXT,
            filepath TEXT,
            file_hash TEXT,
            payload TEXT NOT NULL
        )
        """
    )
    database.execute_sql(
        """
        CREATE TABLE IF NOT EXISTS graphrag_vector_nodes (
            storage_key TEXT PRIMARY KEY,
            index_name TEXT NOT NULL,
            node_id TEXT NOT NULL,
            ref_doc_id TEXT,
            payload TEXT NOT NULL
        )
        """
    )
    database.execute_sql(
        """
        CREATE TABLE IF NOT EXISTS graphrag_vector_indexes (
            index_name TEXT PRIMARY KEY,
            dimension BIGINT NOT NULL
        )
        """
    )


atexit.register(close_all_databases)
