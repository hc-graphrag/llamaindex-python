"""GraphRAG structured storage backed by embedded AlopexDB SQL."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
from alopex import RemoteDatabase, TxnMode

from .alopex_db import get_database


_DATASET_COLUMNS = {
    "processed_files": ["filepath", "hash"],
    "entities": ["name", "type"],
    "relationships": ["source", "target", "type", "description"],
    "communities": ["level", "cluster_id", "parent_cluster", "nodes"],
    "community_summaries": ["community_id", "summary", "key_entities"],
    "text_units": ["id", "text", "entity_ids"],
}


def calculate_file_hash(filepath: str) -> str:
    """Calculate a file's SHA-256 hash."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _save_dataset(dataframe: pd.DataFrame, output_dir: str, dataset: str) -> None:
    rows = []
    for position, record in enumerate(dataframe.to_dict("records")):
        record = _json_value(record)
        rows.append(
            [
                f"{dataset}:{position}",
                dataset,
                position,
                _text(record.get("id")),
                _text(record.get("name")),
                _text(record.get("type")),
                _text(record.get("source")),
                _text(record.get("target")),
                _text(record.get("community_id")),
                _text(record.get("filepath")),
                _text(record.get("hash")),
                json.dumps(record, ensure_ascii=False, separators=(",", ":")),
            ]
        )

    database = get_database(output_dir)
    with database.begin(TxnMode.READ_WRITE) as transaction:
        transaction.execute_sql(
            "DELETE FROM graphrag_records WHERE dataset = ?", [dataset]
        )
        if rows:
            insert_sql = """
            INSERT INTO graphrag_records (
                row_key, dataset, position, record_id, name, record_type,
                source, target, community_id, filepath, file_hash, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            if isinstance(database, RemoteDatabase):
                for row in rows:
                    transaction.execute_sql(insert_sql, row)
            else:
                transaction.execute_many(insert_sql, rows)
        transaction.commit()


def _load_dataset(
    output_dir: str,
    dataset: str,
    condition: str | None = None,
    params: list[Any] | None = None,
) -> pd.DataFrame:
    sql = "SELECT payload FROM graphrag_records WHERE dataset = ?"
    values: list[Any] = [dataset]
    if condition:
        sql += f" AND ({condition})"
        values.extend(params or [])
    sql += " ORDER BY position"
    records = [
        json.loads(row["payload"])
        for row in get_database(output_dir).execute_sql(sql, values)
    ]
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(columns=_DATASET_COLUMNS[dataset])


def load_processed_files_db(output_dir: str) -> pd.DataFrame:
    return _load_dataset(output_dir, "processed_files")


def save_processed_files_db(dataframe: pd.DataFrame, output_dir: str) -> None:
    _save_dataset(dataframe, output_dir, "processed_files")


def load_entities_db(
    output_dir: str,
    *,
    name: str | None = None,
    entity_id: str | None = None,
) -> pd.DataFrame:
    if name is not None:
        return _load_dataset(output_dir, "entities", "name = ?", [name])
    if entity_id is not None:
        return _load_dataset(output_dir, "entities", "record_id = ?", [entity_id])
    return _load_dataset(output_dir, "entities")


def save_entities_db(dataframe: pd.DataFrame, output_dir: str) -> None:
    _save_dataset(dataframe, output_dir, "entities")


def load_relationships_db(
    output_dir: str, *, entity_id: str | None = None
) -> pd.DataFrame:
    if entity_id is not None:
        return _load_dataset(
            output_dir,
            "relationships",
            "source = ? OR target = ?",
            [entity_id, entity_id],
        )
    return _load_dataset(output_dir, "relationships")


def save_relationships_db(dataframe: pd.DataFrame, output_dir: str) -> None:
    _save_dataset(dataframe, output_dir, "relationships")


def load_community_db(output_dir: str) -> pd.DataFrame:
    return _load_dataset(output_dir, "communities")


def save_community_db(dataframe: pd.DataFrame, output_dir: str) -> None:
    _save_dataset(dataframe, output_dir, "communities")


def load_community_summaries_db(
    output_dir: str, *, community_id: str | int | None = None
) -> pd.DataFrame:
    if community_id is not None:
        return _load_dataset(
            output_dir, "community_summaries", "community_id = ?", [str(community_id)]
        )
    return _load_dataset(output_dir, "community_summaries")


def save_community_summaries_db(dataframe: pd.DataFrame, output_dir: str) -> None:
    _save_dataset(dataframe, output_dir, "community_summaries")


def load_text_units_db(output_dir: str) -> pd.DataFrame:
    return _load_dataset(output_dir, "text_units")


def save_text_units_db(dataframe: pd.DataFrame, output_dir: str) -> None:
    _save_dataset(dataframe, output_dir, "text_units")
