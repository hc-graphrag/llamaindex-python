import pandas as pd
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from graphrag_anthropic_llamaindex.alopex_db import (
    close_database,
    get_database,
    get_database_path,
)
from graphrag_anthropic_llamaindex.db_manager import (
    load_community_db,
    load_community_summaries_db,
    load_entities_db,
    load_processed_files_db,
    load_relationships_db,
    load_text_units_db,
    save_community_db,
    save_community_summaries_db,
    save_entities_db,
    save_processed_files_db,
    save_relationships_db,
    save_text_units_db,
)
from graphrag_anthropic_llamaindex.vector_store_manager import get_vector_store


@pytest.fixture
def output_dir(tmp_path):
    path = str(tmp_path / "output")
    yield path
    close_database(path)


def test_sql_data_persists_and_supports_filtered_lookup(output_dir):
    save_entities_db(
        pd.DataFrame(
            [
                {"name": "Alopex", "type": "Database", "tags": ["sql", "vector"]},
                {"name": "GraphRAG", "type": "Application", "tags": ["rag"]},
            ]
        ),
        output_dir,
    )

    assert load_entities_db(output_dir, name="Alopex").to_dict("records") == [
        {"name": "Alopex", "type": "Database", "tags": ["sql", "vector"]}
    ]

    close_database(output_dir)
    assert load_entities_db(output_dir, name="GraphRAG").iloc[0]["type"] == "Application"


def test_all_structured_datasets_share_one_database_and_reopen(output_dir):
    save_processed_files_db(
        pd.DataFrame([{"filepath": "doc.txt", "hash": "abc"}]), output_dir
    )
    save_relationships_db(
        pd.DataFrame([{"source": "A", "target": "B", "type": "uses"}]),
        output_dir,
    )
    save_community_db(
        pd.DataFrame(
            [{"level": 0, "cluster_id": 1, "parent_cluster": None, "nodes": ["A"]}]
        ),
        output_dir,
    )
    save_community_summaries_db(
        pd.DataFrame([{"community_id": 1, "summary": "summary", "key_entities": ["A"]}]),
        output_dir,
    )
    save_text_units_db(
        pd.DataFrame([{"id": "text-1", "text": "content", "entity_ids": ["A"]}]),
        output_dir,
    )

    datasets = {
        row["dataset"]
        for row in get_database(output_dir).execute_sql(
            "SELECT DISTINCT dataset FROM graphrag_records"
        )
    }
    assert datasets == {
        "processed_files",
        "relationships",
        "communities",
        "community_summaries",
        "text_units",
    }

    close_database(output_dir)
    assert load_processed_files_db(output_dir).iloc[0]["hash"] == "abc"
    assert load_relationships_db(output_dir, entity_id="A").iloc[0]["target"] == "B"
    assert load_community_db(output_dir).iloc[0]["nodes"] == ["A"]
    assert load_community_summaries_db(output_dir, community_id=1).iloc[0]["summary"] == "summary"
    assert load_text_units_db(output_dir).iloc[0]["entity_ids"] == ["A"]


def test_vector_store_uses_three_indexes_in_one_embedded_database(output_dir):
    config = {"output_dir": output_dir}
    main = get_vector_store(config, "main")
    entity = get_vector_store(config, "entity")
    community = get_vector_store(config, "community")

    assert main.client is entity.client is community.client
    assert main.index_name != entity.index_name != community.index_name
    assert get_database_path(output_dir).endswith("/alopexdb")

    main.add([TextNode(id_="shared", text="main", embedding=[1.0, 0.0])])
    entity.add([TextNode(id_="shared", text="entity", embedding=[0.0, 1.0])])
    community.add([TextNode(id_="shared", text="community", embedding=[1.0, 1.0])])

    def top_text(store, embedding):
        result = store.query(
            VectorStoreQuery(query_embedding=embedding, similarity_top_k=1)
        )
        return result.nodes[0].text

    assert top_text(main, [1.0, 0.0]) == "main"
    assert top_text(entity, [0.0, 1.0]) == "entity"
    assert top_text(community, [1.0, 1.0]) == "community"


def test_vector_add_query_delete_and_reopen(output_dir):
    config = {"output_dir": output_dir}
    store = get_vector_store(config, "main")
    node = TextNode(
        id_="node-1",
        text="embedded sql and vector search",
        metadata={"kind": "article"},
        embedding=[1.0, 0.0],
    )

    assert store.add([node]) == ["node-1"]
    result = store.query(VectorStoreQuery(query_embedding=[1.0, 0.0], similarity_top_k=1))
    assert result.ids == ["node-1"]
    assert result.nodes[0].text == node.text
    assert result.similarities == pytest.approx([1.0])

    close_database(output_dir)
    reopened = get_vector_store(config, "main")
    assert reopened.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0], similarity_top_k=1)
    ).ids == ["node-1"]

    reopened.delete_nodes(["node-1"])
    assert reopened.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0], similarity_top_k=1)
    ).ids == []


def test_vector_store_rejects_dimension_changes(output_dir):
    store = get_vector_store({"output_dir": output_dir}, "main")
    store.add([TextNode(id_="a", text="a", embedding=[1.0, 0.0])])

    with pytest.raises(ValueError, match="uses dimension 2"):
        store.add([TextNode(id_="b", text="b", embedding=[1.0, 0.0, 0.0])])
