import pytest
from alopex import Database
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from graphrag_anthropic_llamaindex.alopex_server_vector_store import (
    AlopexServerVectorStore,
)


@pytest.fixture
def database():
    client = Database.open_in_memory()
    yield client
    client.close()


def test_server_store_uses_sql_vector_and_hnsw(database):
    store = AlopexServerVectorStore(database, "main_vectors")
    nodes = [
        TextNode(id_="a", text="alpha", embedding=[1.0, 0.0]),
        TextNode(id_="b", text="beta", embedding=[0.0, 1.0]),
    ]

    assert store.add(nodes) == ["a", "b"]
    result = store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0], similarity_top_k=2)
    )
    assert result.ids == ["a", "b"]
    assert [node.text for node in result.nodes] == ["alpha", "beta"]
    assert result.similarities == pytest.approx([1.0, 0.0])
    assert database.execute_sql(
        """
        SELECT index_type FROM information_schema.indexes
        WHERE table_name = 'graphrag_server_main_vectors'
          AND index_name = 'graphrag_server_main_vectors_embedding_hnsw'
        """
    ) == [{"index_type": "HNSW"}]

    store.delete_nodes(["a"])
    assert [node.node_id for node in store.get_nodes()] == ["b"]


def test_server_store_keeps_named_indexes_separate(database):
    main = AlopexServerVectorStore(database, "main_vectors")
    entity = AlopexServerVectorStore(database, "entity_vectors")
    main.add([TextNode(id_="shared", text="main", embedding=[1.0, 0.0])])
    entity.add([TextNode(id_="shared", text="entity", embedding=[0.0, 1.0])])

    assert main.get_nodes(["shared"])[0].text == "main"
    assert entity.get_nodes(["shared"])[0].text == "entity"


def test_server_store_rejects_dimension_changes(database):
    store = AlopexServerVectorStore(database, "main_vectors")
    store.add([TextNode(id_="a", text="a", embedding=[1.0, 0.0])])

    with pytest.raises(ValueError, match="uses dimension 2"):
        store.add([TextNode(id_="b", text="b", embedding=[1.0, 0.0, 0.0])])
