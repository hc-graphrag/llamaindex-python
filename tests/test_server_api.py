from unittest.mock import patch

from graphrag_anthropic_llamaindex.server import (
    DocumentRequest,
    SearchRequest,
    add_document as add_document_route,
    healthz,
    search as search_route,
)


def test_health_endpoint_reports_alopexdb_connection():
    with patch(
        "graphrag_anthropic_llamaindex.server.service.health",
        return_value={"status": "ok", "alopexdb": "ok"},
    ):
        response = healthz()

    assert response == {"status": "ok", "alopexdb": "ok"}


def test_document_endpoint_delegates_to_server_runtime():
    with patch(
        "graphrag_anthropic_llamaindex.server.service.add_document",
        return_value={"status": "indexed", "filename": "doc.txt"},
    ) as add_document_mock:
        response = add_document_route(
            DocumentRequest(filename="doc.txt", content="hello")
        )

    assert response["status"] == "indexed"
    assert add_document_mock.call_args.args[0].filename == "doc.txt"


def test_search_endpoint_returns_llamaindex_results():
    with patch(
        "graphrag_anthropic_llamaindex.server.service.search",
        return_value={"results": [{"text": "answer", "score": 1.0}]},
    ) as search_mock:
        response = search_route(SearchRequest(query="question", mode="global"))

    assert response["results"][0]["text"] == "answer"
    assert search_mock.call_args.args[0].query == "question"
