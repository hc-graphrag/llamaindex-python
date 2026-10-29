"""gradio_app (Gradio 6) のテスト。外部ネットワーク・実LLMは使わない。"""

import os
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio_app  # noqa: E402
from src.graphrag_anthropic_llamaindex.claude_code_llm import ClaudeCodeLLM  # noqa: E402


@pytest.fixture
def chat_app():
    app = gradio_app.GraphRAGApp()
    app.is_initialized = True
    app.config = {}
    app.vector_stores = {}
    return app


def _search_chat(app, message="質問", history=None):
    return app.search_chat(message, history if history is not None else [], "global", "multiple paragraphs", "markdown", 0)


def test_create_interface_builds():
    assert gradio_app.create_interface() is not None


def test_not_initialized_appends_messages_format():
    app = gradio_app.GraphRAGApp()
    _, history = _search_chat(app, "こんにちは")
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "こんにちは"
    assert "設定が初期化されていません" in history[1]["content"]


def test_empty_message_appends_messages_format(chat_app):
    _, history = _search_chat(chat_app, "   ")
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert "検索クエリを入力してください" in history[1]["content"]


def test_search_result_is_appended_as_assistant_message(chat_app):
    node = NodeWithScore(node=TextNode(text="回答本文"), score=1.0)
    router = MagicMock()
    router._retrieve.return_value = [node]
    with patch.object(gradio_app, "SearchModeRouter", return_value=router):
        cleared, history = _search_chat(chat_app, "アリスは誰?", [{"role": "user", "content": "前"}, {"role": "assistant", "content": "答"}])
    assert cleared == ""
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[-2]["content"] == "アリスは誰?"
    assert "回答本文" in history[-1]["content"]


def test_search_failure_is_reported_as_assistant_message(chat_app):
    with patch.object(gradio_app, "SearchModeRouter", side_effect=RuntimeError("boom")):
        _, history = _search_chat(chat_app)
    assert history[-1]["role"] == "assistant" and "boom" in history[-1]["content"]


def test_initialize_config_claude_code_needs_no_api_key(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "llm_provider: claude_code\nclaude_code:\n  model: sonnet\noutput_dir: ./out\ninput_dir: ./in\n"
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GRAPHRAG_CONFIG_BASE", str(tmp_path))
    monkeypatch.setattr(Settings, "_llm", None, raising=False)
    monkeypatch.setattr(Settings, "_embed_model", None, raising=False)
    monkeypatch.setattr(Settings, "_node_parser", None, raising=False)

    app = gradio_app.GraphRAGApp()
    with patch.object(gradio_app, "get_vector_store", return_value=MagicMock()), \
         patch.object(gradio_app, "HuggingFaceEmbedding", return_value=MockEmbedding(embed_dim=8)):
        status = app.initialize_config(str(config))

    assert status.startswith("✅"), status
    assert isinstance(Settings._llm, ClaudeCodeLLM)
    assert Settings._llm.model == "sonnet"


def test_initialize_config_rejects_path_outside_base(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("llm_provider: claude_code\n")
    monkeypatch.setenv("GRAPHRAG_CONFIG_BASE", str(base))

    app = gradio_app.GraphRAGApp()
    status = app.initialize_config(str(outside))

    assert status.startswith("❌"), status
    assert app.config is None


def test_chat_round_trip_through_running_server(chat_app):
    """実際に Gradio サーバーを起動し、クライアント経由で Chatbot の messages 形式が往復できる。"""
    from gradio_client import Client

    node = NodeWithScore(node=TextNode(text="サーバー越しの回答"), score=1.0)
    router = MagicMock()
    router._retrieve.return_value = [node]

    interface = gradio_app.create_interface()
    with patch.object(gradio_app, "app", chat_app), patch.object(gradio_app, "SearchModeRouter", return_value=router):
        interface.launch(server_name="127.0.0.1", server_port=None, prevent_thread_lock=True, quiet=True)
        try:
            client = Client(interface.local_url, verbose=False)
            result = client.predict(
                "アリスは誰?", [], "global", "multiple paragraphs", "markdown", 0, api_name="/send_message"
            )
        finally:
            interface.close()

    cleared, history = result
    assert cleared == ""
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert "サーバー越しの回答" in str(history[1]["content"])
