"""ClaudeCodeLLM のテスト (claude CLI は呼ばず query をモックする)"""

import asyncio
from unittest.mock import patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from llama_index.core.base.llms.types import ChatMessage

from graphrag_anthropic_llamaindex import claude_code_llm
from graphrag_anthropic_llamaindex.claude_code_llm import ClaudeCodeLLM


def _result(text="done", is_error=False):
    return ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1, is_error=is_error,
        num_turns=1, session_id="s", result=text,
    )


def _fake_query(messages, captured=None):
    async def fake(*, prompt, options):
        if captured is not None:
            captured.update(prompt=prompt, options=options)
        for m in messages:
            yield m
    return fake


def _assistant(text):
    return AssistantMessage(content=[TextBlock(text=text)], model="m")


def test_complete_returns_result_text_and_disables_tools():
    captured = {}
    fake = _fake_query([_assistant("ignored"), _result("hello")], captured)
    with patch.object(claude_code_llm, "query", fake):
        out = ClaudeCodeLLM(model="sonnet").complete("hi")
    assert out.text == "hello"
    assert captured["prompt"] == "hi"
    options = captured["options"]
    assert options.model == "sonnet"
    assert options.max_turns == 1
    assert "Bash" in options.disallowed_tools and "Write" in options.disallowed_tools
    assert options.setting_sources == []


def test_chat_goes_through_complete():
    fake = _fake_query([_result("tokyo")])
    with patch.object(claude_code_llm, "query", fake):
        resp = ClaudeCodeLLM().chat([ChatMessage(role="user", content="capital?")])
    assert resp.message.content == "tokyo"


def test_falls_back_to_assistant_text_when_result_is_none():
    fake = _fake_query([_assistant("a"), _assistant("b"), _result(None)])
    with patch.object(claude_code_llm, "query", fake):
        assert ClaudeCodeLLM().complete("x").text == "ab"


def test_error_result_raises():
    fake = _fake_query([_result("boom", is_error=True)])
    with patch.object(claude_code_llm, "query", fake):
        with pytest.raises(RuntimeError, match="boom"):
            ClaudeCodeLLM().complete("x")


def test_sync_complete_inside_running_loop():
    fake = _fake_query([_result("ok")])

    async def main():
        return ClaudeCodeLLM().complete("x").text

    with patch.object(claude_code_llm, "query", fake):
        assert asyncio.run(main()) == "ok"


def test_acomplete():
    fake = _fake_query([_result("async-ok")])
    with patch.object(claude_code_llm, "query", fake):
        out = asyncio.run(ClaudeCodeLLM().acomplete("x"))
    assert out.text == "async-ok"


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
def test_warns_when_api_credential_overrides_subscription(monkeypatch, caplog, name):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv(name, "sk-test")
    with caplog.at_level("WARNING"):
        ClaudeCodeLLM()._options()
    assert name in caplog.text


def test_no_warning_without_api_credentials(monkeypatch, caplog):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with caplog.at_level("WARNING"):
        ClaudeCodeLLM()._options()
    assert caplog.text == ""
