"""
Claude Code (サブスクリプション認証) 経由の LlamaIndex LLM。

公式 Claude Agent SDK は `claude` CLI をサブプロセスとして起動するため、
`claude login` 済みの Pro/Max サブスクリプション認証をそのまま利用でき、
ANTHROPIC_API_KEY は不要。ツールは全て無効化し、純粋なテキスト生成にのみ使う。

注意: Anthropic の規約上、サブスクリプション枠を第三者向けサービスで使うことは
認められていない。本クラスは本人のローカル環境での個人利用を前提とする。
"""

import asyncio
import concurrent.futures
import logging
import os
from typing import Any, AsyncGenerator, Generator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

logger = logging.getLogger(__name__)

# claude CLI の認証では、これらが設定されているとサブスクリプションより優先される
_OVERRIDING_CREDENTIALS = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")

# Claude Code が標準で持つツール。生成専用にするため全て拒否する。
_BUILTIN_TOOLS = [
    "Bash", "BashOutput", "KillShell", "Read", "Write", "Edit", "NotebookEdit",
    "Glob", "Grep", "WebFetch", "WebSearch", "Task", "TodoWrite", "SlashCommand",
]


class ClaudeCodeLLM(CustomLLM):
    """`claude` CLI のログイン済みサブスクリプション認証で動く LLM。"""

    model: str = Field(default="sonnet", description="モデルのエイリアスまたはフルID")
    context_window: int = Field(default=200000, description="コンテキストウィンドウ")
    num_output: int = Field(default=8192, description="最大出力トークン数 (メタデータ用)")
    system_prompt: str = Field(default="", description="システムプロンプト")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=False,
            model_name=self.model,
        )

    def _options(self) -> ClaudeAgentOptions:
        overriding = [name for name in _OVERRIDING_CREDENTIALS if os.environ.get(name)]
        if overriding:
            logger.warning(
                "%s が設定されているため、サブスクリプションではなく API キー課金で実行されます。"
                "サブスクリプションを使うなら環境から外してください。",
                ", ".join(overriding),
            )
        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=self.system_prompt or None,
            max_turns=1,
            disallowed_tools=_BUILTIN_TOOLS,
            # 既存の settings/CLAUDE.md/hooks を読み込まず、再現性を保つ
            setting_sources=[],
        )

    async def _generate(self, prompt: str) -> str:
        text_parts: list[str] = []
        result_text = None
        async for message in query(prompt=prompt, options=self._options()):
            if isinstance(message, AssistantMessage):
                text_parts.extend(
                    block.text for block in message.content if isinstance(block, TextBlock)
                )
            elif isinstance(message, ResultMessage):
                if message.is_error:
                    raise RuntimeError(f"Claude Code がエラーを返しました: {message.result}")
                result_text = message.result
        return result_text if result_text is not None else "".join(text_parts)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text=await self._generate(prompt))

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return CompletionResponse(text=asyncio.run(self._generate(prompt)))
        # 実行中のイベントループ内からの同期呼び出しは別スレッドの専用ループで実行する
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            text = pool.submit(asyncio.run, self._generate(prompt)).result()
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        response = self.complete(prompt, formatted=formatted, **kwargs)

        def gen() -> Generator[CompletionResponse, None, None]:
            yield CompletionResponse(text=response.text, delta=response.text)

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        text = await self._generate(prompt)

        async def gen() -> AsyncGenerator[CompletionResponse, None]:
            yield CompletionResponse(text=text, delta=text)

        return gen()
