"""HTTP API server for GraphRAG/LlamaIndex backed by AlopexDB server."""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.bedrock_converse import BedrockConverse
from pydantic import BaseModel, Field

from .alopex_db import get_database
from .claude_code_llm import ClaudeCodeLLM
from .config_manager import load_config
from .document_processor import add_documents
from .file_filter import FileFilter
from .global_search import SearchModeRouter
from .vector_store_manager import get_vector_store


class DocumentRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["local", "global", "drift", "auto"] = "global"
    response_type: str = "multiple paragraphs"
    output_format: Literal["markdown", "json"] = "markdown"
    min_community_rank: int = 0


class GraphRAGService:
    """Own the server-side LlamaIndex runtime and AlopexDB connection."""

    def __init__(self) -> None:
        self.config_path = os.environ.get(
            "GRAPHRAG_CONFIG", "/app/config/config.yaml"
        )
        self.storage_target = os.environ.get(
            "ALOPEX_SERVER_URL", "http://alopexdb:8080"
        )
        self._lock = RLock()
        self._initialized = False
        self.config: dict[str, Any] = {}
        self.vector_stores: dict[str, Any] = {}

    def health(self) -> dict[str, str]:
        rows = get_database(self.storage_target).execute_sql("SELECT 1 AS ok")
        if rows != [{"ok": 1}]:
            raise RuntimeError(f"Unexpected AlopexDB health response: {rows!r}")
        return {"status": "ok", "alopexdb": "ok"}

    def add_document(self, request: DocumentRequest) -> dict[str, str]:
        with self._lock:
            self._initialize()
            filename = Path(request.filename)
            if filename.name != request.filename or filename.name in {".", ".."}:
                raise ValueError("filename must not contain a directory path")
            input_dir = Path(self.config.get("input_dir", "/app/data"))
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / filename.name).write_text(request.content, encoding="utf-8")

            add_documents(
                str(input_dir),
                self.storage_target,
                self.vector_stores["main"],
                self.vector_stores["entity"],
                self.vector_stores["community"],
                self.config.get("community_detection", {}),
                use_archive_reader=True,
                file_filter=FileFilter(self.config.get("ignore_patterns", [])),
            )
        return {"status": "indexed", "filename": filename.name}

    def search(self, request: SearchRequest) -> dict[str, Any]:
        with self._lock:
            self._initialize()
            router = SearchModeRouter(
                config=self.config,
                mode=request.mode,
                vector_store_main=self.vector_stores["main"],
                vector_store_entity=self.vector_stores["entity"],
                vector_store_community=self.vector_stores["community"],
                response_type=request.response_type,
                min_community_rank=request.min_community_rank,
                output_format=request.output_format,
            )
            results = router._retrieve(QueryBundle(query_str=request.query))
        return {
            "results": [
                {
                    "score": result.score,
                    "text": result.node.text,
                    "metadata": result.node.metadata,
                }
                for result in results
            ]
        }

    def _initialize(self) -> None:
        if self._initialized:
            return
        config = load_config(self.config_path)
        if not config:
            raise RuntimeError(f"Could not load config: {self.config_path}")
        self.config = {**config, "output_dir": self.storage_target}
        self.vector_stores = {
            kind: get_vector_store(self.config, kind)
            for kind in ("main", "entity", "community")
        }
        self._configure_settings(config)
        self._initialized = True

    @staticmethod
    def _configure_settings(config: dict[str, Any]) -> None:
        provider = config.get("llm_provider", "anthropic")
        if provider == "claude_code":
            model = config.get("claude_code", {}).get("model", "sonnet")
            Settings.llm = ClaudeCodeLLM(model=model)
        elif provider == "bedrock":
            bedrock = config.get("bedrock", {})
            params: dict[str, Any] = {
                "model": bedrock.get(
                    "model", "anthropic.claude-3-sonnet-20240229-v1:0"
                ),
                "region_name": bedrock.get("region", "us-east-1"),
            }
            if os.environ.get("AWS_PROFILE_NAME"):
                params["profile_name"] = os.environ["AWS_PROFILE_NAME"]
            Settings.llm = BedrockConverse(**params)
        else:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is required")
            anthropic = config.get("anthropic", {})
            params = {"model": anthropic.get("model", "claude-sonnet-5-5")}
            if anthropic.get("api_base_url"):
                params["api_base_url"] = anthropic["api_base_url"]
            Settings.llm = Anthropic(**params)

        embedding = config.get("embedding_model", {})
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding.get("name", "intfloat/multilingual-e5-small")
        )
        chunking = config.get("chunking", {})
        Settings.node_parser = SentenceSplitter(
            chunk_size=chunking.get("chunk_size", 1024),
            chunk_overlap=chunking.get("chunk_overlap", 20),
        )


service = GraphRAGService()
app = FastAPI(title="GraphRAG LlamaIndex Server", version="1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    try:
        return service.health()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/v1/documents")
def add_document(request: DocumentRequest) -> dict[str, str]:
    try:
        return service.add_document(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/v1/search")
def search(request: SearchRequest) -> dict[str, Any]:
    try:
        return service.search(request)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
