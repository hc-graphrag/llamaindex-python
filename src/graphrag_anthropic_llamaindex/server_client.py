"""Client for the separate GraphRAG/LlamaIndex HTTP server."""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.request import Request, urlopen


class GraphRAGClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/healthz")

    def add_document(self, filename: str, content: str) -> dict[str, Any]:
        return self._request(
            "POST", "/v1/documents", {"filename": filename, "content": content}
        )

    def search(self, query: str, mode: str = "global") -> dict[str, Any]:
        return self._request("POST", "/v1/search", {"query": query, "mode": mode})

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"content-type": "application/json"},
        )
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG server client")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health")
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--mode", default="global")
    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("file")
    args = parser.parse_args()

    client = GraphRAGClient(args.url)
    if args.command == "health":
        result = client.health()
    elif args.command == "search":
        result = client.search(args.query, args.mode)
    else:
        with open(args.file, encoding="utf-8") as file:
            result = client.add_document(args.file.rsplit("/", 1)[-1], file.read())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
