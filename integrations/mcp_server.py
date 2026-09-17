"""MCP bridge: exposes the Hive Mind server as tools for Claude.

Claude Code / Claude Desktop speak MCP over stdio. This process is a thin
client of the HTTP API, so the memory server itself stays a plain REST service
that ChatGPT (via a custom GPT Action) can use at the same time.

Install and register:

    pip install -r integrations/requirements.txt
    claude mcp add hive-mind -e HIVE_MIND_URL=https://your-app.onrender.com \\
        -e HIVE_MIND_API_KEY=your-key -- python /abs/path/integrations/mcp_server.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x, where the same class was called FastMCP
    from mcp.server.fastmcp import FastMCP as _Server

BASE_URL = os.environ.get("HIVE_MIND_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("HIVE_MIND_API_KEY", "")
AGENT_NAME = os.environ.get("HIVE_MIND_AGENT", "claude")
TIMEOUT = float(os.environ.get("HIVE_MIND_TIMEOUT", "30"))

mcp = _Server("hive-mind")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


async def _request(method: str, path: str, **kwargs: Any) -> Any:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        response = await client.request(method, path, headers=_headers(), **kwargs)
        if response.status_code >= 400:
            return {"error": response.status_code, "detail": response.text}
        return response.json()


@mcp.tool()
async def remember(
    project: str,
    title: str,
    content: str,
    type: str = "notes",
    tags: list[str] | None = None,
) -> Any:
    """Store knowledge in shared memory so any agent can retrieve it later.

    Use for decisions, plans, architecture notes and summaries that outlive the
    current conversation. `type` is one of plan, summary, notes, architecture.
    """
    return await _request(
        "POST",
        "/memory/write",
        json={
            "project": project,
            "type": type,
            "title": title,
            "content": content,
            "tags": tags or [],
            "source": AGENT_NAME,
        },
    )


@mcp.tool()
async def recall(
    query: str,
    project: str | None = None,
    top_k: int = 5,
    mode: str = "hybrid",
) -> Any:
    """Semantically search shared memory for anything relevant to `query`."""
    payload: dict[str, Any] = {"query": query, "top_k": top_k, "mode": mode}
    if project:
        payload["project"] = project
    return await _request("POST", "/memory/search", json=payload)


@mcp.tool()
async def project_context(project: str, query: str | None = None) -> Any:
    """Load everything needed to resume work on a project, grouped by type."""
    params = {"query": query} if query else None
    return await _request("GET", f"/projects/{project}/context", params=params)


@mcp.tool()
async def list_projects() -> Any:
    """List every project in shared memory with its memory counts."""
    return await _request("GET", "/projects")


if __name__ == "__main__":
    mcp.run()
