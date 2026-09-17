"""MCP server, mounted on the API at ``/mcp``.

This is what makes the deployment directly connectable from Claude: a remote
MCP endpoint over streamable HTTP, with no local bridge process to install.

The tools call the service layer in-process — no HTTP round trip back to
ourselves — and run in a worker thread, because both the database driver and
the embedding model are blocking.

Stateless mode is deliberate: free-tier hosts restart containers whenever they
feel like it, and a stateless server has no session to lose.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import anyio
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from app.db import session_scope
from app.schemas.memory_schema import (
    CANONICAL_TYPES,
    MemoryRecord,
    MemoryWriteRequest,
    SearchRequest,
)
from app.services import memory_service
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

INSTRUCTIONS = """Shared long-term memory, readable and writable by more than one
AI assistant. What you store here another assistant can retrieve later, and what
you retrieve may well have been written by one.

Call `project_context` before working on a project you have notes about, and
`recall` whenever the answer might already be known. Call `remember` after any
decision, plan or conclusion that should outlive this conversation — write the
reasoning, not just the outcome, because the reader will lack your context."""


def _hit_payload(hit: memory_service.ScoredMemory) -> dict[str, Any]:
    record = MemoryRecord.model_validate(hit.memory).model_dump(mode="json")
    record["similarity"] = round(hit.similarity, 6)
    record["rank"] = hit.rank
    return record


def build_mcp_server() -> MCPServer:
    settings = get_settings()
    mcp = MCPServer(
        name="hive-mind",
        title="Hive Mind Memory",
        instructions=INSTRUCTIONS,
        version="1.0.0",
    )

    @mcp.tool()
    async def remember(
        project: Annotated[str, Field(description="Project this belongs to, e.g. 'hive-mind'.")],
        title: Annotated[str, Field(description="A specific one-line title.")],
        content: Annotated[str, Field(description="The knowledge itself. Include the why.")],
        type: Annotated[
            str, Field(description=f"One of: {', '.join(CANONICAL_TYPES)}.")
        ] = "notes",
        tags: Annotated[list[str] | None, Field(description="Optional labels.")] = None,
        source: Annotated[str, Field(description="Which assistant is writing.")] = "claude",
    ) -> dict[str, Any]:
        """Store knowledge in shared memory, where other assistants can find it.

        Writing the same content twice is safe: it refreshes the existing entry
        instead of duplicating it.
        """

        def _write() -> dict[str, Any]:
            payload = MemoryWriteRequest(
                project=project,
                type=type,
                title=title,
                content=content,
                tags=tags or [],
                source=source,
            )
            with session_scope() as session:
                memory, created = memory_service.create_memory(session, payload)
                return {
                    "created": created,
                    "memory": MemoryRecord.model_validate(memory).model_dump(mode="json"),
                }

        return await anyio.to_thread.run_sync(_write)

    @mcp.tool()
    async def recall(
        query: Annotated[str, Field(description="What you want to know, in plain language.")],
        project: Annotated[str | None, Field(description="Restrict to one project.")] = None,
        top_k: Annotated[int, Field(description="How many results.", ge=1, le=50)] = 5,
        mode: Annotated[
            str, Field(description="'hybrid' also matches exact identifiers.")
        ] = "hybrid",
    ) -> dict[str, Any]:
        """Search shared memory by meaning, not keywords.

        Anything another assistant stored is visible here.
        """

        def _search() -> dict[str, Any]:
            request = SearchRequest(
                query=query, project=project, top_k=top_k, mode=mode  # type: ignore[arg-type]
            )
            with session_scope() as session:
                hits = memory_service.search_memory(session, request)
                return {
                    "query": query,
                    "count": len(hits),
                    "results": [_hit_payload(hit) for hit in hits],
                }

        return await anyio.to_thread.run_sync(_search)

    @mcp.tool()
    async def project_context(
        project: Annotated[str, Field(description="The project to load.")],
        query: Annotated[
            str | None, Field(description="Optional focus for the ranking.")
        ] = None,
    ) -> dict[str, Any]:
        """Load everything needed to resume work on a project, grouped by type."""

        def _context() -> dict[str, Any]:
            with session_scope() as session:
                context = memory_service.get_project_context(
                    session, project.strip().lower(), query=query
                )
                return {
                    **context,
                    "groups": {
                        name: [_hit_payload(hit) for hit in hits]
                        for name, hits in context["groups"].items()
                    },
                }

        return await anyio.to_thread.run_sync(_context)

    @mcp.tool()
    async def list_projects() -> dict[str, Any]:
        """List every project in shared memory, with counts and last-updated times."""

        def _list() -> dict[str, Any]:
            with session_scope() as session:
                projects = memory_service.list_projects(session)
            return {
                "count": len(projects),
                "projects": [
                    {**p, "last_updated": p["last_updated"].isoformat()} for p in projects
                ],
            }

        return await anyio.to_thread.run_sync(_list)

    @mcp.tool()
    async def forget(
        memory_id: Annotated[str, Field(description="The UUID of the memory to delete.")],
    ) -> dict[str, Any]:
        """Delete one memory. Irreversible, so confirm with the user first."""
        import uuid as _uuid

        def _delete() -> dict[str, Any]:
            try:
                parsed = _uuid.UUID(memory_id)
            except ValueError:
                return {"deleted": 0, "error": f"{memory_id!r} is not a valid UUID."}
            with session_scope() as session:
                return {"deleted": memory_service.delete_memory(session, parsed)}

        return await anyio.to_thread.run_sync(_delete)

    logger.info("MCP server ready at /mcp (auth: %s).", "on" if settings.auth_enabled else "off")
    return mcp
