"""Hive Mind Memory Server — a shared long-term memory for AI agents."""

from __future__ import annotations

import hmac
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from starlette.types import ASGIApp, Receive, Scope, Send

from app import db as database
from app.mcp_server import build_mcp_server
from app.routes import memory as memory_routes
from app.routes import projects as project_routes
from app.schemas.memory_schema import HealthResponse
from app.services import memory_service
from app.services.embedding_service import get_embedder
from app.utils.config import Settings, get_settings
from app.utils.security import require_api_key

VERSION = "1.0.0"

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("hive_mind")


def _transport_security(settings: Settings) -> TransportSecuritySettings:
    """DNS-rebinding protection for the MCP endpoint.

    The SDK rejects unknown Host headers with 421, which silently breaks a
    deployment whose hostname it has not been told about. Rather than leave that
    to chance, the allowed hosts are derived from PUBLIC_BASE_URL. With nothing
    configured the check is disabled and startup says so — the API key is still
    enforced either way.
    """
    allowed = settings.mcp_allowed_host_list
    if not allowed:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed,
        allowed_origins=[o for o in settings.cors_origin_list if o != "*"],
    )


#: The MCP transport is a Starlette sub-app. It is mounted rather than
#: reimplemented, and its own lifespan is entered from ours below.
mcp_server = build_mcp_server()
mcp_app = mcp_server.streamable_http_app(
    streamable_http_path="/",
    stateless_http=True,
    transport_security=_transport_security(settings),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Verify the schema and warm the embedding model before serving traffic."""
    if settings.auto_migrate:
        database.ensure_schema()
    else:
        logger.info("AUTO_MIGRATE disabled; assuming the schema already exists.")

    if settings.warm_model_on_startup:
        started = time.perf_counter()
        get_embedder().embed("warm up")
        logger.info("Model warm-up finished in %.1fs.", time.perf_counter() - started)

    if not settings.auth_enabled:
        logger.warning(
            "API_KEYS is empty — every endpoint is public. Set API_KEYS before deploying."
        )

    if not settings.mcp_allowed_host_list:
        logger.warning(
            "PUBLIC_BASE_URL is unset, so the MCP endpoint accepts any Host "
            "header. Set it to this deployment's URL to enable DNS-rebinding "
            "protection (and to make openapi.json importable by ChatGPT)."
        )

    # Mounted Starlette apps do not get their lifespan run for free.
    async with mcp_app.router.lifespan_context(mcp_app):
        logger.info(
            "MCP endpoint live at /mcp (allowed hosts: %s).",
            ", ".join(settings.mcp_allowed_host_list) or "any",
        )
        yield

    database.get_engine().dispose()
    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.app_name,
    version=VERSION,
    summary="Structured, semantically searchable memory shared between AI agents.",
    description=(
        "Write structured knowledge with `POST /memory/write`, retrieve it with "
        "`POST /memory/search`, and rehydrate a whole project with "
        "`GET /projects/{project}/context`."
    ),
    lifespan=lifespan,
    servers=(
        [{"url": settings.public_base_url.rstrip("/")}]
        if settings.public_base_url
        else None
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(memory_routes.router, dependencies=[Depends(require_api_key)])
app.include_router(project_routes.router, dependencies=[Depends(require_api_key)])


class MCPAuthMiddleware:
    """Apply the same API key rule to the mounted MCP app.

    FastAPI route dependencies do not reach a mounted ASGI app, so the check
    lives here. Both ``X-API-Key`` and ``Authorization: Bearer`` are accepted,
    because MCP clients differ in which one they can send.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/mcp"):
            await self.app(scope, receive, send)
            return

        # Starlette's router answers a bare "/mcp" with a 307 to "/mcp/", and
        # not every MCP client follows a redirect on POST. Rewriting the path
        # here — before routing — makes both spellings work directly.
        if scope["path"] == "/mcp":
            scope = {**scope, "path": "/mcp/", "raw_path": b"/mcp/"}

        settings = get_settings()
        if not settings.auth_enabled:
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        presented = headers.get("x-api-key", "")
        if not presented:
            authorization = headers.get("authorization", "")
            if authorization.lower().startswith("bearer "):
                presented = authorization[7:].strip()

        if presented and any(
            hmac.compare_digest(presented, key) for key in settings.api_key_list
        ):
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            status_code=401 if not presented else 403,
            content={
                "detail": (
                    "Missing API key. Send it as the X-API-Key header or as "
                    "Authorization: Bearer <key>."
                    if not presented
                    else "Invalid API key."
                )
            },
        )
        await response(scope, receive, send)


app.add_middleware(MCPAuthMiddleware)
app.mount("/mcp", mcp_app)


@app.exception_handler(OperationalError)
async def _operational_error(request: Request, exc: OperationalError) -> JSONResponse:
    logger.error("Database unavailable on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503, content={"detail": "Database unavailable. Please retry."}
    )


@app.exception_handler(SQLAlchemyError)
async def _database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("Database error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Database error."})


@app.get("/", tags=["meta"], summary="Service banner", operation_id="service_banner")
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": VERSION,
        "docs": "/docs",
        "endpoints": [
            "POST /memory/write",
            "POST /memory/search",
            "GET /memory",
            "GET /memory/{id}",
            "DELETE /memory/{id}",
            "GET /projects",
            "GET /projects/{project}/context",
        ],
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Health check",
    operation_id="health",
)
def health() -> HealthResponse:
    """Liveness + readiness in one call: database reachable, model loaded."""
    database_ok = database.ping()
    total: int | None = None
    if database_ok:
        try:
            with database.session_scope() as session:
                total = memory_service.count_memories(session)
        except SQLAlchemyError:
            database_ok = False

    backend: str | None = None
    try:
        backend = get_embedder().backend_name
    except Exception:  # noqa: BLE001 - health must never raise
        logger.exception("Embedding model failed to load.")

    return HealthResponse(
        status="ok" if database_ok and backend else "degraded",
        database=database_ok,
        embedding_model=settings.embedding_model,
        embedding_backend=backend,
        embedding_dim=settings.embedding_dim,
        memories=total,
        version=VERSION,
    )
