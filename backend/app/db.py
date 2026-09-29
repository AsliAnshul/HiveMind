"""Database engine, session handling and schema bootstrap."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.models import EMBEDDING_DIM
from app.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _session_settings(settings: Settings) -> list[str]:
    """ANN search knobs, applied once per connection rather than per query.

    These used to be ``SET LOCAL`` inside each search transaction, which cost
    two extra round trips on every request — noticeable when the database is a
    continent away. Applied at connection setup they are effectively free,
    because the session pooler gives each connection its own backend.
    """
    if settings.vector_index_type == "hnsw":
        statements = [f"SET hnsw.ef_search = {int(settings.hnsw_ef_search)}"]
    else:
        statements = [f"SET ivfflat.probes = {int(settings.ivfflat_probes)}"]
    if settings.iterative_scan:
        statements.append(
            f"SET {settings.vector_index_type}.iterative_scan = relaxed_order"
        )
    return statements


def _configure_connection(dbapi_connection, _record) -> None:
    """Apply the session settings to a freshly opened connection.

    Autocommit is forced for the duration: a plain ``SET`` inside a transaction
    is undone if that transaction rolls back, which would silently give the
    connection different search behaviour from every other one.
    """
    settings = get_settings()
    statements = _session_settings(settings)
    previous = dbapi_connection.autocommit
    dbapi_connection.autocommit = True
    try:
        cursor = dbapi_connection.cursor()
        try:
            # psycopg2 sends semicolon-separated statements in one round trip.
            cursor.execute("; ".join(statements))
        except Exception as exc:  # noqa: BLE001 - older pgvector lacks these GUCs
            logger.warning(
                "Could not apply ANN search settings (%s); falling back to "
                "server defaults. Filtered searches may return fewer rows.",
                exc,
            )
            try:
                cursor.execute(statements[0])
            except Exception:  # noqa: BLE001 - nothing more to try
                logger.warning("ANN search settings unavailable on this server.")
        finally:
            cursor.close()
    finally:
        dbapi_connection.autocommit = previous


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create the process-wide SQLAlchemy engine."""
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle_seconds,
        future=True,
        connect_args={
            "connect_timeout": settings.db_connect_timeout_seconds,
            "options": f"-c statement_timeout={settings.db_statement_timeout_ms}",
            "application_name": "hive-mind",
        },
    )
    event.listen(engine, "connect", _configure_connection)
    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a transactional session."""
    ensure_schema_once()
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for use outside of request handlers (scripts, startup)."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def vector_index_name(settings: Settings | None = None) -> str:
    """Name of the ANN index for the configured index type."""
    settings = settings or get_settings()
    return f"memory_embedding_{settings.vector_index_type}_idx"


def schema_statements(settings: Settings | None = None) -> list[str]:
    """Return the idempotent DDL that defines the whole store.

    This is the single source of truth for the schema; ``migrations/001_init.sql``
    is generated from it by ``scripts/print_schema.py``.
    """
    settings = settings or get_settings()

    # The index name carries its type, so switching VECTOR_INDEX_TYPE actually
    # builds the new index instead of silently keeping the old one behind an
    # IF NOT EXISTS.
    if settings.vector_index_type == "hnsw":
        vector_index = (
            f"CREATE INDEX IF NOT EXISTS {vector_index_name(settings)} ON memory "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64);"
        )
    else:
        vector_index = (
            f"CREATE INDEX IF NOT EXISTS {vector_index_name(settings)} ON memory "
            "USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = {settings.ivfflat_lists});"
        )

    return [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        f"""CREATE TABLE IF NOT EXISTS memory (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project      TEXT NOT NULL,
    type         TEXT NOT NULL,
    title        TEXT NOT NULL,
    content      TEXT NOT NULL,
    tags         TEXT[] NOT NULL DEFAULT '{{}}',
    source       TEXT,
    metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    content_hash VARCHAR(64) NOT NULL,
    embedding    VECTOR({EMBEDDING_DIM}) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);""",
        "CREATE INDEX IF NOT EXISTS memory_project_idx ON memory (project);",
        "CREATE INDEX IF NOT EXISTS memory_type_idx ON memory (type);",
        "CREATE INDEX IF NOT EXISTS memory_created_at_idx ON memory (created_at DESC);",
        "CREATE UNIQUE INDEX IF NOT EXISTS memory_project_hash_key "
        "ON memory (project, content_hash);",
        "CREATE INDEX IF NOT EXISTS memory_fts_idx ON memory "
        "USING gin (to_tsvector('english', title || ' ' || content));",
        vector_index,
    ]


def ensure_schema() -> None:
    """Apply the schema if it is missing. Safe to run on every boot.

    Each statement runs in its own transaction so that a permission error on
    ``CREATE EXTENSION`` (common on managed Postgres) cannot poison the rest.
    """
    settings = get_settings()
    engine = get_engine()
    with engine.connect() as conn:
        for statement in schema_statements(settings):
            try:
                conn.execute(text(statement))
                conn.commit()
            except ProgrammingError as exc:
                conn.rollback()
                if "CREATE EXTENSION" in statement:
                    # Managed Postgres (e.g. Supabase) may forbid CREATE EXTENSION
                    # for the app role. The extension is usually pre-installed.
                    logger.warning(
                        "Could not create the 'vector' extension automatically (%s). "
                        "Enable it once via the Supabase SQL editor: "
                        "CREATE EXTENSION IF NOT EXISTS vector;",
                        exc.orig,
                    )
                    continue
                raise
    pgvector_version.cache_clear()
    _warn_about_stale_vector_index(settings)
    logger.info(
        "Schema verified (vector index: %s).", vector_index_name(settings)
    )


def _warn_about_stale_vector_index(settings: Settings) -> None:
    """Point out an ANN index left behind by a previous VECTOR_INDEX_TYPE."""
    keep = vector_index_name(settings)
    with get_engine().connect() as conn:
        stale = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'memory' AND indexname LIKE 'memory_embedding_%%' "
                "AND indexname <> :keep"
            ),
            {"keep": keep},
        ).scalars().all()
    for name in stale:
        logger.warning(
            "Index %s is left over from a different VECTOR_INDEX_TYPE. It still "
            "works but costs write throughput and disk; drop it with: DROP INDEX %s;",
            name,
            name,
        )


@lru_cache(maxsize=1)
def pgvector_version() -> tuple[int, ...] | None:
    """The installed pgvector version, e.g. ``(0, 8, 1)``, or None if absent."""
    try:
        with get_engine().connect() as conn:
            raw = conn.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
    except SQLAlchemyError as exc:
        logger.error("Could not read the pgvector version: %s", exc)
        return None
    if not raw:
        return None
    try:
        return tuple(int(part) for part in str(raw).split("."))
    except ValueError:
        logger.warning("Unrecognised pgvector version %r.", raw)
        return None


_schema_ready = False
_schema_lock = threading.Lock()


def schema_is_ready() -> bool:
    """Whether the schema has been verified at least once this process."""
    return _schema_ready


def ensure_schema_once() -> bool:
    """Verify the schema if it has not been verified yet. Never raises.

    Called at startup and again before serving each request, so a database that
    was unreachable at boot — a paused free-tier project, say — is picked up as
    soon as it comes back, with no redeploy. After the first success this is a
    boolean check.
    """
    global _schema_ready
    if _schema_ready:
        return True
    with _schema_lock:
        if _schema_ready:
            return True
        try:
            ensure_schema()
        except SQLAlchemyError as exc:
            logger.warning(
                "Schema not verified — the database is unreachable (%s). The "
                "service will keep running and retry on the next request.",
                getattr(exc, "orig", exc),
            )
            return False
        _schema_ready = True
        return True


def ping() -> bool:
    """Cheap liveness probe for the database."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as exc:
        logger.error("Database ping failed: %s", exc)
        return False
