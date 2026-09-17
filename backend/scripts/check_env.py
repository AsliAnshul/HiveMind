"""Validate .env before you try to run anything.

    python scripts/check_env.py

Checks the connection string, reaches the database, confirms pgvector is
available, verifies the schema can be created, and loads the embedding model.
Every failure prints the specific fix.
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS, FAIL, WARN = "  ok  ", " FAIL ", " warn "
failures = 0


def report(status: str, label: str, detail: str = "") -> None:
    global failures
    if status is FAIL:
        failures += 1
    print(f"[{status}] {label}" + (f"\n         {detail}" if detail else ""))


def load_settings():
    """Build Settings, which is what actually reads .env.

    Reading os.environ here instead would miss the .env file entirely — the
    whole point of this script is to check the file the user just edited.
    """
    from pydantic import ValidationError

    from app.utils.config import Settings

    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        missing = {
            str(error["loc"][0]) for error in exc.errors() if error["type"] == "missing"
        }
        if "database_url" in missing:
            report(
                FAIL,
                "DATABASE_URL is set",
                "Not found in backend/.env or the environment. Paste the Supabase "
                "URI (Connect -> Session pooler) into the DATABASE_URL= line.",
            )
        else:
            report(FAIL, "Configuration is valid", str(exc))
        return None


def _check_address_family(hostname: str) -> None:
    """Warn when the host has no IPv4 address.

    Supabase's direct-connection host (``db.<ref>.supabase.co``) publishes only
    an AAAA record. That is fine on an IPv6-capable machine and fatal on most
    free hosting tiers, Render included, which have no IPv6 outbound — so the
    app would work locally and fail the moment it is deployed. The session
    pooler host resolves on IPv4 and is the fix.
    """
    import socket

    try:
        families = {info[0] for info in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        report(FAIL, "Hostname resolves", f"{hostname} did not resolve ({exc}).")
        return

    if socket.AF_INET in families:
        return

    report(
        WARN,
        "Host has no IPv4 address",
        f"{hostname} is IPv6-only. Render and most free tiers cannot reach it, "
        "so this works locally and breaks on deploy. Use the Session pooler "
        "URI instead (Supabase -> Connect -> Session pooler): the host looks "
        "like aws-0-<region>.pooler.supabase.com and the user becomes "
        "postgres.<project-ref>.",
    )


def check_connection_string(url: str) -> bool:
    """Sanity-check the resolved connection string. Returns False if unusable."""
    if "YOUR-PASSWORD" in url:
        report(
            FAIL,
            "DATABASE_URL has a real password",
            "The placeholder is still there. Replace [YOUR-PASSWORD] with the "
            "database password (Project Settings -> Database -> Reset database "
            "password if you do not have it).",
        )
        return False

    parts = urlsplit(url)
    report(PASS, f"DATABASE_URL is set (host {parts.hostname}, port {parts.port or 5432})")

    if not parts.password:
        report(
            WARN,
            "URI carries a password",
            "No password component found. If the database requires one, the "
            "connection below will fail.",
        )
    _check_address_family(parts.hostname or "")

    if parts.port == 6543:
        report(
            WARN,
            "Connection uses the transaction pooler (port 6543)",
            "Prepared statements are disabled there, which SQLAlchemy relies on. "
            "Switch to the session pooler on port 5432.",
        )
    if "sslmode" not in url and parts.hostname not in {"localhost", "127.0.0.1"}:
        report(
            WARN,
            "No sslmode in the URI",
            "Hosted Postgres requires TLS. Append ?sslmode=require to the URI.",
        )
    return True


def main() -> int:
    print("Hive Mind — environment check\n")

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", ".env")):
        report(WARN, "backend/.env exists", "Not found. Copy .env.example to .env.")

    settings = load_settings()
    if settings is None or not check_connection_string(settings.database_url):
        print("\nFix the above, then run this again.")
        return 1

    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    from app.db import ensure_schema, get_engine, pgvector_version

    # --- reachability ------------------------------------------------------
    try:
        with get_engine().connect() as conn:
            version = conn.scalar(text("SHOW server_version"))
            user = conn.scalar(text("SELECT current_user"))
        report(PASS, f"Database reachable (Postgres {version}, as {user})")
    except SQLAlchemyError as exc:
        report(
            FAIL,
            "Database reachable",
            f"{type(exc).__name__}: {getattr(exc, 'orig', exc)}\n"
            "         Wrong password, wrong host, or the Supabase project is "
            "paused (open the dashboard and restore it).",
        )
        return 1

    # --- schema and pgvector ----------------------------------------------
    # ensure_schema() creates the extension, so it has to run before the
    # extension is reported on — otherwise a fresh database looks broken.
    try:
        ensure_schema()
        with get_engine().connect() as conn:
            rows = conn.scalar(text("SELECT count(*) FROM memory"))
        schema_error = None
    except SQLAlchemyError as exc:
        rows, schema_error = None, exc

    installed = pgvector_version()
    if installed:
        report(PASS, f"pgvector installed ({'.'.join(map(str, installed))})")
        if installed < (0, 8, 0):
            report(
                WARN,
                "pgvector is older than 0.8",
                "Iterative index scans are unavailable, so heavily filtered "
                "searches may return fewer rows. Keep VECTOR_INDEX_TYPE=hnsw.",
            )
    else:
        try:
            with get_engine().connect() as conn:
                available = conn.scalar(
                    text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
                )
                conn.rollback()
        except SQLAlchemyError:
            available = None
        report(
            FAIL,
            "pgvector installed",
            "It is available but could not be created — the role lacks "
            "permission. Run this once in the Supabase SQL editor: "
            "CREATE EXTENSION IF NOT EXISTS vector;"
            if available
            else "This database does not offer the 'vector' extension. On "
            "Supabase: Database -> Extensions -> enable 'vector'.",
        )
        return 1

    if schema_error is not None:
        report(
            FAIL,
            "Schema ready",
            f"{type(schema_error).__name__}: {getattr(schema_error, 'orig', schema_error)}\n"
            "         Run backend/migrations/001_init.sql in the Supabase SQL "
            "editor, then set AUTO_MIGRATE=false.",
        )
        return 1
    report(PASS, f"Schema ready (memory table holds {rows} rows)")

    # --- auth --------------------------------------------------------------
    if settings.auth_enabled:
        report(PASS, f"API key auth enabled ({len(settings.api_key_list)} key(s))")
    else:
        report(
            WARN,
            "API key auth disabled",
            "API_KEYS is empty, so every endpoint is public. Fine locally, not "
            "once this is deployed. Generate one: openssl rand -hex 32",
        )

    # --- model -------------------------------------------------------------
    try:
        from app.services.embedding_service import get_embedder

        embedder = get_embedder()
        vector = embedder.embed("environment check")
        report(
            PASS,
            f"Embedding model loaded via {embedder.backend_name} ({len(vector)} dims)",
        )
    except Exception as exc:  # noqa: BLE001 - report any loading failure
        report(
            FAIL,
            "Embedding model loaded",
            f"{type(exc).__name__}: {exc}\n"
            "         pip install -r requirements-lite.txt (ONNX) or "
            "requirements.txt (PyTorch).",
        )
        return 1

    if settings.public_base_url:
        report(PASS, f"PUBLIC_BASE_URL set ({settings.public_base_url})")
    else:
        report(
            WARN,
            "PUBLIC_BASE_URL not set",
            "Only ChatGPT Actions needs it. Set it to the deployed URL later.",
        )

    print()
    if failures:
        print(f"{failures} check(s) failed. Fix them and run this again.")
        return 1
    print("All good. Start the server:  uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
