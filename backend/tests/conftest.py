"""Test fixtures.

The API tests need a real Postgres with pgvector — the whole point of the
system is the vector index, so there is nothing meaningful to test against a
stub. They skip themselves when ``DATABASE_URL`` is not set.
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_PLACEHOLDER = "placeholder"

#: The suite pins its own key rather than inheriting whatever is in .env, and
#: sends it on every request — so auth is exercised by every test, not bypassed.
TEST_API_KEY = "pytest-key"
os.environ["API_KEYS"] = TEST_API_KEY


def _resolve_database_url() -> str:
    """The URL the application would actually use.

    Reading os.environ alone would miss backend/.env, which is where the URL
    normally lives — and the tests would then skip themselves silently on a
    perfectly well configured machine.
    """
    from pydantic import ValidationError

    from app.utils.config import Settings

    try:
        return Settings().database_url  # type: ignore[call-arg]
    except ValidationError:
        return ""


#: Exported so every test module (and SQLAlchemy) sees the same database.
DATABASE_URL = _resolve_database_url()
if DATABASE_URL and _PLACEHOLDER not in DATABASE_URL:
    os.environ["DATABASE_URL"] = DATABASE_URL


def _database_configured() -> bool:
    return bool(DATABASE_URL) and _PLACEHOLDER not in DATABASE_URL


@pytest.fixture(scope="session")
def project() -> str:
    """A throwaway project name so tests never touch real memories."""
    return f"test-{uuid.uuid4().hex[:8]}"


def _database_reachable() -> str:
    """Empty string when the database answers, else why it did not.

    Distinguishing "not configured" from "configured but unreachable" turns a
    wall of errors into one clear skip message — which is what a paused
    Supabase project deserves.
    """
    from sqlalchemy.exc import SQLAlchemyError

    from app.db import ping

    try:
        return "" if ping() else "the database did not answer"
    except SQLAlchemyError as exc:
        return str(getattr(exc, "orig", exc)).splitlines()[0]


@pytest.fixture(scope="session")
def client(project: str) -> Iterator:
    if not _database_configured():
        pytest.skip("DATABASE_URL is not set; skipping database-backed tests.")

    problem = _database_reachable()
    if problem:
        pytest.skip(
            f"DATABASE_URL is set but the database is unreachable ({problem}). "
            "A paused Supabase project looks exactly like this — restore it in "
            "the dashboard, then re-run."
        )

    from fastapi.testclient import TestClient
    from sqlalchemy import text

    from app.db import get_engine
    from app.main import app

    with TestClient(app, headers={"X-API-Key": TEST_API_KEY}) as test_client:
        yield test_client

    try:
        with get_engine().begin() as conn:
            conn.execute(text("DELETE FROM memory WHERE project = :p"), {"p": project})
    except Exception as exc:  # noqa: BLE001 - teardown must not mask results
        print(f"\nCould not clean up test project {project!r}: {exc}")
