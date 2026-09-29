"""The service must survive an unreachable database.

A paused Supabase project used to crash the process during startup: Render
reported a failed deploy, /health and /mcp went down with it, and nothing
recovered until a human redeployed.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from app import db as database


@pytest.fixture
def unreachable_database(monkeypatch):
    """Make schema verification fail the way a paused project does."""
    monkeypatch.setattr(database, "_schema_ready", False)

    def boom() -> None:
        raise OperationalError(
            "SELECT 1", {}, Exception("FATAL: (ENOTFOUND) tenant/user not found")
        )

    monkeypatch.setattr(database, "ensure_schema", boom)


def test_ensure_schema_once_reports_failure_instead_of_raising(unreachable_database) -> None:
    assert database.ensure_schema_once() is False
    assert database.schema_is_ready() is False


def test_the_app_still_starts(unreachable_database) -> None:
    """The exact regression: uvicorn exited with status 3 on this path.

    The app is rebuilt from scratch rather than reusing the module-level one,
    because MCP's session manager refuses to start twice in a process and the
    rest of the suite has already started it once.
    """
    import importlib

    from fastapi.testclient import TestClient

    import app.main

    fresh = importlib.reload(app.main)
    try:
        with TestClient(fresh.app) as client:
            body = client.get("/health").json()
            # The database itself is reachable here; only schema verification
            # was made to fail, so "ok" with schema_ready False is correct.
            assert body["schema_ready"] is False
            # The banner needs no database at all, so it must still answer.
            assert client.get("/").status_code == 200
    finally:
        # Leave the module as the rest of the session found it.
        importlib.reload(app.main)


def test_schema_is_retried_once_the_database_returns(monkeypatch) -> None:
    """Recovery must not need a redeploy."""
    monkeypatch.setattr(database, "_schema_ready", False)
    attempts = []

    def flaky() -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise OperationalError("SELECT 1", {}, Exception("down"))

    monkeypatch.setattr(database, "ensure_schema", flaky)

    assert database.ensure_schema_once() is False
    assert database.ensure_schema_once() is True
    assert database.schema_is_ready() is True

    # Verified once, never re-run.
    assert database.ensure_schema_once() is True
    assert len(attempts) == 2


def test_verified_schema_is_not_rechecked_per_request(monkeypatch) -> None:
    monkeypatch.setattr(database, "_schema_ready", True)
    monkeypatch.setattr(
        database, "ensure_schema", lambda: pytest.fail("should not re-run")
    )
    assert database.ensure_schema_once() is True
