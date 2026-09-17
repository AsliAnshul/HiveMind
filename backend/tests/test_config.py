"""Configuration parsing — no database, no model."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/placeholder")

from app.utils.config import Settings  # noqa: E402

BASE = {"database_url": "postgresql://u:p@localhost/db"}


def test_comma_separated_keys_are_parsed() -> None:
    """A plain CSV value must not be treated as JSON — that used to crash boot."""
    settings = Settings(**BASE, api_keys="claude-key, chatgpt-key")
    assert settings.api_key_list == ["claude-key", "chatgpt-key"]
    assert settings.auth_enabled


def test_empty_keys_mean_no_auth() -> None:
    assert Settings(**BASE, api_keys="").auth_enabled is False


def test_database_url_is_normalised() -> None:
    settings = Settings(database_url="postgres://u:p@host/db")
    assert settings.database_url.startswith("postgresql+psycopg2://")


def test_mcp_hosts_follow_the_public_url() -> None:
    """A deployment must not 421 itself: the public hostname is allowed."""
    settings = Settings(**BASE, public_base_url="https://hive-mind.onrender.com")
    hosts = settings.mcp_allowed_host_list
    assert "hive-mind.onrender.com" in hosts
    assert "hive-mind.onrender.com:*" in hosts
    assert "localhost" in hosts


def test_mcp_hosts_empty_without_a_public_url() -> None:
    """Nothing configured means the Host check is off, not that everything 421s."""
    assert Settings(**BASE).mcp_allowed_host_list == []


def test_extra_mcp_hosts_are_honoured() -> None:
    settings = Settings(**BASE, mcp_allowed_hosts="memory.example.com, alias.example.com")
    hosts = settings.mcp_allowed_host_list
    assert "memory.example.com" in hosts
    assert "alias.example.com" in hosts
