"""Application configuration, loaded once from the environment."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Only ``DATABASE_URL`` is mandatory; every other value has a working default.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- database -----------------------------------------------------------
    database_url: str = Field(
        ...,
        description="Postgres connection string (Supabase session or direct connection).",
    )
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_recycle_seconds: int = 1800
    db_connect_timeout_seconds: int = 10
    db_statement_timeout_ms: int = 30_000

    # --- schema management --------------------------------------------------
    auto_migrate: bool = Field(
        True,
        description="Create the extension, table and indexes on startup if missing.",
    )
    # hnsw is the default: an ivfflat index built before the table has data has
    # no clusters to search and silently returns fewer rows than asked for.
    # ivfflat is still fully supported — see README §3.
    vector_index_type: Literal["ivfflat", "hnsw"] = "hnsw"
    ivfflat_lists: int = 100
    ivfflat_probes: int = 10
    hnsw_ef_search: int = 100
    iterative_scan: bool = Field(
        True,
        description=(
            "pgvector >= 0.8: keep scanning the ANN index until top_k rows survive "
            "the WHERE clause. Ignored on older pgvector."
        ),
    )

    # --- embeddings ---------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_backend: Literal["auto", "sentence-transformers", "fastembed"] = "auto"
    embedding_chunk_words: int = 180
    embedding_max_chunks: int = 8
    warm_model_on_startup: bool = True

    # --- api ----------------------------------------------------------------
    # Kept as raw strings: pydantic-settings tries to JSON-decode list-typed
    # fields straight from the environment, which makes a plain
    # "key-a,key-b" value a startup crash. The parsed views are below.
    api_keys: str = Field(
        default="",
        description="Comma-separated keys accepted in the X-API-Key header. Empty = open.",
    )
    cors_origins: str = "*"
    default_top_k: int = 5
    max_top_k: int = 50
    context_limit: int = 10
    context_query: str = "project overview architecture plan tasks"

    public_base_url: str = Field(
        default="",
        description=(
            "Public URL of this deployment, e.g. https://hive-mind.onrender.com. "
            "Emitted as servers[] in openapi.json, which ChatGPT Actions requires."
        ),
    )

    mcp_allowed_hosts: str = Field(
        default="",
        description=(
            "Comma-separated Host header values the /mcp endpoint accepts "
            "(DNS-rebinding protection). The host from PUBLIC_BASE_URL is added "
            "automatically. Empty and no PUBLIC_BASE_URL means the check is off."
        ),
    )

    app_name: str = "Hive Mind Memory Server"
    app_env: str = "development"
    log_level: str = "INFO"

    @field_validator("database_url")
    @classmethod
    def _normalise_database_url(cls, value: str) -> str:
        """Force the psycopg2 driver and require sslmode for hosted Postgres."""
        url = value.strip()
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url[len("postgresql://") :]
        return url

    @property
    def api_key_list(self) -> list[str]:
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def mcp_allowed_host_list(self) -> list[str]:
        """Host values the MCP endpoint will answer to.

        The deployed hostname comes from PUBLIC_BASE_URL, so a correct
        deployment needs no extra configuration. Loopback entries are included
        so local development keeps working on any port.
        """
        hosts = [h.strip() for h in self.mcp_allowed_hosts.split(",") if h.strip()]
        if self.public_base_url:
            netloc = urlsplit(self.public_base_url).netloc
            if netloc:
                # A proxy may or may not include the port in the Host header,
                # so accept the bare name and any port on it.
                bare = netloc.split(":")[0]
                hosts.extend([netloc, bare, f"{bare}:*"])
        if hosts:
            hosts.extend(["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*"])

        seen: list[str] = []
        for host in hosts:
            if host not in seen:
                seen.append(host)
        return seen

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key_list)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()  # type: ignore[call-arg]
