"""Pydantic request/response models."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from app.utils.config import get_settings

#: Types that get their own bucket in the project-context response. Any other
#: (lower-cased, slugified) type is accepted and lands in ``other``.
CANONICAL_TYPES: tuple[str, ...] = ("plan", "summary", "notes", "architecture")

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9 _./-]{0,62}[a-z0-9]$|^[a-z0-9]$")

NonEmptyStr = Annotated[str, Field(min_length=1, max_length=200_000)]


def _normalise_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


class MemoryWriteRequest(BaseModel):
    """Payload for ``POST /memory/write``."""

    model_config = ConfigDict(extra="forbid")

    project: Annotated[str, Field(min_length=1, max_length=200)]
    type: Annotated[str, Field(min_length=1, max_length=64)] = "notes"
    title: Annotated[str, Field(min_length=1, max_length=500)]
    content: NonEmptyStr
    tags: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=list, max_length=32
    )
    source: Annotated[str, Field(max_length=64)] | None = Field(
        default=None,
        description="Which agent wrote this, e.g. 'claude' or 'chatgpt'.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("project", "type")
    @classmethod
    def _slug(cls, value: str) -> str:
        normalised = _normalise_key(value)
        if not _SLUG_RE.match(normalised):
            raise ValueError(
                "must be lower-case letters, digits, spaces, '-', '_', '.' or '/'"
            )
        return normalised

    @field_validator("tags")
    @classmethod
    def _tags(cls, values: list[str]) -> list[str]:
        seen: list[str] = []
        for tag in values:
            normalised = _normalise_key(tag)
            if normalised and normalised not in seen:
                seen.append(normalised)
        return seen

    @field_validator("title", "content")
    @classmethod
    def _stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class MemoryRecord(BaseModel):
    """A stored memory row, without its embedding."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    project: str
    type: str
    title: str
    content: str
    tags: list[str]
    source: str | None
    # ``meta`` on the ORM model (``metadata`` is reserved by SQLAlchemy),
    # ``metadata`` everywhere in the API.
    metadata: dict[str, Any] = Field(
        default_factory=dict, validation_alias=AliasChoices("meta", "metadata")
    )
    created_at: datetime
    updated_at: datetime


class MemoryWriteResponse(BaseModel):
    memory: MemoryRecord
    created: bool = Field(
        description="False when an identical memory already existed and was refreshed."
    )


class SearchRequest(BaseModel):
    """Payload for ``POST /memory/search``."""

    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, Field(min_length=1, max_length=8_000)]
    project: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    type: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    tags: list[str] = Field(default_factory=list, max_length=16)
    top_k: int = Field(default_factory=lambda: get_settings().default_top_k, ge=1)
    mode: Literal["vector", "hybrid"] = Field(
        default="vector",
        description="'hybrid' fuses pgvector similarity with Postgres full-text ranking.",
    )
    min_similarity: float = Field(
        default=-1.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Opt-in cutoff: drop results below this cosine score. The default "
            "keeps every one of the top_k, because 'the closest thing I have' "
            "is usually what an agent wants even when it is a weak match."
        ),
    )

    @field_validator("project", "type")
    @classmethod
    def _slug(cls, value: str | None) -> str | None:
        return _normalise_key(value) if value else value

    @field_validator("tags")
    @classmethod
    def _tags(cls, values: list[str]) -> list[str]:
        return [_normalise_key(tag) for tag in values if tag.strip()]

    @field_validator("top_k")
    @classmethod
    def _cap(cls, value: int) -> int:
        return min(value, get_settings().max_top_k)


class SearchHit(MemoryRecord):
    similarity: float = Field(description="Cosine similarity in [-1, 1]; higher is closer.")
    rank: int


class SearchResponse(BaseModel):
    query: str
    project: str | None
    mode: str
    count: int
    results: list[SearchHit]


class ProjectContextResponse(BaseModel):
    project: str
    query: str
    total_memories: int
    returned: int
    groups: dict[str, list[SearchHit]]


class ProjectSummary(BaseModel):
    project: str
    memory_count: int
    types: dict[str, int]
    last_updated: datetime


class ProjectListResponse(BaseModel):
    count: int
    projects: list[ProjectSummary]


class MemoryListResponse(BaseModel):
    count: int
    total: int
    memories: list[MemoryRecord]


class DeleteResponse(BaseModel):
    deleted: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    schema_ready: bool = False
    embedding_model: str
    embedding_backend: str | None
    embedding_dim: int
    memories: int | None = None
    version: str
