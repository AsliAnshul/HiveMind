"""SQLAlchemy models for the hive mind memory store."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.utils.config import get_settings

EMBEDDING_DIM = get_settings().embedding_dim


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    """Timestamp for new rows.

    Filled client-side as well as by ``server_default`` so that an insert does
    not need a follow-up SELECT to learn its own timestamps — one fewer round
    trip per write, which matters when the database is in another region.
    """
    return datetime.now(tz=timezone.utc)


def content_fingerprint(project: str, type_: str, title: str, content: str) -> str:
    """Stable hash used to make writes idempotent within a project."""
    payload = "\x1f".join(
        part.strip() for part in (project, type_, title, content)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Memory(Base):
    """A single unit of knowledge written by an agent."""

    __tablename__ = "memory"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}", default=list
    )
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}", default=dict
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=utcnow,
        onupdate=utcnow,
    )

    __table_args__ = (
        UniqueConstraint("project", "content_hash", name="memory_project_hash_key"),
        Index("memory_project_idx", "project"),
        Index("memory_type_idx", "type"),
        Index("memory_created_at_idx", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Memory {self.id} {self.project}/{self.type}: {self.title!r}>"
