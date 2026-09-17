"""Business logic for storing and retrieving memories.

Routes stay thin: everything that touches embeddings or SQL lives here.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import Select, delete, func, literal_column, select
from sqlalchemy.orm import Session

from app.models import Memory, content_fingerprint, utcnow
from app.schemas.memory_schema import CANONICAL_TYPES, MemoryWriteRequest, SearchRequest
from app.services.embedding_service import get_embedder
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

#: Reciprocal-rank-fusion constant. 60 is the value from the original RRF paper.
RRF_K = 60
#: How many rows each retrieval arm contributes before fusion.
CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATES = 20


@dataclass(slots=True)
class ScoredMemory:
    """A memory row plus its retrieval score."""

    memory: Memory
    similarity: float
    rank: int = 0


def embedding_input(title: str, content: str) -> str:
    """The text that actually gets embedded for a memory row."""
    return f"{title.strip()}\n\n{content.strip()}"


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def create_memory(session: Session, payload: MemoryWriteRequest) -> tuple[Memory, bool]:
    """Insert a memory, or refresh the existing identical one.

    Re-writing the same content into the same project is a no-op apart from
    ``updated_at``, which keeps agents from filling the store with duplicates
    when they re-summarise the same conversation.

    Returns ``(row, created)``.
    """
    fingerprint = content_fingerprint(
        payload.project, payload.type, payload.title, payload.content
    )
    existing = session.scalar(
        select(Memory).where(
            Memory.project == payload.project, Memory.content_hash == fingerprint
        )
    )
    if existing is not None:
        existing.tags = payload.tags or existing.tags
        existing.source = payload.source or existing.source
        existing.meta = {**existing.meta, **payload.metadata}
        existing.updated_at = utcnow()
        session.commit()
        logger.info("Refreshed duplicate memory %s in %s", existing.id, existing.project)
        return existing, False

    vector = get_embedder().embed(embedding_input(payload.title, payload.content))
    memory = Memory(
        id=uuid.uuid4(),
        project=payload.project,
        type=payload.type,
        title=payload.title,
        content=payload.content,
        tags=payload.tags,
        source=payload.source,
        meta=payload.metadata,
        content_hash=fingerprint,
        embedding=vector,
    )
    session.add(memory)
    session.commit()
    logger.info("Stored memory %s in %s (%s)", memory.id, memory.project, memory.type)
    return memory, True


def delete_memory(session: Session, memory_id: uuid.UUID) -> int:
    result = session.execute(delete(Memory).where(Memory.id == memory_id))
    session.commit()
    return int(result.rowcount or 0)


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def get_memory(session: Session, memory_id: uuid.UUID) -> Memory | None:
    return session.get(Memory, memory_id)


def list_memories(
    session: Session,
    project: str | None = None,
    type_: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Memory], int]:
    filters = []
    if project:
        filters.append(Memory.project == project)
    if type_:
        filters.append(Memory.type == type_)

    total = session.scalar(select(func.count()).select_from(Memory).where(*filters)) or 0
    rows = list(
        session.scalars(
            select(Memory)
            .where(*filters)
            .order_by(Memory.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    )
    return rows, total


def count_memories(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Memory)) or 0


def list_projects(session: Session) -> list[dict]:
    """One row per project with per-type counts."""
    rows = session.execute(
        select(
            Memory.project,
            Memory.type,
            func.count().label("n"),
            func.max(Memory.updated_at).label("last_updated"),
        ).group_by(Memory.project, Memory.type)
    ).all()

    projects: dict[str, dict] = {}
    for project, type_, n, last_updated in rows:
        entry = projects.setdefault(
            project,
            {"project": project, "memory_count": 0, "types": {}, "last_updated": last_updated},
        )
        entry["memory_count"] += n
        entry["types"][type_] = entry["types"].get(type_, 0) + n
        if last_updated > entry["last_updated"]:
            entry["last_updated"] = last_updated

    return sorted(projects.values(), key=lambda p: p["last_updated"], reverse=True)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def _apply_filters(
    stmt: Select,
    project: str | None,
    type_: str | None,
    tags: list[str],
) -> Select:
    if project:
        stmt = stmt.where(Memory.project == project)
    if type_:
        stmt = stmt.where(Memory.type == type_)
    if tags:
        stmt = stmt.where(Memory.tags.contains(tags))
    return stmt


def _vector_candidates(
    session: Session,
    vector: list[float],
    project: str | None,
    type_: str | None,
    tags: list[str],
    limit: int,
) -> list[ScoredMemory]:
    distance = Memory.embedding.cosine_distance(vector).label("distance")
    stmt = _apply_filters(select(Memory, distance), project, type_, tags)
    stmt = stmt.order_by(distance).limit(limit)
    return [
        ScoredMemory(memory=memory, similarity=float(1.0 - dist))
        for memory, dist in session.execute(stmt).all()
    ]


def _keyword_candidates(
    session: Session,
    query: str,
    project: str | None,
    type_: str | None,
    tags: list[str],
    limit: int,
) -> list[uuid.UUID]:
    """Ids ranked by Postgres full-text relevance (best first)."""
    # literal_column keeps the expression byte-identical to the GIN index
    # definition in db.schema_statements, so Postgres can actually use it.
    tsvector = literal_column(
        "to_tsvector('english', memory.title || ' ' || memory.content)"
    )
    tsquery = func.websearch_to_tsquery(literal_column("'english'"), query)
    rank = func.ts_rank_cd(tsvector, tsquery).label("rank")

    stmt = _apply_filters(select(Memory.id, rank), project, type_, tags)
    stmt = stmt.where(tsvector.op("@@")(tsquery)).order_by(rank.desc()).limit(limit)
    return [row[0] for row in session.execute(stmt).all()]


def _fetch_with_similarity(
    session: Session, ids: list[uuid.UUID], vector: list[float]
) -> list[ScoredMemory]:
    if not ids:
        return []
    distance = Memory.embedding.cosine_distance(vector).label("distance")
    rows = session.execute(select(Memory, distance).where(Memory.id.in_(ids))).all()
    return [ScoredMemory(memory=m, similarity=float(1.0 - d)) for m, d in rows]


def search_memory(session: Session, request: SearchRequest) -> list[ScoredMemory]:
    """Semantic (or hybrid) search over the store.

    ``vector`` mode is pure pgvector cosine similarity. ``hybrid`` mode fuses
    that ranking with Postgres full-text search using reciprocal rank fusion,
    which rescues exact-keyword matches (ticket ids, function names) that a
    384-dimension embedding can blur away.
    """
    vector = get_embedder().embed(request.query)
    pool = max(request.top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)

    vector_hits = _vector_candidates(
        session, vector, request.project, request.type, request.tags, pool
    )

    if request.mode == "vector":
        ranked = vector_hits
    else:
        keyword_ids = _keyword_candidates(
            session, request.query, request.project, request.type, request.tags, pool
        )
        ranked = _fuse(session, vector, vector_hits, keyword_ids)

    results: list[ScoredMemory] = []
    for hit in ranked:
        if hit.similarity < request.min_similarity:
            continue
        hit.rank = len(results) + 1
        results.append(hit)
        if len(results) == request.top_k:
            break
    return results


def _fuse(
    session: Session,
    vector: list[float],
    vector_hits: list[ScoredMemory],
    keyword_ids: list[uuid.UUID],
) -> list[ScoredMemory]:
    """Reciprocal rank fusion of the vector and keyword rankings."""
    by_id: dict[uuid.UUID, ScoredMemory] = {hit.memory.id: hit for hit in vector_hits}
    missing = [mid for mid in keyword_ids if mid not in by_id]
    for hit in _fetch_with_similarity(session, missing, vector):
        by_id[hit.memory.id] = hit

    scores: dict[uuid.UUID, float] = {}
    for position, hit in enumerate(vector_hits):
        scores[hit.memory.id] = scores.get(hit.memory.id, 0.0) + 1.0 / (RRF_K + position + 1)
    for position, memory_id in enumerate(keyword_ids):
        scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (RRF_K + position + 1)

    return sorted(
        (by_id[mid] for mid in scores if mid in by_id),
        key=lambda hit: (scores[hit.memory.id], hit.similarity),
        reverse=True,
    )


def get_project_context(
    session: Session,
    project: str,
    query: str | None = None,
    limit: int | None = None,
) -> dict:
    """Aggregate the most relevant memories of a project, grouped by type."""
    settings = get_settings()
    effective_query = (query or settings.context_query).strip() or settings.context_query
    effective_limit = limit or settings.context_limit

    total = (
        session.scalar(
            select(func.count()).select_from(Memory).where(Memory.project == project)
        )
        or 0
    )

    hits = search_memory(
        session,
        SearchRequest(
            query=effective_query,
            project=project,
            top_k=effective_limit,
            mode="hybrid",
        ),
    )

    groups: dict[str, list[ScoredMemory]] = {name: [] for name in CANONICAL_TYPES}
    groups["other"] = []
    for hit in hits:
        bucket = hit.memory.type if hit.memory.type in CANONICAL_TYPES else "other"
        groups[bucket].append(hit)

    return {
        "project": project,
        "query": effective_query,
        "total_memories": total,
        "returned": len(hits),
        "groups": groups,
    }
