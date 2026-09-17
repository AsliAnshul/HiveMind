"""Memory endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.memory_schema import (
    DeleteResponse,
    MemoryListResponse,
    MemoryRecord,
    MemoryWriteRequest,
    MemoryWriteResponse,
    SearchRequest,
    SearchResponse,
)
from app.services import memory_service

router = APIRouter(prefix="/memory", tags=["memory"])


def _to_hit(scored: memory_service.ScoredMemory) -> dict:
    record = MemoryRecord.model_validate(scored.memory).model_dump()
    record["similarity"] = round(scored.similarity, 6)
    record["rank"] = scored.rank
    return record


@router.post(
    "/write",
    response_model=MemoryWriteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store a memory",
    operation_id="write_memory",
)
def write_memory(
    payload: MemoryWriteRequest, db: Session = Depends(get_db)
) -> MemoryWriteResponse:
    """Embed and store one unit of knowledge.

    Writing byte-identical content to the same project twice returns the
    original row with ``created: false`` instead of duplicating it.
    """
    memory, created = memory_service.create_memory(db, payload)
    return MemoryWriteResponse(
        memory=MemoryRecord.model_validate(memory), created=created
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search",
    operation_id="search_memory",
)
def search_memory(request: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    """Embed the query and return the closest memories by cosine similarity."""
    hits = memory_service.search_memory(db, request)
    return SearchResponse.model_validate(
        {
            "query": request.query,
            "project": request.project,
            "mode": request.mode,
            "count": len(hits),
            "results": [_to_hit(hit) for hit in hits],
        }
    )


@router.get(
    "",
    response_model=MemoryListResponse,
    summary="List memories",
    operation_id="list_memories",
)
def list_memories(
    project: str | None = Query(default=None),
    type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> MemoryListResponse:
    rows, total = memory_service.list_memories(db, project, type, limit, offset)
    return MemoryListResponse(
        count=len(rows),
        total=total,
        memories=[MemoryRecord.model_validate(row) for row in rows],
    )


@router.get(
    "/{memory_id}",
    response_model=MemoryRecord,
    summary="Fetch one memory",
    operation_id="get_memory",
)
def get_memory(memory_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryRecord:
    memory = memory_service.get_memory(db, memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return MemoryRecord.model_validate(memory)


@router.delete(
    "/{memory_id}",
    response_model=DeleteResponse,
    summary="Delete a memory",
    operation_id="delete_memory",
)
def delete_memory(memory_id: uuid.UUID, db: Session = Depends(get_db)) -> DeleteResponse:
    deleted = memory_service.delete_memory(db, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return DeleteResponse(deleted=deleted)
