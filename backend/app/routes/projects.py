"""Project-level aggregation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.routes.memory import _to_hit
from app.schemas.memory_schema import (
    ProjectContextResponse,
    ProjectListResponse,
    ProjectSummary,
)
from app.services import memory_service
from app.utils.config import get_settings

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get(
    "",
    response_model=ProjectListResponse,
    summary="List projects",
    operation_id="list_projects",
)
def list_projects(db: Session = Depends(get_db)) -> ProjectListResponse:
    projects = memory_service.list_projects(db)
    return ProjectListResponse(
        count=len(projects),
        projects=[ProjectSummary.model_validate(p) for p in projects],
    )


@router.get(
    "/{project}/context",
    response_model=ProjectContextResponse,
    summary="Aggregated project context",
    operation_id="project_context",
)
def project_context(
    project: str,
    query: str | None = Query(
        default=None,
        description="Override the default relevance query used to rank entries.",
    ),
    limit: int = Query(default=0, ge=0, le=50, description="0 uses CONTEXT_LIMIT."),
    db: Session = Depends(get_db),
) -> ProjectContextResponse:
    """Everything an agent needs to resume work on a project, grouped by type."""
    settings = get_settings()
    normalised = project.strip().lower()
    context = memory_service.get_project_context(
        db, normalised, query=query, limit=limit or settings.context_limit
    )
    if context["total_memories"] == 0:
        raise HTTPException(status_code=404, detail=f"No memories for project {normalised!r}.")

    return ProjectContextResponse.model_validate(
        {
            **context,
            "groups": {
                name: [_to_hit(hit) for hit in hits]
                for name, hits in context["groups"].items()
            },
        }
    )
