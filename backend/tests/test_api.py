"""End-to-end API tests against a real pgvector database."""

from __future__ import annotations

import pytest

MEMORIES = [
    {
        "type": "architecture",
        "title": "Auth design",
        "content": (
            "Users sign in with email and password. The API issues a short-lived "
            "JWT access token and a rotating refresh token stored in an httpOnly "
            "cookie. Sessions are revoked by bumping a token version column."
        ),
        "tags": ["auth", "backend"],
        "source": "claude",
    },
    {
        "type": "plan",
        "title": "Q3 roadmap",
        "content": (
            "Ship the billing rewrite, migrate invoice reconciliation off the "
            "nightly cron onto Stripe webhooks, then start the mobile beta."
        ),
        "tags": ["roadmap"],
        "source": "chatgpt",
    },
    {
        "type": "notes",
        "title": "Coffee machine",
        "content": "The office coffee machine is broken again. Ticket OPS-4417 filed.",
        "tags": ["office"],
        "source": "claude",
    },
    {
        "type": "summary",
        "title": "Customer call — Acme",
        "content": (
            "Acme wants SSO before they renew. They also asked about audit logs "
            "and a sandbox environment for their integration team."
        ),
        "tags": ["sales"],
        "source": "chatgpt",
    },
]


@pytest.fixture(scope="module", autouse=True)
def seeded(client, project: str) -> list[dict]:
    written = []
    for memory in MEMORIES:
        response = client.post("/memory/write", json={"project": project, **memory})
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["created"] is True
        written.append(body["memory"])
    return written


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] is True
    assert body["embedding_dim"] == 384


def test_write_returns_full_row(client, seeded) -> None:
    row = seeded[0]
    assert row["project"]
    assert row["type"] == "architecture"
    assert row["tags"] == ["auth", "backend"]
    assert row["source"] == "claude"
    assert row["id"] and row["created_at"]


def test_write_is_idempotent(client, project: str) -> None:
    payload = {"project": project, **MEMORIES[0]}
    first = client.post("/memory/write", json=payload).json()
    second = client.post("/memory/write", json=payload)
    assert second.status_code == 201
    assert second.json()["created"] is False
    assert second.json()["memory"]["id"] == first["memory"]["id"]


def test_semantic_search_finds_paraphrase(client, project: str) -> None:
    """No shared keywords with the stored text — only meaning."""
    response = client.post(
        "/memory/search",
        json={"query": "how do people log in?", "project": project, "top_k": 1},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["title"] == "Auth design"
    assert results[0]["similarity"] > 0.2
    assert results[0]["rank"] == 1


def test_search_orders_by_similarity(client, project: str) -> None:
    results = client.post(
        "/memory/search",
        json={"query": "what did the customer ask for?", "project": project, "top_k": 4},
    ).json()["results"]
    scores = [hit["similarity"] for hit in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["title"] == "Customer call — Acme"


def test_hybrid_search_rescues_exact_identifiers(client, project: str) -> None:
    """An embedding blurs 'OPS-4417'; full-text does not."""
    results = client.post(
        "/memory/search",
        json={"query": "OPS-4417", "project": project, "top_k": 3, "mode": "hybrid"},
    ).json()["results"]
    assert "Coffee machine" in [hit["title"] for hit in results]


def test_project_filter_isolates_memories(client, project: str) -> None:
    response = client.post(
        "/memory/search",
        json={"query": "authentication", "project": f"{project}-empty", "top_k": 5},
    )
    assert response.json()["results"] == []


def test_type_and_tag_filters(client, project: str) -> None:
    results = client.post(
        "/memory/search",
        json={"query": "plans", "project": project, "type": "plan", "top_k": 5},
    ).json()["results"]
    assert [hit["type"] for hit in results] == ["plan"]

    tagged = client.post(
        "/memory/search",
        json={"query": "anything", "project": project, "tags": ["sales"], "top_k": 5},
    ).json()["results"]
    assert [hit["title"] for hit in tagged] == ["Customer call — Acme"]


def test_top_k_is_respected(client, project: str) -> None:
    results = client.post(
        "/memory/search", json={"query": "project", "project": project, "top_k": 2}
    ).json()["results"]
    assert len(results) == 2


def test_project_context_is_grouped(client, project: str) -> None:
    body = client.get(f"/projects/{project}/context").json()
    assert body["project"] == project
    assert body["total_memories"] >= 4
    assert set(body["groups"]) == {"plan", "summary", "notes", "architecture", "other"}
    assert body["groups"]["plan"][0]["title"] == "Q3 roadmap"
    assert body["groups"]["architecture"][0]["title"] == "Auth design"
    assert body["returned"] == sum(len(v) for v in body["groups"].values())


def test_unknown_project_context_is_404(client) -> None:
    assert client.get("/projects/does-not-exist-xyz/context").status_code == 404


def test_projects_listing(client, project: str) -> None:
    body = client.get("/projects").json()
    entry = next(p for p in body["projects"] if p["project"] == project)
    assert entry["memory_count"] >= 4
    assert entry["types"]["plan"] == 1


def test_list_get_and_delete(client, project: str) -> None:
    listing = client.get("/memory", params={"project": project}).json()
    assert listing["total"] >= 4

    memory_id = listing["memories"][0]["id"]
    assert client.get(f"/memory/{memory_id}").json()["id"] == memory_id

    assert client.delete(f"/memory/{memory_id}").json() == {"deleted": 1}
    assert client.get(f"/memory/{memory_id}").status_code == 404
    assert client.delete(f"/memory/{memory_id}").status_code == 404


def test_auth_is_enforced(client, project: str) -> None:
    """The fixture client is authenticated; a bare request must not be."""
    unauthenticated = client.post(
        "/memory/search", json={"query": "anything"}, headers={"X-API-Key": ""}
    )
    assert unauthenticated.status_code == 401

    wrong_key = client.post(
        "/memory/search", json={"query": "anything"}, headers={"X-API-Key": "nope"}
    )
    assert wrong_key.status_code == 403


def test_validation_rejects_empty_content(client, project: str) -> None:
    response = client.post(
        "/memory/write",
        json={"project": project, "type": "notes", "title": "x", "content": "   "},
    )
    assert response.status_code == 422
