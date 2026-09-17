"""The /mcp endpoint — what Claude connects to after deployment.

These run against the mounted MCP app through the same TestClient, so they
cover the auth middleware, the path handling and the tools themselves.
"""

from __future__ import annotations

import json
import uuid

import pytest

from tests.conftest import TEST_API_KEY

INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "1.0"},
    },
}
MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _parse(response) -> dict:
    """Read a JSON-RPC reply out of either a JSON body or an SSE stream."""
    body = response.text
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for line in body.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:])
        raise AssertionError(f"no data frame in SSE response: {body!r}")
    return json.loads(body)


@pytest.fixture
def mcp(client):
    """A callable that sends one JSON-RPC request to /mcp and returns the reply."""

    def send(method: str, params: dict | None = None, request_id: int = 2) -> dict:
        response = client.post(
            "/mcp",
            headers=MCP_HEADERS,
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            },
        )
        assert response.status_code == 200, response.text
        return _parse(response)

    return send


def test_initialize_reports_the_server(client) -> None:
    response = client.post("/mcp", headers=MCP_HEADERS, json=INIT)
    assert response.status_code == 200
    result = _parse(response)["result"]
    assert result["serverInfo"]["name"] == "hive-mind"


def test_both_url_spellings_work(client) -> None:
    """A bare /mcp must not 307 — some clients drop the body on a redirect."""
    for path in ("/mcp", "/mcp/"):
        response = client.post(path, headers=MCP_HEADERS, json=INIT)
        assert response.status_code == 200, path


def test_auth_is_enforced_on_mcp(client) -> None:
    missing = client.post("/mcp", headers={**MCP_HEADERS, "X-API-Key": ""}, json=INIT)
    assert missing.status_code == 401

    wrong = client.post("/mcp", headers={**MCP_HEADERS, "X-API-Key": "nope"}, json=INIT)
    assert wrong.status_code == 403


def test_bearer_token_is_accepted(client) -> None:
    """Not every MCP client can send a custom header; most can send a bearer."""
    response = client.post(
        "/mcp",
        headers={**MCP_HEADERS, "X-API-Key": "", "Authorization": f"Bearer {TEST_API_KEY}"},
        json=INIT,
    )
    assert response.status_code == 200


def test_tools_are_exposed(mcp) -> None:
    tools = {tool["name"] for tool in mcp("tools/list")["result"]["tools"]}
    assert tools == {"remember", "recall", "project_context", "list_projects", "forget"}


def test_write_over_mcp_is_readable_over_rest(client, mcp, project: str) -> None:
    """The whole point: one assistant writes, the other reads."""
    title = f"Cross-agent check {uuid.uuid4().hex[:6]}"
    written = mcp(
        "tools/call",
        {
            "name": "remember",
            "arguments": {
                "project": project,
                "type": "architecture",
                "title": title,
                "content": (
                    "Deployment runs on a free container with the database "
                    "hosted separately, and embeddings computed in-process."
                ),
                "source": "claude",
            },
        },
    )
    payload = json.loads(written["result"]["content"][0]["text"])
    assert payload["created"] is True

    # ...and now read it back through the REST API, in different words.
    results = client.post(
        "/memory/search",
        json={"query": "where does this thing run?", "project": project, "top_k": 3},
    ).json()["results"]
    match = next(hit for hit in results if hit["title"] == title)
    assert match["source"] == "claude"


def test_rest_write_is_visible_over_mcp(client, mcp, project: str) -> None:
    title = f"Written by the other side {uuid.uuid4().hex[:6]}"
    client.post(
        "/memory/write",
        json={
            "project": project,
            "type": "plan",
            "title": title,
            "content": "Invoices will be reconciled from webhooks rather than a nightly job.",
            "source": "chatgpt",
        },
    )
    recalled = mcp(
        "tools/call",
        {
            "name": "recall",
            "arguments": {
                "query": "how do we handle invoice reconciliation?",
                "project": project,
                "top_k": 3,
            },
        },
    )
    payload = json.loads(recalled["result"]["content"][0]["text"])
    titles = [hit["title"] for hit in payload["results"]]
    assert title in titles


def test_project_context_over_mcp(client, mcp, project: str) -> None:
    payload = json.loads(
        mcp("tools/call", {"name": "project_context", "arguments": {"project": project}})[
            "result"
        ]["content"][0]["text"]
    )
    assert payload["project"] == project
    assert set(payload["groups"]) == {"plan", "summary", "notes", "architecture", "other"}


def test_forget_rejects_a_bad_uuid(mcp) -> None:
    payload = json.loads(
        mcp("tools/call", {"name": "forget", "arguments": {"memory_id": "not-a-uuid"}})[
            "result"
        ]["content"][0]["text"]
    )
    assert payload["deleted"] == 0
    assert "valid UUID" in payload["error"]
