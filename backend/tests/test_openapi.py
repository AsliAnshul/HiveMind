"""The schema ChatGPT imports.

Three things broke a real import and are regression-tested here: the API key
header appearing as an operation parameter, an untyped object response, and the
document's declared OpenAPI version — ChatGPT accepts 3.1.0/3.1.1 only.
"""

from __future__ import annotations

from typing import Any

import pytest


def walk(node: Any, path: str = ""):
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from walk(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, f"{path}/{index}")


@pytest.fixture(scope="module")
def spec(client) -> dict:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def test_version_is_31(spec) -> None:
    """ChatGPT rejects anything else: "Input should be '3.1.1' or '3.1.0'"."""
    assert spec["openapi"] in {"3.1.0", "3.1.1"}


def test_document_validates_against_the_spec(spec) -> None:
    validate = pytest.importorskip("openapi_spec_validator").validate
    validate(spec)


def test_api_key_header_is_not_an_operation_parameter(spec) -> None:
    """Otherwise a GPT asks the model to invent the key instead of using auth."""
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            names = {p.get("name", "").lower() for p in operation.get("parameters", [])}
            assert "x-api-key" not in names, f"{method.upper()} {path}"
            assert "x_api_key" not in names, f"{method.upper()} {path}"


def test_no_response_is_an_untyped_object(spec) -> None:
    """"object schema missing properties" — the warning the banner produced."""
    offenders = []
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            for status, response in operation.get("responses", {}).items():
                for media in response.get("content", {}).values():
                    schema = media.get("schema", {})
                    if (
                        schema.get("type") == "object"
                        and "properties" not in schema
                        and "additionalProperties" not in schema
                        and "$ref" not in schema
                    ):
                        offenders.append(f"{method.upper()} {path} -> {status}")
    assert offenders == []


def test_banner_response_is_typed(client) -> None:
    body = client.get("/").json()
    assert set(body) == {"service", "version", "docs", "endpoints"}
    assert "POST /mcp" in body["endpoints"]


def test_every_operation_has_a_unique_id(spec) -> None:
    ids = [op["operationId"] for p in spec["paths"].values() for op in p.values()]
    assert len(ids) == len(set(ids))
