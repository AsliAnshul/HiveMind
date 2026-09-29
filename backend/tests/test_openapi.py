"""The schema ChatGPT imports.

Two things broke a real import and are regression-tested here: the API key
header appearing as an operation parameter, and OpenAPI 3.1's nullable form.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.openapi_compat import OPENAPI_30_VERSION, to_openapi_30


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
    response = client.get("/openapi-3.0.json")
    assert response.status_code == 200
    return response.json()


def test_version_is_30(spec) -> None:
    assert spec["openapi"] == OPENAPI_30_VERSION


def test_no_31_style_nullables_remain(spec) -> None:
    """The exact construct ChatGPT's 3.0 parser rejects."""
    offenders = [
        path
        for path, node in walk(spec)
        if isinstance(node.get("anyOf"), list)
        and any(isinstance(b, dict) and b.get("type") == "null" for b in node["anyOf"])
    ]
    assert offenders == []

    list_types = [
        path for path, node in walk(spec) if isinstance(node.get("type"), list)
    ]
    assert list_types == []


def test_nullable_is_preserved_not_dropped(spec) -> None:
    """Optional fields must still be optional, just spelled the 3.0 way."""
    search = spec["components"]["schemas"]["SearchRequest"]["properties"]
    assert search["project"]["nullable"] is True
    assert search["project"]["type"] == "string"


def test_api_key_header_is_not_an_operation_parameter(client) -> None:
    """Otherwise a GPT asks the model to invent the key instead of using auth."""
    for document in (client.get("/openapi.json").json(), client.get("/openapi-3.0.json").json()):
        for path, operations in document["paths"].items():
            for method, operation in operations.items():
                names = {
                    p.get("name", "").lower() for p in operation.get("parameters", [])
                }
                assert "x-api-key" not in names, f"{method.upper()} {path}"
                assert "x_api_key" not in names, f"{method.upper()} {path}"


def test_all_operations_survive_the_downgrade(client, spec) -> None:
    original = client.get("/openapi.json").json()
    def ids(doc):
        return sorted(op["operationId"] for p in doc["paths"].values() for op in p.values())
    assert ids(spec) == ids(original)


def test_refs_stay_resolvable(spec) -> None:
    """A $ref with sibling keywords is invalid in 3.0; it must be wrapped."""
    defined = set(spec["components"]["schemas"])
    for path, node in walk(spec):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            assert ref.split("/")[-1] in defined, path
            assert set(node) == {"$ref"}, f"$ref with siblings at {path}"


def test_both_documents_validate_against_the_spec(client) -> None:
    """Caught info.summary, which is 3.1-only and which hand-written checks missed."""
    validate = pytest.importorskip("openapi_spec_validator").validate
    for url in ("/openapi.json", "/openapi-3.0.json"):
        validate(client.get(url).json())


def test_info_summary_is_folded_into_description() -> None:
    converted = to_openapi_30(
        {
            "openapi": "3.1.0",
            "info": {"title": "t", "version": "1", "summary": "S", "description": "D"},
        }
    )
    assert "summary" not in converted["info"]
    assert converted["info"]["description"] == "S\n\nD"


def test_converter_handles_a_bare_null_union() -> None:
    converted = to_openapi_30({"openapi": "3.1.0", "x": {"anyOf": [{"type": "null"}]}})
    assert converted["x"] == {"nullable": True}
