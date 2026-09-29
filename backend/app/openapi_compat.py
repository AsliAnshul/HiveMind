"""Downgrade the OpenAPI document from 3.1 to 3.0.3.

FastAPI emits OpenAPI 3.1, whose JSON Schema dialect expresses an optional
value as ``anyOf: [{...}, {"type": "null"}]``. ChatGPT's Action importer reads
3.0, where nullability is the ``nullable: true`` keyword instead, and rejects or
mangles the 3.1 form.

``/openapi.json`` stays the accurate 3.1 document. ``/openapi-3.0.json`` serves
the translation, so importing into a custom GPT works without hand-editing a
schema.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

OPENAPI_30_VERSION = "3.0.3"


def _convert(node: Any) -> Any:
    if isinstance(node, list):
        return [_convert(item) for item in node]
    if not isinstance(node, dict):
        return node

    result = {key: _convert(value) for key, value in node.items()}

    # anyOf: [X, {"type": "null"}]  ->  X with nullable: true
    branches = result.get("anyOf")
    if isinstance(branches, list):
        non_null = [
            b for b in branches if not (isinstance(b, dict) and b.get("type") == "null")
        ]
        if len(non_null) != len(branches):
            if len(non_null) == 1 and isinstance(non_null[0], dict):
                merged = {**non_null[0]}
                # A sibling $ref cannot carry keywords in 3.0, so wrap it.
                if "$ref" in merged:
                    result.pop("anyOf")
                    result["allOf"] = [{"$ref": merged.pop("$ref")}]
                    result.update(merged)
                else:
                    result.pop("anyOf")
                    result.update(merged)
                result["nullable"] = True
            elif non_null:
                result["anyOf"] = non_null
                result["nullable"] = True
            else:
                result.pop("anyOf")
                result["nullable"] = True

    # type: ["string", "null"]  ->  type: "string", nullable: true
    declared = result.get("type")
    if isinstance(declared, list):
        concrete = [t for t in declared if t != "null"]
        result["type"] = concrete[0] if concrete else "object"
        if len(concrete) != len(declared):
            result["nullable"] = True

    # 3.1 keywords with no 3.0 equivalent
    if "const" in result:
        result["enum"] = [result.pop("const")]
    if isinstance(result.get("examples"), list):
        examples = result.pop("examples")
        if examples:
            result["example"] = examples[0]
    for unsupported in ("prefixItems", "$schema", "unevaluatedProperties"):
        result.pop(unsupported, None)
    if result.get("exclusiveMinimum") is not None and not isinstance(
        result["exclusiveMinimum"], bool
    ):
        result["minimum"] = result.pop("exclusiveMinimum")
        result["exclusiveMinimum"] = True
    if result.get("exclusiveMaximum") is not None and not isinstance(
        result["exclusiveMaximum"], bool
    ):
        result["maximum"] = result.pop("exclusiveMaximum")
        result["exclusiveMaximum"] = True

    return result


def to_openapi_30(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a 3.0.3 rendering of a FastAPI-generated 3.1 document."""
    converted = _convert(deepcopy(spec))
    converted["openapi"] = OPENAPI_30_VERSION

    # Document-level fields that only exist in 3.1.
    info = converted.get("info")
    if isinstance(info, dict):
        summary = info.pop("summary", None)
        if summary:
            description = info.get("description")
            info["description"] = f"{summary}\n\n{description}" if description else summary
        license_ = info.get("license")
        if isinstance(license_, dict):
            license_.pop("identifier", None)
    converted.pop("webhooks", None)
    converted.pop("$self", None)

    return converted
