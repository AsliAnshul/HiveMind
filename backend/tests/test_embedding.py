"""Embedding behaviour: no database required."""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/placeholder")

from app.services.embedding_service import get_embedder  # noqa: E402


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


def test_vector_is_normalised_and_correct_size(embedder) -> None:
    vector = embedder.embed("the cap table needs a new share class")
    assert len(vector) == 384
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-4)


def test_embedding_is_deterministic(embedder) -> None:
    text = "postgres connection pooling strategy"
    assert cosine(embedder.embed(text), embedder.embed(text)) > 0.999


def test_semantics_beat_keywords(embedder) -> None:
    query = embedder.embed("how do we authenticate users?")
    related = embedder.embed("login is handled with JWT tokens issued at sign-in")
    unrelated = embedder.embed("the office coffee machine is broken again")
    assert cosine(query, related) > cosine(query, unrelated)


def test_many_chunks_encode_without_a_ragged_batch(embedder) -> None:
    """Regression: batching long chunks raised ValueError inside fastembed.

    A document long enough to need several chunks used to fail with
    "inhomogeneous shape" — a 500 on any sufficiently long memory, on the
    backend the deployment actually runs.
    """
    long_document = "the deployment pipeline runs on push to main. " * 200
    chunks = embedder._chunk(long_document)
    assert len(chunks) > 1, "test needs a document that actually splits"

    vector = embedder.embed(long_document)
    assert len(vector) == 384
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-4)


def test_long_text_is_chunked_not_truncated(embedder) -> None:
    """A fact buried 1 000 words into a document still influences the vector."""
    filler = "the deployment pipeline runs on push to main. " * 120
    buried = filler + " The billing service reconciles Stripe invoices nightly."
    query = embedder.embed("stripe invoice reconciliation")
    assert cosine(query, embedder.embed(buried)) > cosine(query, embedder.embed(filler))
