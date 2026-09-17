"""Embedding generation.

The model is loaded exactly once per process and reused. Two interchangeable
backends produce the same ``all-MiniLM-L6-v2`` vectors:

* ``sentence-transformers`` — the reference implementation (PyTorch).
* ``fastembed`` — the same model exported to ONNX, ~10x smaller to install and
  comfortable inside a 512 MB free-tier container.

Both return L2-normalised 384-dimension vectors, so cosine similarity is a dot
product and pgvector's ``<=>`` operator returns ``1 - cosine_similarity``.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np

from app.utils.config import get_settings

logger = logging.getLogger(__name__)


class _Backend(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformersBackend:
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


class FastEmbedBackend:
    name = "fastembed"

    # fastembed uses fully-qualified Hugging Face ids.
    _ALIASES = {"all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"}

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=self._ALIASES.get(model_name, model_name))

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


class Embedder:
    """Chunk-aware wrapper around a backend."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._backend = _load_backend(settings.embedding_backend, settings.embedding_model)
        dim = len(self._backend.encode(["dimension probe"])[0])
        if dim != settings.embedding_dim:
            raise RuntimeError(
                f"Model {settings.embedding_model!r} produces {dim}-d vectors but the "
                f"schema expects {settings.embedding_dim}. Set EMBEDDING_DIM to {dim} "
                "and recreate the memory table."
            )
        logger.info(
            "Embedding model %s loaded via %s (%d dims).",
            settings.embedding_model,
            self._backend.name,
            dim,
        )

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def embed(self, text: str) -> list[float]:
        """Embed a single document or query.

        Long documents are split into word chunks, embedded individually and
        mean-pooled, so a 5 000-word architecture note is not truncated to its
        first paragraph by the model's 256-token window.
        """
        chunks = self._chunk(text)
        if not chunks:
            raise ValueError("Cannot embed empty text.")
        vectors = self._backend.encode(chunks)
        pooled = vectors.mean(axis=0)
        norm = float(np.linalg.norm(pooled))
        if norm < 1e-12:
            return vectors[0].tolist()
        return (pooled / norm).astype(np.float32).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        size = self._settings.embedding_chunk_words
        if len(words) <= size:
            return [" ".join(words)]
        chunks = [
            " ".join(words[i : i + size]) for i in range(0, len(words), size)
        ]
        return chunks[: self._settings.embedding_max_chunks]


def _load_backend(preference: str, model_name: str) -> _Backend:
    if preference == "sentence-transformers":
        return SentenceTransformersBackend(model_name)
    if preference == "fastembed":
        return FastEmbedBackend(model_name)

    # auto: prefer the lighter runtime, fall back to the reference one.
    try:
        return FastEmbedBackend(model_name)
    except Exception as exc:  # noqa: BLE001 - fastembed is optional
        logger.info("fastembed unavailable (%s); using sentence-transformers.", exc)
        return SentenceTransformersBackend(model_name)


_embedder: Embedder | None = None
_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Return the singleton embedder, loading the model on first use."""
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder


def embed(text: str) -> list[float]:
    """Embed ``text`` into a normalised 384-dimension vector."""
    return get_embedder().embed(text)
