"""Download the embedding model at build time.

Free-tier hosts restart containers often; downloading ~90 MB of weights on the
first request after every cold start is the difference between a 400 ms and a
40 s response. Run this in the build step so the weights ship with the image.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/build-time-placeholder")

from app.services.embedding_service import get_embedder  # noqa: E402


def main() -> None:
    embedder = get_embedder()
    vector = embedder.embed("prefetch")
    print(f"Model ready via {embedder.backend_name} ({len(vector)} dims).")


if __name__ == "__main__":
    main()
