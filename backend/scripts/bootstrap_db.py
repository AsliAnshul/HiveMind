"""Create the extension, table and indexes, then report what exists.

    python scripts/bootstrap_db.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text  # noqa: E402

from app.db import ensure_schema, get_engine  # noqa: E402


def main() -> None:
    ensure_schema()
    with get_engine().connect() as conn:
        extension = conn.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        indexes = conn.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'memory' ORDER BY 1")
        ).scalars()
        rows = conn.scalar(text("SELECT count(*) FROM memory"))

    if not extension:
        raise SystemExit(
            "pgvector is not installed. Run 'CREATE EXTENSION vector;' as a superuser."
        )
    print(f"pgvector      : {extension}")
    print(f"memory rows   : {rows}")
    print("indexes       :")
    for name in indexes:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
