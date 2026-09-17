# Hive Mind

A shared long-term memory for AI assistants. ChatGPT and Claude write
structured knowledge into one Postgres table, and either one retrieves it later
by meaning rather than by exact words — so what one of them figures out, the
other can use.

FastAPI · Postgres + pgvector · `all-MiniLM-L6-v2` · no Redis, no Docker, no
external vector database. A standalone project, unrelated to anything beside it.

```
POST /memory/write               store a plan, summary, note or architecture doc
POST /memory/search              semantic (or hybrid) search
GET  /projects/{p}/context       rehydrate a whole project, grouped by type
     /mcp                        the same tools over MCP, for Claude
     /openapi.json               the same API as a ChatGPT Action
```

Both assistants connect to the deployed URL directly. Nothing to install
locally for either one.

- **[CONNECT.md](CONNECT.md)** — wiring up ChatGPT and Claude, with the GPT
  instructions to paste and a test that proves they share one brain.
- **[deploy.md](deploy.md)** — every free hosting option, with real numbers.
- **[backend/README.md](backend/README.md)** — setup, Supabase, example
  requests, design decisions.

Quick start:

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lite.txt
cp .env.example .env            # fill in DATABASE_URL
python scripts/check_env.py     # verifies everything before you start
uvicorn app.main:app --reload
```

No pgvector to hand? `backend/scripts/local_pg.sh start` brings one up with no
root access and nothing installed system-wide.
