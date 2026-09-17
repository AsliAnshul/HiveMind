# Hive Mind Memory Server

A shared long-term memory for AI agents. Claude and ChatGPT write structured
knowledge into one Postgres table, and either one can retrieve it later by
meaning rather than by exact words.

FastAPI + Postgres/pgvector + `all-MiniLM-L6-v2` embeddings. No Redis, no
Docker, no external vector database.

---

## 1. What it does

| Endpoint | Purpose |
| --- | --- |
| `POST /memory/write` | Embed and store one unit of knowledge |
| `POST /memory/search` | Semantic (or hybrid) search across memories |
| `GET /memory` | List memories, filtered by project/type |
| `GET /memory/{id}` | Fetch one memory |
| `DELETE /memory/{id}` | Delete one memory |
| `GET /projects` | Every project with its memory counts |
| `GET /projects/{project}/context` | Rehydrate a project, grouped by type |
| `GET /health` | Database + model readiness |
| `POST /mcp` | The same tools over MCP, for Claude to connect to directly |

Interactive docs at `/docs`, machine-readable schema at `/openapi.json`.

### Design decisions worth knowing

- **Writes are idempotent.** Byte-identical content in the same project returns
  the original row with `created: false` instead of duplicating it. Agents
  re-summarise the same conversation constantly; this keeps the store clean.
- **Long documents are chunked, not truncated.** MiniLM sees 256 tokens. A long
  note is split into word chunks, embedded, and mean-pooled, so a fact on page
  four still influences the vector.
- **Vectors are L2-normalised**, so pgvector's `<=>` is exactly
  `1 - cosine_similarity`. The API returns `similarity` in `[-1, 1]`.
- **Hybrid mode** (`"mode": "hybrid"`) fuses the vector ranking with Postgres
  full-text ranking via reciprocal rank fusion. Embeddings blur exact tokens
  like `OPS-4417` or `useMemo`; full-text does not. `/projects/{p}/context`
  always uses hybrid.
- **ANN search knobs are set per connection, not per query.** `hnsw.ef_search`
  and `iterative_scan` used to be `SET LOCAL` inside each search transaction,
  which cost two extra round trips on every request. Applied once at connection
  setup they are free, and search latency against a database in another region
  dropped from ~1300 ms to ~570 ms. They are applied with autocommit on, so a
  rolled-back transaction cannot quietly revert them.
- **Writes do not re-read their own row.** Timestamps are filled client-side as
  well as by `server_default`, so an insert needs no follow-up `SELECT`.
- **Two interchangeable embedding backends.** `sentence-transformers` (PyTorch,
  the reference) and `fastembed` (the same model in ONNX, ~6x smaller install).
  They produce identical vectors — verified at cosine 1.0 — so you can develop
  with one and deploy with the other against the same database.

---

## 2. Setup

Requires Python 3.11+ and a Postgres database with pgvector.

```bash
git clone <your-repo> hive-mind
cd hive-mind/backend

python3.11 -m venv .venv
source .venv/bin/activate

# Reference stack (PyTorch). On a CPU-only machine, install torch from the CPU
# index first to avoid ~2 GB of CUDA wheels:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# ...or the lightweight ONNX stack used in production:
#   pip install -r requirements-lite.txt   # then set EMBEDDING_BACKEND=fastembed

cp .env.example .env     # then fill in DATABASE_URL
python scripts/check_env.py
```

`check_env.py` is the fastest way to find out whether your setup is right: it
validates the connection string, reaches the database, confirms pgvector is
there, creates the schema, and loads the model — printing the specific fix for
whatever fails.

```
[  ok  ] DATABASE_URL is set
[  ok  ] Database reachable (Postgres 17.4, as postgres)
[  ok  ] pgvector installed (0.8.0)
[  ok  ] Schema ready (memory table holds 0 rows)
[  ok  ] API key auth enabled (1 key(s))
[  ok  ] Embedding model loaded via fastembed (384 dims)
```

### Configuration

Only `DATABASE_URL` is required. Everything else has a working default — see
`.env.example` for the full list. The ones that matter:

| Variable | Default | Notes |
| --- | --- | --- |
| `DATABASE_URL` | — | Postgres URI. `postgres://` and `postgresql://` are both accepted. |
| `API_KEYS` | empty | Comma-separated. Empty means **no auth**. Set it before going public. |
| `EMBEDDING_BACKEND` | `auto` | `auto` prefers `fastembed`, falls back to `sentence-transformers`. |
| `AUTO_MIGRATE` | `true` | Creates the extension, table and indexes on startup. |
| `VECTOR_INDEX_TYPE` | `hnsw` | `ivfflat` is supported too — read the note in §3 first. |
| `PUBLIC_BASE_URL` | empty | Your deployed URL. Required for ChatGPT Actions — it becomes `servers[]` in `openapi.json`. |
| `CONTEXT_LIMIT` | `10` | Entries returned by the project-context endpoint. |

---

## 3. Supabase setup

1. Create a project at [supabase.com](https://supabase.com) (free tier: 500 MB
   database, enough for roughly a million memories).
2. **Enable pgvector** — Dashboard → Database → Extensions → search `vector` →
   enable. Or run in the SQL editor:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Copy the connection string — Connect → **Session pooler**, and replace
   `[YOUR-PASSWORD]` with your database password.

   > **Take the pooler URI, not the direct one.** They look almost identical:
   >
   > ```
   > pooler  postgresql://postgres.<ref>:pw@aws-0-<region>.pooler.supabase.com:5432/postgres
   > direct  postgresql://postgres:pw@db.<ref>.supabase.co:5432/postgres
   > ```
   >
   > `db.<ref>.supabase.co` publishes **only an AAAA record**. It is reachable
   > from an IPv6-capable laptop and unreachable from Render and most other free
   > tiers, which have no IPv6 outbound — so the direct URI passes every local
   > test and then fails on deploy. `scripts/check_env.py` warns when the host
   > has no IPv4 address.

   - Port `5432` (session pooler), not `6543` (transaction pooler) — the
     transaction pooler disables prepared statements, which SQLAlchemy uses.
   - Supabase requires TLS: keep `?sslmode=require` on the URI.
4. Create the schema. Either let the app do it on boot (`AUTO_MIGRATE=true`,
   the default), or run it yourself:
   ```bash
   python scripts/bootstrap_db.py            # applies + verifies
   ```
   or paste `migrations/001_init.sql` into the Supabase SQL editor.

`scripts/bootstrap_db.py` prints the pgvector version, the row count and every
index, so you can confirm the store is real before writing anything to it.

> **Why the index defaults to hnsw, not ivfflat.** An ivfflat index partitions
> vectors into `lists` clusters at build time and searches only `probes` of
> them. Built before the table has data, it has nothing to cluster — and the
> failure is silent: you get *fewer rows than you asked for*, not an error. On a
> four-row test project, `"how do people log in?"` returned **zero** results
> under a freshly built ivfflat index and the correct three under hnsw. hnsw
> builds incrementally and needs no training pass.
>
> ivfflat is still fully supported (`VECTOR_INDEX_TYPE=ivfflat`) and is the
> better choice once you are past ~100 k memories and care more about index size
> than recall — the test suite passes against both. On pgvector ≥ 0.8 the
> iterative scan described below covers for an under-trained ivfflat index; on
> older pgvector it does not, which is the reason for the default. Either way,
> rebuild it once there is real data in the table:
> `REINDEX INDEX memory_embedding_ivfflat_idx;`
>
> Both indexes share a second trap: `WHERE project = ...` is applied *after* the
> index returns its candidates, so a filtered search can come back short. The
> server sets `iterative_scan = relaxed_order` (pgvector ≥ 0.8) on every search,
> which keeps walking the index until `top_k` rows survive the filter. The index
> name carries its type, so switching `VECTOR_INDEX_TYPE` really does build the
> new index, and startup warns about the leftover one.

---

## 4. Run the server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

First boot downloads the model (~90 MB) and creates the schema. Then:

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "database": true,
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_backend": "fastembed",
  "embedding_dim": 384,
  "memories": 0,
  "version": "1.0.0"
}
```

### No pgvector handy?

`scripts/local_pg.sh start` brings up a throwaway Postgres + pgvector on port
55432 with no root access and nothing installed system-wide (Debian/Ubuntu; it
unpacks the distro's pgvector package locally and uses PostgreSQL 18's
`extension_control_path`). It prints the `DATABASE_URL` to export. `stop` and
`destroy` do what they say. Development convenience only — point at Supabase
for anything real.

### Tests

```bash
pip install -r requirements-dev.txt
DATABASE_URL="postgresql://..." pytest -q
```

The embedding tests run anywhere. The API tests need a real pgvector database
and skip themselves without `DATABASE_URL`; they write to a throwaway project
and delete it afterwards.

---

## 5. Example requests

Set `API_KEYS=hive-dev-key` in `.env` and pass `-H "X-API-Key: hive-dev-key"`
with every call below (omit it if you left `API_KEYS` empty).

**Write a memory**

```bash
curl -X POST http://127.0.0.1:8000/memory/write \
  -H "Content-Type: application/json" \
  -d '{
    "project": "hive-mind",
    "type": "architecture",
    "title": "Auth design",
    "content": "Users sign in with email and password. The API issues a short-lived JWT access token plus a rotating refresh token in an httpOnly cookie.",
    "tags": ["auth", "backend"],
    "source": "claude"
  }'
```

```json
{
  "memory": {
    "id": "0f0f6c3e-...",
    "project": "hive-mind",
    "type": "architecture",
    "title": "Auth design",
    "content": "Users sign in with email...",
    "tags": ["auth", "backend"],
    "source": "claude",
    "metadata": {},
    "created_at": "2026-09-17T09:12:44.318Z",
    "updated_at": "2026-09-17T09:12:44.318Z"
  },
  "created": true
}
```

**Semantic search** — the query shares no words with the stored text, and the
right memory still comes first:

```bash
curl -X POST http://127.0.0.1:8000/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how do people log in?", "project": "hive-mind", "top_k": 5}'
```

```json
{
  "query": "how do people log in?",
  "project": "hive-mind",
  "mode": "vector",
  "count": 3,
  "results": [
    { "title": "Auth design",          "similarity": 0.419, "rank": 1, "...": "..." },
    { "title": "Customer call — Acme", "similarity": 0.240, "rank": 2, "...": "..." },
    { "title": "Coffee machine",       "similarity": 0.096, "rank": 3, "...": "..." }
  ]
}
```

**Hybrid search** — for exact identifiers an embedding would blur:

```bash
curl -X POST http://127.0.0.1:8000/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "OPS-4417", "top_k": 5, "mode": "hybrid"}'
```

**Project context** — everything an agent needs to resume work:

```bash
curl http://127.0.0.1:8000/projects/hive-mind/context
```

```json
{
  "project": "hive-mind",
  "query": "project overview architecture plan tasks",
  "total_memories": 12,
  "returned": 10,
  "groups": {
    "plan": [ ... ],
    "summary": [ ... ],
    "notes": [ ... ],
    "architecture": [ ... ],
    "other": [ ... ]
  }
}
```

**Other calls**

```bash
curl "http://127.0.0.1:8000/memory?project=hive-mind&limit=20"
curl http://127.0.0.1:8000/projects
curl -X DELETE http://127.0.0.1:8000/memory/<uuid>
```

---

## 6. Wiring up the agents

Full instructions, including the GPT instruction block to paste and a test
that proves both assistants see the same rows, are in
[`../CONNECT.md`](../CONNECT.md). The short version:

**Claude** — the server speaks MCP itself, at `/mcp`. One command, no local
process:

```bash
claude mcp add --transport http hive-mind https://your-app.onrender.com/mcp \
  --header "X-API-Key: your-key"
```

Tools: `remember`, `recall`, `project_context`, `list_projects`, `forget`. They
call the service layer in-process, so there is no HTTP round trip back to the
API. `integrations/mcp_server.py` remains as a stdio bridge for clients that
cannot do remote MCP.

**ChatGPT** — create a custom GPT, add an Action, import
`https://your-app.onrender.com/openapi.json`, and set authentication to *API
Key* with custom header `X-API-Key`. Requires `PUBLIC_BASE_URL` to be set on the
deployment, or the schema has no `servers[]` entry and ChatGPT refuses it.

Both assistants then read and write the same rows, which is the entire point.

---

## 7. Deployment

See [`../deploy.md`](../deploy.md) for every free hosting option, with the
tradeoffs and step-by-step instructions.

---

## 8. Project layout

```
backend/
├── app/
│   ├── main.py                     FastAPI app, lifespan, health, error handlers
│   ├── mcp_server.py               MCP tools served at /mcp
│   ├── db.py                       Engine, sessions, idempotent schema DDL
│   ├── models.py                   SQLAlchemy model for the memory table
│   ├── routes/
│   │   ├── memory.py               /memory/*
│   │   └── projects.py             /projects/*
│   ├── services/
│   │   ├── embedding_service.py    Model singleton, chunking, both backends
│   │   └── memory_service.py       Writes, search, fusion, project context
│   ├── schemas/memory_schema.py    Pydantic request/response models
│   └── utils/
│       ├── config.py               Settings singleton
│       └── security.py             X-API-Key dependency
├── migrations/001_init.sql         Generated DDL for the Supabase SQL editor
├── scripts/
│   ├── check_env.py                Validate .env end to end
│   ├── bootstrap_db.py             Apply + verify the schema
│   ├── local_pg.sh                 Rootless local Postgres + pgvector
│   ├── prefetch_model.py           Download weights at build time
│   └── print_schema.py             Regenerate 001_init.sql
├── tests/
│   ├── test_embedding.py           No database needed
│   ├── test_config.py              Settings parsing, no database
│   ├── test_api.py                 End-to-end against pgvector
│   └── test_mcp.py                 The /mcp endpoint, both directions
├── requirements.txt                Reference stack (PyTorch)
├── requirements-lite.txt           Deployment stack (ONNX)
└── requirements-dev.txt
```

Routes never touch embeddings or SQL; services never touch HTTP.
