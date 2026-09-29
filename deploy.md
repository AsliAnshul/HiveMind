# Deploying the Hive Mind Memory Server — free options

The system has exactly two moving parts: **a Postgres database with pgvector**
and **one Python web process**. The embedding model runs inside that process,
so there is no third service to pay for.

Measured on this machine, which is what makes the free tiers viable:

| Backend | Install size | Peak RSS | Latency / embed |
| --- | --- | --- | --- |
| `fastembed` (ONNX) | ~400 MB | **251 MB** | 14.5 ms |
| `sentence-transformers` (PyTorch) | ~2.5 GB with CUDA wheels, ~900 MB CPU-only | 1 660 MB | 4.3 ms |

Both produce **identical vectors** (verified at cosine 1.0), so this is purely
a hosting decision — you can develop with PyTorch and deploy with ONNX against
the same database.

Measured against a Supabase project in `ap-northeast-2` from a laptop in India
(so most of this is intercontinental round-trip time, not the server):

| Call | Latency |
| --- | --- |
| `POST /memory/search` | ~570 ms |
| `POST /memory/write` | ~760 ms |
| `GET /projects/{p}/context` | ~820 ms |

Each request costs a handful of database round trips, so **deploy in the region
closest to your database** — that single choice matters more than anything else
here. A Supabase project in Seoul pairs with Render's Singapore region.

**Deploy with `fastembed`.** A 512 MB free tier cannot hold PyTorch; it holds
the ONNX runtime with room to spare. That is why `requirements-lite.txt` and
`EMBEDDING_BACKEND=fastembed` exist.

> Free-tier limits change often. The numbers below were accurate when this was
> written — check the provider's pricing page before you commit to one.

---

## Part 1 — The database (pick one)

| Service | Free tier | pgvector | Catch |
| --- | --- | --- | --- |
| **Supabase** ⭐ | 500 MB database, 2 projects | Built in, one click | Project pauses after 7 days with no activity; restore is one click |
| **Neon** | 0.5 GB storage, 191 compute-hours/month | `CREATE EXTENSION vector` works | Scales to zero; first query after idle takes ~1 s |
| **Aiven for PostgreSQL** | 1 GB, single node | Available | One free service per account |
| **Render Postgres** | 1 GB | Available | **Deleted after 30 days** — fine for a demo, not for a brain |

500 MB holds roughly a million memories: a 384-dimension vector is 1.5 KB, plus
the text itself.

**Recommended: Supabase.** Setup is in
[`backend/README.md` §3](backend/README.md#3-supabase-setup) — enable the
`vector` extension, copy the **session pooler** URI (port 5432), keep
`?sslmode=require`.

> The pooler part is not a preference. Supabase's direct-connection host,
> `db.<project-ref>.supabase.co`, resolves to an IPv6 address only, and Render
> has no IPv6 outbound. Deploy with that URI and every database call fails with
> a timeout while the same string works perfectly on your laptop. The pooler
> host has IPv4. Run `python backend/scripts/check_env.py` before deploying; it
> checks for exactly this.

Supabase pausing matters for a memory server that sits idle for a week, and it
is not a soft failure: a paused project loses its DNS entirely, so
`db.<ref>.supabase.co` stops resolving and the shared pooler answers
`FATAL: (ENOTFOUND) tenant/user ... not found`. Restoring it from the dashboard
takes about a minute and the data survives.

**Set up the keep-alive cron in Part 4 on day one.** `/health` runs a query, so
pinging it counts as activity and the project never pauses. The alternative is
Neon, which suspends and resumes transparently instead of pausing.

The service itself tolerates all of this: if the database is unreachable at
boot it logs a warning, keeps serving `/health` and `/mcp`, answers data
requests with 503, and creates the schema on the first request after the
database comes back — no redeploy.

---

## Part 2 — The web service

### Option A: Render — free, recommended ⭐

**What you get:** 512 MB RAM, 0.1 CPU, 750 instance-hours/month, TLS and a
`*.onrender.com` domain, deploy on git push.

**The catch:** the service spins down after 15 minutes with no traffic, and the
next request waits ~50 s for a cold start. Part 4 fixes that.

The repo already contains `render.yaml`, so Render configures itself:

1. Push this repo to GitHub.
2. [dashboard.render.com](https://dashboard.render.com) → **New → Blueprint** →
   pick the repo. Render reads `render.yaml`.
3. Set the two secrets it asks for (they are `sync: false`, so they never live
   in git):
   - `DATABASE_URL` — the Supabase session-pooler URI.
   - `API_KEYS` — one or more random strings, comma-separated with no spaces.
     Generate them with `openssl rand -hex 32`. Issue one per assistant
     (`claude-key,chatgpt-key`) so either can be revoked alone; both still see
     all the same rows.
   - `PUBLIC_BASE_URL` — leave it empty for now. It is the service's own
     `https://….onrender.com` URL, which Render only shows you once the service
     exists. Set it in Environment after the first deploy and let Render
     redeploy. Both assistants need it: ChatGPT refuses to import a schema with
     no `servers[]` entry, and the MCP endpoint answers an unrecognised `Host`
     header with 421, so Claude cannot connect without it.
4. Deploy. The build installs `requirements-lite.txt` and pre-downloads the
   model into `.model-cache` so cold starts do not re-fetch 90 MB of weights.
5. Verify — locally first if you like, with the same `DATABASE_URL`:
   ```bash
   python backend/scripts/check_env.py
   ```
   then against the deployment:
   ```bash
   curl https://your-app.onrender.com/health
   curl -X POST https://your-app.onrender.com/memory/write \
     -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"project":"demo","type":"notes","title":"First","content":"It works."}'
   ```

The schema is created on first boot (`AUTO_MIGRATE=true`). If the Supabase role
cannot `CREATE EXTENSION`, the log says so and you run that one line in the
Supabase SQL editor.

### Option B: Railway — easiest, not free for long

**What you get:** $5 of trial credit, then the Hobby plan at $5/month. This
service uses roughly $3–4/month of that, so it is effectively free for the
first month and cheap after. Call it what it is.

`railway.json` is in the repo:

1. [railway.app](https://railway.app) → **New Project → Deploy from GitHub**.
2. Variables → add `DATABASE_URL`, `API_KEYS`, `EMBEDDING_BACKEND=fastembed`,
   and `PUBLIC_BASE_URL` once the domain exists.
3. Settings → Networking → **Generate Domain**.

Railway does not sleep, so there are no cold starts — that is what you are
paying for.

### Option C: Hugging Face Spaces — free and generous, needs a container

**What you get:** 2 vCPU, **16 GB RAM**, free, no spin-down under 48 h of
inactivity. The only free option with enough memory to run the PyTorch backend.

The catch is that a FastAPI app needs the Docker SDK, so you add a Dockerfile —
the one thing the project otherwise avoids:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r backend/requirements-lite.txt && \
    python backend/scripts/prefetch_model.py
ENV HF_HOME=/app/.model-cache FASTEMBED_CACHE_PATH=/app/.model-cache
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860"]
```

Create a Space with SDK **Docker**, push the repo, and add `DATABASE_URL` and
`API_KEYS` as **Secrets** (not Variables). Note that a free Space is public by
default — set it to Private, and keep `API_KEYS` set regardless.

### Option D: Google Cloud Run — free tier, scales to zero

2 million requests and 360 000 GB-seconds per month, free, with no card charge
at this volume. Also container-based, same Dockerfile as above with
`--port 8080`. Set `--min-instances=0 --memory=1Gi --cpu=1`. Cold starts are
~5–10 s, much better than Render's, but the setup is the most involved here.

### Option E: Oracle Cloud Always Free — the best long-term free option

4 ARM cores and 24 GB RAM, free forever, no expiry. It is a plain VM: you
install Postgres + pgvector and the app yourself, run it under systemd, and put
Caddy in front for TLS. An hour of setup buys you a box that outclasses every
paid tier above. Worth it if this becomes infrastructure you actually rely on.

### Not suitable

- **Vercel / Netlify functions** — serverless cold starts reload the model on
  every invocation, and the 250 MB bundle limit rules out the ONNX runtime.
- **PythonAnywhere free** — no ASGI server and outbound network is allowlisted.
- **Fly.io** — no longer has a true free allowance for new organisations.

---

## Part 3 — Recommended stack

| Piece | Choice | Cost |
| --- | --- | --- |
| Database | Supabase free | ₹0 |
| Web service | Render free | ₹0 |
| Embeddings | `fastembed`, in-process | ₹0 |
| Keep-alive | cron-job.org or GitHub Actions | ₹0 |
| **Total** | | **₹0/month** |

Upgrade path when it matters: Render Starter ($7/mo) removes spin-down;
Supabase Pro ($25/mo) removes pausing and raises storage to 8 GB.

---

## Part 4 — Keeping it awake (free)

Render sleeps after 15 minutes idle; Supabase pauses after 7 days idle. One
cron fixes both, because `/health` touches the database.

**cron-job.org** (free, no account plumbing): new cron job → URL
`https://your-app.onrender.com/health` → every 10 minutes.

**Or GitHub Actions**, committed as `.github/workflows/keepalive.yml`:

```yaml
name: keepalive
on:
  schedule:
    - cron: "*/10 * * * *"
  workflow_dispatch:
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -sf "${{ secrets.HIVE_MIND_URL }}/health" || exit 1
```

Add `HIVE_MIND_URL` under Settings → Secrets → Actions. Pinging every 10
minutes uses about 720 of Render's 750 monthly instance-hours, which fits — but
it is the whole budget, so run only this one service on the account.

---

## Part 5 — Connecting the agents

Both assistants point at the same URL. Full instructions, the GPT instruction
block to paste, and a two-step test that proves they share one memory are in
[CONNECT.md](CONNECT.md). In brief:

**Claude** — the server speaks MCP itself at `/mcp`, so there is nothing to
install:

```bash
claude mcp add --transport http hive-mind https://your-app.onrender.com/mcp \
  --header "X-API-Key: your-key"
```

**ChatGPT** — custom GPT → Action → import
`https://your-app.onrender.com/openapi.json` → auth *API Key*, custom header
`X-API-Key`.

Set `PUBLIC_BASE_URL` to the deployment's own URL first. ChatGPT rejects a
schema without a `servers[]` entry, and the MCP endpoint answers an
unrecognised `Host` header with 421 — that variable is what resolves both.

## Part 6 — Operating it

**Secrets.** `API_KEYS` is the only thing between the internet and your
memories. Generate with `openssl rand -hex 32`, issue a different key per agent
(`API_KEYS=claude-key,chatgpt-key`) so you can revoke one without touching the
other. Never commit `.env`.

**Index tuning.** The default ANN index is hnsw, which needs no tuning. If you
switch to `VECTOR_INDEX_TYPE=ivfflat`, rebuild it once real data exists —
an ivfflat index built on an empty table silently returns fewer rows than
requested:

```sql
REINDEX INDEX memory_embedding_ivfflat_idx;
```

Under ~100 000 memories, hnsw is the better choice on every axis except index
size. See `backend/README.md` §3 for the full explanation.

**Backups.** Supabase free has no automatic backups. A weekly dump is enough:

```bash
pg_dump "$DATABASE_URL" -t memory --data-only -Fc -f hive-mind-$(date +%F).dump
```

**Changing the embedding model** invalidates every stored vector — a 768-dim
model will not even fit the column. If you must: set `EMBEDDING_DIM`, drop and
recreate the table, and re-embed from a dump of the text columns. Decide once,
early.

**Watching it.** `/health` reports database reachability, which backend loaded
and the total memory count. Render and Railway both poll it automatically and
restart on failure. Startup logs one line per subsystem — look for
`MCP endpoint live at /mcp (allowed hosts: …)` to confirm Claude will be able to
connect.
