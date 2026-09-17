# Hive Mind Memory Server

A standalone service: shared long-term memory for AI assistants. Claude and
ChatGPT write structured knowledge into one Postgres table and retrieve it by
meaning.

**This project is independent.** It shares no code, data, deployment or roadmap
with anything else in the parent directory, and instructions from a parent
`AGENTS.md` do not apply here. Do not import from sibling projects, and do not
treat their conventions as precedent.

- `backend/` — the FastAPI service. Everything lives here.
- `integrations/` — a stdio MCP bridge, for clients that cannot use the remote
  `/mcp` endpoint.

Ground rules:

- Routes → services → db. Routes never touch embeddings or SQL; services never
  touch HTTP.
- Two embedding backends produce identical vectors. Never change
  `EMBEDDING_MODEL` or `EMBEDDING_DIM` casually: every stored vector becomes
  meaningless and the column will not even fit a different width.
- ANN search fails *silently* — a misconfigured index returns fewer rows rather
  than an error. Any change to indexing or filtering needs a test that asserts
  an expected row is actually present.
- The API is consumed by two different assistants. Response shapes are a
  contract; changing a field name breaks a GPT Action schema that was imported
  weeks ago.
- Tests need a real pgvector database. `backend/scripts/local_pg.sh start`
  provides one without root.
