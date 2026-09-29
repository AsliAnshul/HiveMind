# Connecting ChatGPT and Claude

After deployment you have one URL and one API key. This page turns those into a
memory both assistants share: whatever one writes, the other can find.

Throughout, substitute your own:

```
URL = https://your-app.onrender.com
KEY = the value of API_KEYS from backend/.env
```

Two doors into the same database:

| Assistant | Door | What it needs |
| --- | --- | --- |
| Claude | `URL/mcp` — MCP over HTTP | the URL and the key |
| ChatGPT | `URL/openapi.json` — GPT Action | the URL and the key |

Nothing to install on your machine for either one.

---

## Before you start

`PUBLIC_BASE_URL` must be set on the deployment to its own URL, and then
redeployed. Two things depend on it:

- ChatGPT refuses to import an OpenAPI schema with no `servers[]` entry.
- The MCP endpoint rejects unrecognised `Host` headers with a 421, and this is
  what tells it which hostname is legitimately yours.

Check both are live before wiring anything up:

```bash
curl -s "$URL/health"
curl -s -o /dev/null -w "%{http_code}\n" -X POST "$URL/mcp" \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'
```

`/health` should report `"status": "ok"`, and the MCP call should print `200`.
A `421` means `PUBLIC_BASE_URL` does not match the URL you are calling.

---

## Claude

### Claude Code (one command)

```bash
claude mcp add --transport http hive-mind "$URL/mcp" \
  --header "X-API-Key: $KEY"
```

Then `/mcp` inside Claude Code lists it, and the tools `remember`, `recall`,
`project_context`, `list_projects` and `forget` become available.

### Claude Desktop

Settings → Developer → Edit Config, then:

```json
{
  "mcpServers": {
    "hive-mind": {
      "type": "http",
      "url": "https://your-app.onrender.com/mcp",
      "headers": { "X-API-Key": "your-key" }
    }
  }
}
```

Restart Claude Desktop. If your version does not accept `headers`, use the
stdio bridge instead:

```json
{
  "mcpServers": {
    "hive-mind": {
      "command": "python",
      "args": ["/absolute/path/to/hive-mind/integrations/mcp_server.py"],
      "env": {
        "HIVE_MIND_URL": "https://your-app.onrender.com",
        "HIVE_MIND_API_KEY": "your-key",
        "HIVE_MIND_AGENT": "claude"
      }
    }
  }
}
```

That bridge needs `pip install -r integrations/requirements.txt` and speaks to
the same REST API, so both routes end up in the same table.

### claude.ai (Settings → Connectors)

Add a custom connector pointing at `https://your-app.onrender.com/mcp`. If the
version you have offers no place to put a header, use Claude Desktop or Claude
Code instead — the API key is not optional, and a connector that cannot send
one cannot authenticate.

---

## ChatGPT

Requires ChatGPT Plus (custom GPTs are a paid feature).

1. **Create the GPT** — ChatGPT → Explore GPTs → **Create** → *Configure*.
2. **Add the action** — *Create new action* → **Import from URL** →

   ```
   https://your-app.onrender.com/openapi-3.0.json
   ```

   Note the `-3.0`. FastAPI's `/openapi.json` is OpenAPI **3.1**, and ChatGPT's
   importer reads **3.0** — it rejects 3.1's `anyOf: [X, {"type": "null"}]`
   spelling of an optional field. `/openapi-3.0.json` is the same API
   translated, and both documents are validated against the official schemas by
   the test suite.

   Seven operations appear: `write_memory`, `search_memory`, `project_context`,
   `list_memories`, `get_memory`, `delete_memory`, `list_projects`.
3. **Authentication** → *API Key* → Auth Type **Custom** → Header name
   `X-API-Key` → paste the key.

   The key must *not* appear as a parameter on any operation — it is hidden from
   the schema on purpose, so the Action's auth configuration supplies it rather
   than the model trying to guess it.
4. **Instructions** — paste the block below into the GPT's *Instructions* field.
   Without it the GPT has the tools but no habit of using them, which is the
   usual reason a setup like this quietly stops being useful.

```text
You have a shared long-term memory that another AI assistant (Claude) also reads
and writes. It is not your private scratchpad — write for a stranger who lacks
all of your context.

Before answering anything about an ongoing project, call project_context with
that project name. If you are unsure whether something is already known, call
search_memory before asking the user to repeat themselves.

After any decision, plan, architectural choice or conclusion worth keeping, call
write_memory with:
  project  a short lowercase slug, reused consistently for the same work
  type     plan | summary | notes | architecture
  title    one specific line, not "Notes"
  content  the reasoning, the alternatives rejected, and why — not just the outcome
  source   "chatgpt"

Rules:
- Re-writing identical content is harmless; it refreshes rather than duplicates.
- Never invent a project name. Call list_projects if you are not sure which exists.
- Never call delete_memory without explicit confirmation from the user.
- When you use something from memory, say so, including who wrote it and when.
```

---

## Verifying they really share one brain

Ask ChatGPT (with the GPT open):

> Remember in project `shared-test`: we chose Render over Railway because
> Railway stops being free after the trial credit.

Then ask me, in a new conversation:

> Search hive mind memory: why did we pick our hosting provider?

You should get ChatGPT's note back, with `"source": "chatgpt"` on it. Run it in
reverse to confirm the other direction. Then delete the test:

```bash
curl -s "$URL/memory?project=shared-test" -H "X-API-Key: $KEY"
curl -s -X DELETE "$URL/memory/<id>" -H "X-API-Key: $KEY"
```

---

## Using it well

**One project slug per body of work**, lower-case, reused exactly. `search_memory`
without a `project` searches everything, which is usually what you want when you
cannot remember where something went.

**Write the why.** A memory saying "chose Render" is nearly useless six weeks
later. "Chose Render over Railway because Railway's free credit expires and this
must cost nothing" survives.

**Separate keys per assistant** — `API_KEYS=claude-key,chatgpt-key` — so you can
revoke one without breaking the other. Both still see all the same rows; the key
is for authentication, not isolation.

**The `source` field** records which assistant wrote each entry, so when the two
disagree you can see who said what.

---

## When something breaks

| Symptom | Cause |
| --- | --- |
| `401 Missing X-API-Key` | Header absent, or the client dropped it on a redirect. Use the exact `URL/mcp`. |
| `403 Invalid API key` | Key does not match `API_KEYS` on the server. Check for a trailing space. |
| `421 Invalid Host header` | `PUBLIC_BASE_URL` does not match the URL being called. Fix it and redeploy. |
| ChatGPT: "could not import schema" | Either `PUBLIC_BASE_URL` is unset, so the document has no `servers[]`, or you imported `/openapi.json` (3.1) instead of `/openapi-3.0.json`. |
| ChatGPT asks you for an "x-api-key" argument | You are on an older deployment where the header was still in the schema. Redeploy. |
| First call after idle takes ~50 s | Render free tier cold start. See the keep-alive cron in `deploy.md`. |
| `503 Database unavailable` | Supabase project paused after 7 days idle. Open the dashboard and restore; the service recovers by itself, no redeploy. |
| `FATAL: (ENOTFOUND) tenant/user ... not found` | Same thing seen from the pooler. A paused project loses its DNS, so `db.<ref>.supabase.co` stops resolving while the shared pooler host still does. Restore the project. |
| `/health` says `"status": "degraded"` | The service is up but the database is not answering. `schema_ready` and `database` in the same response tell you which half is broken. |
| Works locally, every call times out once deployed | `DATABASE_URL` uses Supabase's direct host (`db.<ref>.supabase.co`), which is IPv6-only. Switch to the session pooler URI. |
| Claude lists no tools | The MCP server failed to start. Check the deployment logs for `MCP endpoint live at /mcp`. |
