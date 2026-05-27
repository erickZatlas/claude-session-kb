# Claude Knowledge Base

A small local web app that makes the knowledge in your [claude-mem](https://github.com/thedotmack/claude-mem)
store browsable and searchable — a D3 graph of how sessions, observations, files, and concepts
connect, plus keyword **and** semantic search. A FastAPI backend reads claude-mem live (read-only)
and stays current as you work.

- **Explainer** (`/`) — the per-task session workflow (one slug per task; `ctask` / `cont` /
  `cresume` / `cfind`) and how it ties into the local memory ecosystem.
- **Knowledge base** (`/kb.html`) — graph + search over your observations and session summaries.

## Run it

```bash
pip install -r requirements.txt      # fastapi, uvicorn, numpy, transformers, torch
python app.py                        # serves http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000> for the explainer, or <http://127.0.0.1:8000/kb.html> for the KB.

On first start the server reads claude-mem and builds the semantic index in the background
(embeds every record once with `all-MiniLM-L6-v2` — a few minutes on CPU). **Keyword search works
immediately**; semantic search switches on once indexing finishes. The index is cached to
`.cache/`, so subsequent starts are fast and only new records get embedded.

## How it works

```
                      reads (read-only, live)
  ~/.claude-mem/claude-mem.db  ───────────────▶  store.py   (records, keyword search, freshness)
                                                 embeddings.py (MiniLM vectors, cosine, .cache)
                                                      │
  browser  ◀── REST /api/search · /api/record ───────┤  app.py (FastAPI)
           ◀── SSE  /api/stream (live refresh) ───────┘
           static/: kb.html · kb.js (D3 graph) · theme.css · index.html
```

- **No rebuild step.** The backend re-reads claude-mem whenever it grows (`ensure_fresh`), so the
  API is always current. Semantic queries are embedded **server-side** — no model in the browser.
- **SSE for freshness only.** Search is plain request/response; `/api/stream` pushes a `refresh`
  event when claude-mem gains new entries, and the page re-runs the current view automatically.
- **The graph is a session map, not a result dump.** By default it shows your sessions as nodes
  (sized by activity), linked when they share files — an overview of your work. Click a session to
  drill into its observations; a search scopes the map to the sessions that contain the hits.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | counts, projects, freshness, indexing status |
| `GET /api/search?q=&project=&kind=&mode=keyword\|semantic&session=&limit=` | ranked records |
| `GET /api/record/{id}` | one record + its session |
| `GET /api/sessions?project=` | sessions |
| `GET /api/graph?project=` | session-overview nodes + shared-file links |
| `GET /api/stream` | SSE; `refresh` when claude-mem grows, else `ping` |

## Files

| File | Role |
|------|------|
| `app.py` | FastAPI: REST + SSE + serves the frontend. |
| `store.py` | Live read of claude-mem, keyword search, freshness reload. |
| `embeddings.py` | `all-MiniLM-L6-v2` embeddings, cosine search, incremental disk cache. |
| `static/index.html` | The explainer (self-contained, D3 diagrams). |
| `static/kb.html` + `static/kb.js` | The KB UI — fetches the API, draws the graph, listens to SSE. |
| `static/theme.css` | Shared light theme (IBM Plex). |
| `.cache/` | Generated embedding cache (git-ignored). |

## Session workflow helpers (`shell/claude.sh`)

The explainer documents a companion habit: **one slug per task**, shared across the Claude Code
session name, the terminal tab title, and an optional git worktree. `shell/claude.sh` provides the
helpers — source it from your shell rc:

| Command | Does |
|---|---|
| `ctask <slug> [-w]` | start a new named session (`claude -n <slug>`), set the tab title, optionally make a worktree |
| `cont` | continue the most recent session in this directory |
| `cresume <slug>` | reopen a session by name |
| `cfind <phrase>` | find a past session by what was said in it |

## Notes

- Everything is local and read-only against claude-mem; nothing is written back to it.
- D3 loads from a CDN (first open needs internet); the embedding model is server-side and is
  fetched once to `~/.cache/huggingface`.
