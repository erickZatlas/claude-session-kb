# Claude Knowledge Base

A small local web app that makes the knowledge in your [claude-mem](https://github.com/thedotmack/claude-mem)
store browsable and searchable — a **search-first overview of your sessions as cards**, plus
**keyword and semantic search** over every observation and summary claude-mem has captured.
A FastAPI backend reads `~/.claude-mem` live (read-only) and stays current as you work.

![Architecture](docs/images/architecture.png)

- **Explainer + complete guide** (`/`) — the per-task session workflow (`ctask` / `cont` /
  `cresume` / `cfind`) and how to use every feature of the KB.
- **Knowledge base** (`/kb.html`) — session cards + search + drill-down force graph.

## Run it

```bash
pip install -r requirements.txt      # fastapi, uvicorn, numpy, transformers, torch
python app.py                        # serves http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000> for the guide, or <http://127.0.0.1:8000/kb.html> for the KB.

On first start the server reads claude-mem and builds the semantic index in the background
(embeds every record once with `all-MiniLM-L6-v2` — a few minutes on CPU). **Keyword search
works immediately**; semantic search switches on once indexing finishes. The index is cached
to `.cache/`, so subsequent starts are fast and only new records get embedded.

## What it looks like

The UI has three pieces, all reachable from <http://127.0.0.1:8000/kb.html>:

- **Overview** — a scrollable column of session cards. Each card has a kebab-case topic
  title (`stuck-charges`, `opera-cloud`, `bem-1password`), a project + observation-count +
  relative-time strip, the LLM's 1–2 sentence summary, and a one-click **Copy resume**
  button. Sort by *recent* or *activity* via the chip in the toolbar.
- **Search** — typing a query swaps the cards from sessions to ranked **record cards** —
  one per hit, with a type badge, snippet, project/date/files, and a clickable
  `from session: <kebab>` link that drills into the source session. Toggle *semantic* in
  the toolbar to rank by meaning instead of literal keyword.
- **Drill-down** — clicking any session card opens a force graph of that session's
  observations and summaries (colour-coded by type), a records list, and a detail pane.
  Click **← all sessions** to return to the cards.

The in-app guide at <http://127.0.0.1:8000> walks through every interaction in detail.
UI screenshots are intentionally omitted from this repo because they would otherwise
show real session content from a private claude-mem.

## How it works

- **No rebuild step.** The backend re-reads claude-mem whenever it grows (`ensure_fresh`),
  so the API is always current. Semantic queries are embedded **server-side** — no model in
  the browser.
- **SSE for freshness only.** Search is plain request/response; `/api/stream` pushes a
  `refresh` event when claude-mem gains new entries, and the page re-runs the current view
  automatically.
- **Two views, one app.** Cards (overview) = readable text where text belongs.
  Force graph (drill-down) = one session's observations expanded around its anchor. They
  share search / filter state and live freshness.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | counts, projects, freshness, indexing + enrichment status |
| `GET /api/search?q=&project=&kind=&mode=keyword\|semantic&session=&limit=` | ranked records |
| `GET /api/record/{id}` | one record + its session |
| `GET /api/sessions?project=` | sessions (with kebab labels + LLM summaries when enriched) |
| `GET /api/graph?project=` | session-overview nodes + shared-file links (kept for a future map toggle; currently unused) |
| `GET /api/stream` | SSE; `refresh` when claude-mem grows, else `ping` with progress |

## Files

| File | Role |
|------|------|
| `app.py` | FastAPI: REST + SSE + serves the frontend (no-cache + version-busted assets) |
| `store.py` | Live read of claude-mem, keyword search, freshness reload, TF-IDF labels |
| `embeddings.py` | `all-MiniLM-L6-v2` embeddings, cosine search, incremental disk cache |
| `llm.py` | Optional DeepSeek client for kebab labels + 1–2 sentence session summaries |
| `static/index.html` | Explainer + complete user guide (13 sections, D3 diagrams) |
| `static/kb.html` + `static/kb.js` | The KB UI — card overview + D3 force-graph drill-down + EventSource client |
| `static/theme.css` | Shared dark "observatory" theme (Inter + JetBrains Mono) |
| `shell/claude.sh` | Per-task session helpers — see below |
| `.cache/` | Generated embedding cache + LLM cache (git-ignored) |

## Session workflow helpers (`shell/claude.sh`)

The guide documents a companion habit: **one slug per task**, shared across the Claude Code
session name, the terminal tab title, and an optional git worktree. Source the file from
your shell rc:

| Command | Does |
|---|---|
| `ctask <slug> [-w]` | start a new named session (`claude -n <slug>`), set the tab title, optionally make a worktree |
| `cont` | continue the most recent session in this directory |
| `cresume <slug>` | reopen a session by name |
| `cfind <phrase>` | find a past session by what was said in it |

## Optional LLM clarification (DeepSeek)

Set `DEEPSEEK_API_KEY` in your environment and the server will, in a background pass, ask
the LLM for:

- **A kebab-case topic label per session** — 1–3 lowercase tokens joined by hyphens
  (`stuck-charges`, `opera-cloud`, `ihg-awaiting-checkin`) — replacing the TF-IDF heuristic.
- **A 1–2 sentence summary per session** — shown in the detail panel when you click a
  session anchor.

Results cache to `.cache/llm.json`, keyed by content hash + the system prompt itself; any
prompt tweak invalidates only the affected entries. Cost is pennies on the first run, free
on every subsequent restart.

**Privacy:** session content (your prompts + claude-mem observation titles) goes to
DeepSeek's API under their TOS. If that's not acceptable for your data, leave
`DEEPSEEK_API_KEY` unset — the page degrades cleanly to the TF-IDF labels and an empty
session-detail card.

## Notes

- Everything is local and read-only against claude-mem; nothing is written back to it.
- D3 loads from a CDN (first open needs internet); the embedding model is server-side and
  is fetched once to `~/.cache/huggingface`.
- The full user guide (every feature, every interaction, every workflow) lives at `/` when
  the server is running — open <http://127.0.0.1:8000>.
