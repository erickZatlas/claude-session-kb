# Claude Session KB

A small local memory system for Claude Code. Captures every session you have,
generates topical observations from your prompts, surfaces relevant past work
**inside Claude's prompt context automatically**, and exposes the whole corpus
to Claude as MCP tools and to you as a web UI.

Self-contained — one FastAPI app + a SQLite at `~/.claude-kb/data.db` + four
hooks wired into Claude Code's lifecycle. No external memory plugin required.

![Architecture](docs/images/architecture.png)

## What you get

- **Pre-emptive recall** — every prompt you send, a UserPromptSubmit hook runs
  semantic search over your history and injects the top-3 matching sessions as
  `additionalContext`, so Claude sees "you've worked on this before" before it
  starts thinking.
- **Session capture** — SessionStart / UserPromptSubmit / Stop hooks record
  every session, every prompt, and trigger observation generation on completion.
- **Topical observations** — when a session ends, the Stop hook calls DeepSeek
  with a prompt engineered to extract **domain-specific tags** (`OperaPostCharge`,
  `AWAITING_CHECKIN`, `bem-stuck-payments`, file names, identifiers) — never
  generic filler like `how-it-works` / `what-changed`.
- **MCP tools** — Claude can search its own history with `search_my_sessions`
  and pull a specific session's observations with `get_session`. One call,
  no curl.
- **Web UI** — a search-first card overview + drill-down force graph for
  browsing the corpus by hand at `/kb.html`.

Current corpus: 105 sessions / 4,800 observations / 105 LLM-generated
summaries / 103 distinct kebab labels.

## Run it

```bash
pip install -r requirements.txt      # fastapi, uvicorn, numpy, transformers, torch, mcp, openai
python app.py                        # serves http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000> for the in-app teaching guide, or
<http://127.0.0.1:8000/kb.html> for the KB browser.

First start embeds every observation with `all-MiniLM-L6-v2` (a few minutes on
CPU, ~260s for 4,800 records). Cached to `.cache/`; subsequent starts only
embed deltas. **Keyword search works immediately**; semantic recall switches on
once indexing finishes.

## How it works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Claude Code session                            │
│                                                                             │
│  SessionStart ──► capture.py ──► POST /api/capture/start                    │
│                                                                             │
│  UserPromptSubmit ─┬─► recall.py  ──► GET  /api/recall  ──► additionalContext│
│                    └─► capture.py ──► POST /api/capture/prompt              │
│                                                                             │
│  Stop ─┬─► capture.py ──► POST /api/capture/end                             │
│        └─► capture.py ──► POST /api/observe/{sid}  (DeepSeek + tags)        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                          FastAPI app on localhost:8000
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
       ~/.claude-kb/data.db    .cache/vectors.npy +ids.json    .cache/llm.json
       (sessions + prompts            (MiniLM matrix              (DeepSeek
        + observations)                 over our obs)              response cache)
```

- **Hooks fail silently.** Network blip, server down, missing DB — they swallow
  the error and exit 0. Claude Code is never blocked.
- **No rebuild step.** `store.py` watches `data.db`'s `max(created_at)` and
  reloads only when it grows. Each capture or observe write triggers a reload
  on the next read.
- **Cache key includes the system prompt.** Tuning `OBSERVATIONS_SYSTEM` or
  `LABEL_SYSTEM` automatically invalidates the right cache entries; no manual
  cache nukes.

## MCP tools

Registered in `~/.claude.json` under `mcpServers.session-kb`. Two tools:

```
search_my_sessions(query, limit=5, project?, min_score=0.45)
    → semantic recall over your history. Same backend as the
      UserPromptSubmit hook, but the threshold defaults higher (0.45)
      because when Claude *chooses* to search it wants confident hits.

get_session(session_id)
    → fetch one session's label / summary / observations + topical tags.
      Pair with search_my_sessions to drill into a candidate.
```

Server file: `mcp_server.py`. Thin stdio wrapper — all heavy lifting stays in
the FastAPI app. If the app isn't running the tools return a clear "kb server
unreachable" message with the start command.

## Hooks

Registered in `~/.claude/settings.json`:

| Event | Script | Endpoint hit |
|---|---|---|
| `SessionStart` | `hooks/capture.py` | `POST /api/capture/start` |
| `UserPromptSubmit` | `hooks/recall.py` | `GET /api/recall` → injects `additionalContext` |
| `UserPromptSubmit` | `hooks/capture.py` | `POST /api/capture/prompt` |
| `Stop` | `hooks/capture.py` | `POST /api/capture/end` + `POST /api/observe/{sid}` |

Recall hook tunables in `hooks/recall.py`:

- `MIN_SCORE = 0.32` — drop hits below this cosine; raise to 0.45 to silence weak matches
- `LIMIT = 3` — sessions injected per prompt
- `MIN_PROMPT_LEN = 16` — skip very short prompts (no signal)
- `TIMEOUT_S = 2.0` — hook never blocks Claude for more than 2s

## Web UI

`/kb.html` — search-first overview:

- **Default (no query):** card column of your sessions sorted by *recent* or
  *activity*. Each card: kebab label, project + obs count + relative time,
  LLM summary, copy-resume button.
- **Search active:** ranked record cards; click `from session: <kebab>` to
  drill into the source.
- **Drill-down:** force graph of one session's observations + the records
  list + detail pane. **← all sessions** returns to the cards.

`/` — full 13-section teaching guide with every interaction documented.

## API

| Endpoint | Purpose |
|---|---|
| `GET  /api/meta` | counts, projects, freshness, indexing + enrichment progress |
| `GET  /api/search?q=&project=&kind=&mode=keyword\|semantic&session=&limit=` | ranked records |
| `GET  /api/record/{id}` | one record + its session |
| `GET  /api/sessions?project=` | sessions w/ kebab labels + LLM summaries |
| `GET  /api/graph?project=` | session-overview topology (links via shared tags) |
| `GET  /api/recall?q=&limit=&min_score=&project=&exclude=` | semantic recall |
| `GET  /api/observe/{sid}` | read obs + label + summary for a session |
| `POST /api/observe/{sid}[?sync=true]` | (re-)generate obs (background by default) |
| `POST /api/observe/backfill` | generate for every session missing obs |
| `POST /api/labels/backfill[?force=true&use_llm=true]` | re-derive kebab labels |
| `POST /api/capture/{start,prompt,end}` | hooks talk to these |
| `GET  /api/capture/stats` | capture-store health (sessions / prompts / active) |
| `POST /api/legacy/import` + `import-observations` | one-shot bootstrap endpoints used during the historical migration; harmless no-ops if the legacy DB isn't present |
| `GET  /api/stream` | SSE; `refresh` when our DB grows, else `ping` with progress |

## Files

| File | Role |
|------|------|
| `app.py` | FastAPI: REST + SSE + serves the frontend (no-cache, version-busted) |
| `store.py` | Live read of `~/.claude-kb/data.db`, keyword search, freshness reload |
| `store_capture.py` | Capture store: sessions/prompts/observations tables + helpers |
| `observe.py` | DeepSeek topical-tag observation generation; client-side tag extractor for Phase D imports |
| `embeddings.py` | `all-MiniLM-L6-v2` embeddings, cosine search, incremental disk cache |
| `llm.py` | DeepSeek client for kebab labels + 1–2 sentence summaries; hash-keyed disk cache |
| `mcp_server.py` | Stdio MCP server exposing `search_my_sessions` + `get_session` |
| `hooks/recall.py` | UserPromptSubmit hook — pre-emptive context injection |
| `hooks/capture.py` | SessionStart / UserPromptSubmit / Stop hook — captures + triggers observe |
| `static/index.html` | Explainer + 13-section user guide |
| `static/kb.html` + `kb.js` + `theme.css` | The KB browser — cards + D3 force-graph + EventSource client |
| `.cache/` | Embedding cache + LLM response cache (git-ignored) |

## Optional LLM (DeepSeek)

Set `DEEPSEEK_API_KEY` in your environment to enable:

- **Topical observations** (Stop hook) — 3 to 7 per session, with domain-specific
  tags. The system prompt explicitly bans generic tokens; a client-side filter
  drops anything that sneaks through.
- **Kebab labels** per session — 1–3 lowercase tokens joined by hyphens
  (`stuck-charges`, `oxi-outage-probe`, `bem-stripe-cdp-fix`). Real English
  words and known acronyms only; typos corrected.
- **1–2 sentence summaries** per session.

All three share `llm.py`'s disk cache at `.cache/llm.json`, keyed by
`sha256(model + kind + system_prompt + payload)`. First run on a 100-session
corpus cost ~$0.05 on DeepSeek; every subsequent restart is free.

Without `DEEPSEEK_API_KEY` the system degrades cleanly to a TF-IDF + regex tag
extractor; no errors, just less polish.

**Privacy:** enabling DeepSeek sends session content (your prompts + observation
titles) to their API under their TOS. Sessions can contain confidential project
data — opt in only if that's acceptable. The base capture + recall stack runs
fully local without the key.

## Notes

- Everything writes to `~/.claude-kb/` only. No remote DB, no telemetry, no
  cloud sync.
- D3 loads from a CDN for the web UI (first open needs internet); the
  embedding model is server-side, fetched once to `~/.cache/huggingface`.
- UI screenshots are intentionally omitted from this repo — they would
  otherwise leak real session content from confidential work.
- The full teaching guide (every feature, every interaction, every workflow)
  lives at `/` when the server is running — <http://127.0.0.1:8000>.
