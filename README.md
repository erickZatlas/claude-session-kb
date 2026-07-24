# Claude Session KB

A small local memory system for Claude Code. Captures every session you have —
prompts AND every tool call — generates topical observations from both,
surfaces relevant past work **inside Claude's prompt context automatically**,
and exposes the whole corpus to Claude as MCP tools and to you as a web UI.

Self-contained — one FastAPI app + a SQLite at `~/.claude-kb/data.db` + five
hooks wired into Claude Code's lifecycle. No external memory plugin required.

![Architecture](docs/images/architecture.png)

## What you get

- **Pre-emptive recall** — every prompt you send, a `UserPromptSubmit` hook
  runs semantic search over your history and injects the top-3 matching
  sessions as `additionalContext`, so Claude sees "you've worked on this
  before" before it starts thinking. Ranks by `cosine · time-decay +
  file-overlap-boost + same-project-boost` — six-month-old work doesn't outrank
  last week's, sessions that touched the file you're asking about climb, and
  work in your current project is gently preferred (a boost, not a filter).
- **Session + tool-call capture** — `SessionStart` / `UserPromptSubmit` /
  **`PostToolUse`** / `Stop` hooks record everything: each session, each
  prompt, and every tool call (Read / Edit / Write / Bash / Grep / Glob /
  MCP) with its touched files. PostToolUse capture is what
  [`claude-mem`](https://github.com/thedotmack/claude-mem) gives you; the
  rest of this list is what we add on top.
- **Background summarizer** — an inline worker thread drains the tool-call
  queue every 60s, hands each session's batch to DeepSeek, and writes
  observations as they accumulate. No separate daemon — the FastAPI process
  hosts both the API and the worker.
- **Topical observations** — DeepSeek prompted to emit **domain-specific
  tags** (`HttpClient`, `RetryPolicy`, `cache-invalidation`, file
  names, identifiers) — never generic filler like `how-it-works` /
  `what-changed`. Past-tense titles naming the concrete thing done.
- **Cross-session "lessons"** — manual rollup that reads every observation
  in the last N days and distills 5–15 durable patterns (project
  conventions, architectural facts, recurring bug pedigrees) with
  back-links to the sessions that support each one.
- **MCP tools** — Claude can search its own history with
  `search_my_sessions`, fetch one session's observations with
  `get_session`, find sessions that touched a specific file with
  `find_sessions_by_file`, and list distilled lessons with `list_lessons`.
- **Web UI** — a search-first card overview + drill-down force graph for
  browsing the corpus by hand at `/kb.html`.

Current corpus: 106 sessions · 4,805 observations · 10 lessons · 106
LLM-generated summaries · ~3500 indexed tool calls.

## Run it

```bash
pip install -r requirements.txt      # fastapi, uvicorn, numpy, transformers, torch, mcp, openai
python app.py                        # serves http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000> for the in-app teaching guide, or
<http://127.0.0.1:8000/kb.html> for the KB browser.

First start embeds every observation with `all-MiniLM-L6-v2` (a few minutes
on CPU, ~260s for 4,800 records). Cached to `.cache/`; subsequent starts
only embed deltas. **Keyword search works immediately**; semantic recall
switches on once indexing finishes.

### Run as a service (recommended)

`python app.py` under `nohup` dies on reboot and silently disables the recall
hook. A systemd **user** unit keeps it alive:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/claude-session-kb.service ~/.config/systemd/user/
# optional: echo 'DEEPSEEK_API_KEY=sk-...' > ~/.config/claude-kb.env
systemctl --user daemon-reload
systemctl --user enable --now claude-session-kb
loginctl enable-linger "$USER"        # survive logout
curl -s http://127.0.0.1:8000/api/health
```

### Tests

```bash
pip install -r requirements.txt       # includes pytest
pytest -q                             # pure-logic unit tests (no model/network)
```

Covers recall ranking (decay, file/project boost, knowledge gap-trim), lessons
merge, auto-memory frontmatter parsing + project resolution, tag extraction, and
the store's record projection. Write-path tests use throwaway temp DBs.

## How it works

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            Claude Code session                             │
│                                                                            │
│  SessionStart      ─► capture.py ─► POST /api/capture/start                │
│  UserPromptSubmit ─┬─► recall.py  ─► GET  /api/recall ─► additionalContext │
│                    └─► capture.py ─► POST /api/capture/prompt              │
│  PostToolUse       ─► capture.py ─► POST /api/capture/tool                 │
│                                       (every Read/Edit/Bash/Grep/MCP)      │
│  Stop             ─┬─► capture.py ─► POST /api/capture/end                 │
│                    └─► capture.py ─► POST /api/observe/{sid}               │
└────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                          FastAPI app on localhost:8000
              (REST + SSE + serves frontend + MCP backend + worker thread)
                                       │
              ┌────────────────────────┼───────────────────────┐
              ▼                        ▼                       ▼
     ~/.claude-kb/data.db    .cache/vectors.npy +ids   .cache/llm.json
     ─────────────────────   (MiniLM 384-d matrix      (DeepSeek response
     sessions   (id, project,  over every observation)  cache, sha256 keyed)
                cwd, label,
                summary, …)
     prompts    (text, ts)
     observations (title, text, tags JSON, type)
     tool_calls   ← Phase E queue: every PostToolUse, status=pending|processed
     session_files (path, kind, count) ← file-aware recall + 📁 chip
     lessons      (title, text, tags, source_session_ids)
     memory_facts (name, mem_type, text, tags) ← auto-memory synced from disk
```

- **Durable knowledge in recall.** `/api/recall` returns a `lessons` array
  *alongside* the matching sessions: the top distilled **lessons** plus the
  hand-authored **auto-memory** facts (see below). The `UserPromptSubmit` hook
  renders them in a `### Relevant lessons & memory` block **above** the sessions,
  so the densest, most durable learnings lead. Knowledge is **not** time-decayed
  (durability is the point) and uses a looser score floor than sessions.
- **Auto-memory ingest.** The hand-authored facts under
  `~/.claude/projects/*/memory/*.md` (one fact per file, frontmatter
  `name`/`description`/`metadata.type`, written by Claude Code's memory tool) are
  mirrored into the `memory_facts` table by `memory_ingest.py` and **projected
  into the same MiniLM corpus** as observations + lessons — so they show up in
  recall, `/api/search`, the MCP tools, and the web UI. Synced on startup and by
  a background **filesystem-watch loop** (`SESSION_KB_MEMORY_WATCH_INTERVAL_S`,
  default 10s) that ingests + re-embeds autonomously; a file is re-embedded only
  when its `mtime` + content hash change, and a deleted file is pruned.
  *Storage decision:* memory facts get their **own table** (not folded
  into `lessons`, which `distill` wipes+rewrites; not `observations`, which are
  session-scoped) but share the read-side record corpus via
  `store._fetch_memory_facts`.
- **Three background workers** run inside the FastAPI process (all honor a
  shutdown event):
  - *Tool-call summarizer* — every 60s, drains the `tool_calls` queue, hands
    each session's batch (≤50 calls) to DeepSeek, writes observations, marks the
    batch processed. 5-min cool-off before a session is eligible so we don't fire
    mid-burst (`SESSION_KB_WORKER_INTERVAL_S`, `_MIN_AGE_S`, `_BATCH`).
  - *Memory filesystem watch* — re-scans the auto-memory dirs every 10s; a change
    triggers an immediate reload + re-embed (`SESSION_KB_MEMORY_WATCH_INTERVAL_S`).
  - *Lessons distiller* — runs the cross-session lessons pass every 6h
    (`SESSION_KB_LESSONS_INTERVAL_S`, `_DAYS`; set interval `0` to disable).
    Skips ticks when `DEEPSEEK_API_KEY` is unset.
- **Lessons merge, not wipe.** Distillation now **merges** by title — refresh
  text/tags, bump `last_seen`, union evidence — so a durable lesson the distiller
  didn't re-derive this run survives. `POST /api/lessons/distill?replace=true`
  forces the old wipe-and-rewrite when you want a clean slate.
- **Hooks fail silently.** Network blip, server down, missing DB — they
  swallow the error and exit 0. Claude Code is never blocked.
- **No rebuild step.** `store.py` watches `data.db`'s `max(created_at)` and
  reloads only when it grows. Each capture or observe write triggers a
  reload on the next read.
- **Self-compacting index.** `Embedder.sync` only appends, and content-hashed
  memory/lesson ids orphan their old vectors on every edit/re-distill. The
  workers auto-drop those orphans once they exceed
  `SESSION_KB_COMPACT_ORPHAN_RATIO` of the live record count (or `POST
  /api/index/compact`), so the matrix stays bounded.
- **Cache key includes the system prompt.** Tuning any of the three system
  prompts (observations / tool-observations / lessons) automatically
  invalidates the right cache entries; no manual cache nukes.

## MCP tools

Registered in `~/.claude.json` under `mcpServers.session-kb`. Five tools:

```
search_my_sessions(query, limit=5, project?, min_score=0.45)
    → semantic recall over your history. Same backend as the UserPromptSubmit
      hook, but the threshold defaults higher (0.45) — when Claude *chooses*
      to search it wants confident hits. Also surfaces the matching distilled
      lessons + auto-memory facts (the `lessons` block) alongside sessions.

get_session(session_id)
    → one session's label + summary + observations + topical tags.
      Pair with search_my_sessions to drill into a candidate.

find_sessions_by_file(path, limit=10)
    → sessions that read or edited a specific file via PostToolUse capture.
      Accepts an absolute path or a bare basename (suffix-matched).
      Use when you're about to touch file X and want prior work on it.

list_lessons(tag?, limit=20)
    → distilled cross-session lessons — durable facts that recur across many
      sessions (project conventions, architectural decisions, bug patterns).
      Use when grounding a task in long-standing project knowledge rather
      than one specific past session.

list_memory_facts(type?, limit=50)
    → the hand-authored auto-memory facts (~/.claude/projects/*/memory/*.md):
      who the user is, feedback/preferences, project constraints, references.
      Optionally filter by type (user|feedback|project|reference).
```

Server file: `mcp_server.py`. Thin stdio wrapper — all heavy lifting stays
in the FastAPI app. If the app isn't running the tools return a clear
"kb server unreachable" message with the start command.

## Hooks

Registered in `~/.claude/settings.json`:

| Event | Script | Endpoint hit |
|---|---|---|
| `SessionStart` | `hooks/capture.py` | `POST /api/capture/start` |
| `UserPromptSubmit` | `hooks/recall.py` | `GET /api/recall` → injects `additionalContext` |
| `UserPromptSubmit` | `hooks/capture.py` | `POST /api/capture/prompt` |
| `PostToolUse` | `hooks/capture.py` | `POST /api/capture/tool` |
| `Stop` | `hooks/capture.py` | `POST /api/capture/end` + `POST /api/observe/{sid}` |

`hooks/capture.py` dispatches on `hook_event_name`; one script, all four
roles.

Recall hook tunables in `hooks/recall.py`:

- `MIN_SCORE = 0.32` — drop session hits below this; raise to 0.45 to silence weak matches
- `LIMIT = 3` — sessions injected per prompt
- `LESSON_LIMIT = 4` — lessons/memory facts injected per prompt
- `LESSON_MIN_SCORE = 0.30` — score floor for the lessons/memory block
- `MIN_PROMPT_LEN = 16` — skip very short prompts (no signal)
- `TIMEOUT_S = 2.0` — hook never blocks Claude for more than 2s

Worker + recall tunables in env (read at request time — no restart needed):

| Env var | Default | What it does |
|---|---|---|
| `SESSION_KB_WORKER_INTERVAL_S` | 60 | Background tool-call summarizer wake interval |
| `SESSION_KB_WORKER_MIN_AGE_S`  | 300 | Wait this many sec after the last tool call before summarizing |
| `SESSION_KB_WORKER_BATCH`      | 50  | Tool calls per DeepSeek call |
| `SESSION_KB_HALFLIFE_DAYS`     | 30  | Recall time-decay half-life. Set 99999 to disable |
| `SESSION_KB_FILE_BOOST_PER_PATH` | 0.10 | Score boost per file the query mentions that this session touched |
| `SESSION_KB_FILE_BOOST_CAP`    | 0.25 | Max additive file-overlap boost |
| `SESSION_KB_KNOWLEDGE_MIN_SCORE` | 0.28 | Score floor for lessons/memory in the recall `lessons` block |
| `SESSION_KB_KNOWLEDGE_LIMIT`   | 4   | Max lessons/memory facts returned by `/api/recall` |
| `SESSION_KB_MEMORY_WATCH_INTERVAL_S` | 10 | Auto-memory filesystem-watch loop interval |
| `SESSION_KB_LESSONS_INTERVAL_S` | 21600 | Scheduled lessons-distill interval (6h); `0` disables the worker |
| `SESSION_KB_LESSONS_DAYS`      | 30  | Rolling window (days) the scheduled distiller scans |
| `SESSION_KB_PROJECT_BOOST`     | 0.05 | Additive recall boost for sessions/memory whose project matches the caller's `boost_project` |
| `SESSION_KB_KNOWLEDGE_GAP`     | 0.10 | Tail-trim: drop lessons/memory more than this far below the top knowledge score |
| `SESSION_KB_COMPACT_ORPHAN_RATIO` | 1.25 | Auto-compact the embedding matrix once orphan vectors exceed this ratio of live records |

## Web UI

`/kb.html` — search-first overview:

- **Default (no query):** card column of your sessions sorted by *recent*
  or *activity*. Each card: kebab label, project chip, worktree chip
  (`wt: <name>` when the session was started inside `<repo>/.claude/worktrees/`),
  obs count, 📁 N (files touched), relative time, LLM summary, copy-resume
  button.
- **Search active:** ranked record cards; click `from session: <kebab>` to
  drill into the source. Kind chips (`all` / `observations` / `lessons` /
  `memory` / `summaries`) scope results — lessons + auto-memory records are
  searchable here too (they have no source session, so they show without the
  `from session:` link).
- **Drill-down:** force graph of one session's observations + the records
  list + detail pane. A teal **session banner** at the top of the
  drill-down view always shows which session you're in (kebab label +
  project + worktree + obs count + summary). **← all sessions** returns to
  the cards.

`/` — full teaching guide.

## API

| Endpoint | Purpose |
|---|---|
| `GET  /api/meta` | counts, projects, freshness, indexing + enrichment progress |
| `GET  /api/health` | cheap liveness probe (no reload): indexed/records/loadedAt + worker liveness |
| `GET  /api/search?q=&project=&kind=&mode=keyword\|semantic&session=&limit=` | ranked records |
| `GET  /api/record/{id}` | one record + its session |
| `GET  /api/sessions?project=` | sessions w/ kebab labels + LLM summaries + filesCount + worktree |
| `DELETE /api/sessions/{sid}[?purge=true&force=true]` | delete a session: trash its transcript (reversible; `purge=true` unlinks) + cascade-delete its KB rows, then reload store + compact vectors. Refuses a live session (409) unless `force=true` |
| `GET  /api/sessions/by-file?path=&limit=` | sessions that touched a file (suffix-matched) |
| `GET  /api/graph?project=` | session-overview topology |
| `GET  /api/recall?q=&limit=&min_score=&project=&exclude=&boost_project=` | semantic recall → `{sessions, lessons}` (sessions w/ decay + file boost + soft same-`boost_project` boost; `lessons` = top distilled lessons + auto-memory facts, gap-trimmed) |
| `GET  /api/observe/{sid}` | read obs + label + summary for a session |
| `POST /api/observe/{sid}[?sync=true]` | (re-)generate obs (background by default) |
| `POST /api/observe/backfill` | generate for every session missing obs |
| `POST /api/labels/backfill[?force=true&use_llm=true]` | re-derive kebab labels |
| `POST /api/capture/{start,prompt,tool,end}` | hooks talk to these |
| `GET  /api/capture/stats` | capture-store health (sessions / prompts / tools / active) |
| `GET  /api/worker/status` | tool-call summarizer state (running, batches processed, …) |
| `POST /api/worker/tick[?min_age_s=N]` | drive one summarizer pass synchronously |
| `POST /api/lessons/distill[?days=N&sync=true&replace=false]` | distill from last N days of obs; **merges** by default (`replace=true` wipes+rewrites) |
| `GET  /api/lessons[?tag=&limit=]` | list distilled lessons |
| `GET  /api/lessons/status` | distillation state + schedule (interval_s, days, scheduled) |
| `GET  /api/memory[?type=&limit=]` | list ingested auto-memory facts (filter by user\|feedback\|project\|reference) |
| `POST /api/memory/sync` | re-scan `~/.claude/projects/*/memory/*.md` now → returns ingest stats |
| `GET  /api/memory/status` | last-scan stats + watch-loop interval |
| `POST /api/index/compact` | drop orphaned embedding vectors now → `{removed, indexed}` (also auto-runs from the workers) |
| `POST /api/legacy/import` + `import-observations` | one-shot bootstrap endpoints used during the historical migration |
| `GET  /api/stream` | SSE; `refresh` when our DB grows, else `ping` with progress |

## Files

| File | Role |
|------|------|
| `app.py` | FastAPI: REST + SSE + serves the frontend + worker thread orchestration |
| `store.py` | Live read of `~/.claude-kb/data.db`; projects observations + **lessons** + **memory_facts** into one record corpus for embed/search/recall; exposes `filesCount` + `worktree` per session |
| `store_capture.py` | Capture store: sessions/prompts/observations/**tool_calls**/**session_files**/**lessons**/**memory_facts** tables + helpers |
| `memory_ingest.py` | Sync `~/.claude/projects/*/memory/*.md` (frontmatter + body) → `memory_facts`; dependency-free parser, mtime/hash-gated, prunes deleted files |
| `observe.py` | Three DeepSeek prompts: per-session observations from prompts, per-batch observations from tool calls, cross-session lessons distiller |
| `embeddings.py` | `all-MiniLM-L6-v2` embeddings, cosine search, incremental disk cache |
| `llm.py` | DeepSeek client for kebab labels + 1–2 sentence summaries; hash-keyed disk cache |
| `mcp_server.py` | Stdio MCP server exposing the 5 tools |
| `hooks/recall.py` | `UserPromptSubmit` hook — injects a lessons/memory block + related sessions |
| `hooks/capture.py` | `SessionStart` / `UserPromptSubmit` / `PostToolUse` / `Stop` hook dispatcher |
| `scripts/rename-session.py` | Rename an old (closed) Claude Code session by editing its transcript JSONL + syncing the kb label |
| `scripts/delete-session.py` | CLI to delete sessions: trash (default) or `--purge` the transcript + `--sync-kb` to drop DB rows; thin wrapper over `session_delete.py` |
| `session_delete.py` | Shared core for session deletion (transcript trash/purge + KB cascade delete); used by the CLI, the `DELETE /api/sessions/{sid}` endpoint, and tests |
| `scripts/backfill-worktree-projects.py` | One-shot: rewrite legacy sessions whose project was set to a worktree dir name |
| `skills/rename-session/SKILL.md` | Skill definition the global Claude registry picks up |
| `skills/delete-session/SKILL.md` | Skill for deleting old sessions from the CLI |
| `static/index.html` | Explainer + teaching guide |
| `static/kb.html` + `kb.js` + `theme.css` | The KB browser — cards + D3 force-graph + EventSource client |
| `deploy/claude-session-kb.service` | systemd user-unit template (auto-start/restart the backend) |
| `tests/` | pytest unit tests for the pure logic (ranking, merge, ingest, tags, projection) |
| `.cache/` | Embedding cache + LLM response cache (git-ignored) |

## Worktree-aware capture

Sessions started inside `<repo>/.claude/worktrees/<name>/...` are attributed
to the **parent repo** (so `my-api-service` shows up in the project
dropdown, not its dozens of transient worktrees), while the worktree name
is preserved as a `wt: <name>` chip on the session card and a derived
`worktree` field on the session dict. No schema migration; the worktree is
parsed from `cwd` at read time. Existing rows can be fixed with:

```bash
python3 scripts/backfill-worktree-projects.py
```

## Optional LLM (DeepSeek)

Set `DEEPSEEK_API_KEY` in your environment to enable:

- **Topical observations from prompts** (Stop hook + `/api/observe/{sid}`) —
  3–7 per session, with domain-specific tags.
- **Topical observations from tool calls** (background worker) — 1–4 per
  batch of ~50 PostToolUse events.
- **Cross-session lessons** (manual `/api/lessons/distill`) — 5–15 durable
  patterns across the corpus.
- **Kebab labels** per session — `auth-refactor`, `cache-invalidation`, etc.
- **1–2 sentence summaries** per session.

All five share `llm.py`'s disk cache at `.cache/llm.json`, keyed by
`sha256(model + kind + system_prompt + payload)`. First run on a
100-session corpus cost ~$0.05; every subsequent restart is free. A full
lessons distillation over 4,800 observations is a single DeepSeek call
(~$0.005).

Without `DEEPSEEK_API_KEY` the system degrades cleanly: tool-call capture
still writes to the queue (and the file index still works for recall
boost), prompts still capture, recall still ranks — only the LLM-generated
text fields go missing.

**Privacy:** enabling DeepSeek sends session content (your prompts +
observation titles + truncated tool inputs) to their API under their TOS.
Sessions can contain confidential project data — opt in only if that's
acceptable. The base capture + recall stack runs fully local without the
key.

## Notes

- Everything writes to `~/.claude-kb/` only. No remote DB, no telemetry, no
  cloud sync.
- D3 loads from a CDN for the web UI (first open needs internet); the
  embedding model is server-side, fetched once to `~/.cache/huggingface`.
- UI screenshots are intentionally omitted from this repo — they would
  otherwise leak real session content from confidential work.
- The full teaching guide (every feature, every interaction, every
  workflow) lives at `/` when the server is running —
  <http://127.0.0.1:8000>.
