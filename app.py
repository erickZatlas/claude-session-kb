"""
app.py — FastAPI backend for the Claude knowledge base.

  REST:
    GET /api/meta                                  counts, projects, freshness
    GET /api/search?q=&project=&kind=&mode=&limit= ranked records (keyword | semantic)
    GET /api/record/{id}                           one record + its session
    GET /api/sessions?project=                     sessions
  SSE:
    GET /api/stream                                pushes a `refresh` event when claude-mem grows
  Static:
    /                                              the explainer (static/index.html)
    /kb.html                                       the knowledge base UI

Reads ~/.claude-mem live (read-only) and re-reads automatically when it grows, so the
API is always current. Semantic search embeds the query server-side (no browser model).

Run:  python app.py     (or: uvicorn app:app --reload)
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import llm
import memory_ingest
import observe
from embeddings import Embedder
from store import Enricher, Store
from store_capture import CaptureStore

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

store = Store()
embedder = Embedder()
enricher = Enricher()
capture = CaptureStore()  # Phase B: dual-write capture into our own SQLite
_state = {"indexing": False}
_sync_lock = threading.Lock()

# Auto-memory ingest (~/.claude/projects/*/memory/*.md → memory_facts). Polled
# on a throttle from the same code path as freshness, since the files change on
# disk independently of our DB. A scan only writes when a file's mtime+hash
# changed, so the common case is a cheap no-op and ensure_fresh() stays quiet.
_memory_lock = threading.Lock()
_memory_state = {"last_scan_at": 0.0, "scanned": 0, "inserted": 0,
                 "updated": 0, "unchanged": 0, "pruned": 0}
MEMORY_SCAN_THROTTLE_S = float(os.environ.get("SESSION_KB_MEMORY_SCAN_THROTTLE_S", "30"))


def _index(records):
    """Embed any not-yet-indexed records (background)."""
    if not _sync_lock.acquire(blocking=False):
        return
    try:
        _state["indexing"] = True
        embedder.sync(records)
    finally:
        _state["indexing"] = False
        _sync_lock.release()


def _enrich():
    """Background LLM pass: clean labels + session summaries via DeepSeek (cached)."""
    enricher.sync(store.sessions, store.records)


def _scan_memory() -> dict:
    """Run one auto-memory ingest pass (disk → memory_facts). Writes only on
    changed files, so the bumped updated_at lets ensure_fresh() pick them up.
    Never raises — a missing memory dir just yields zeros."""
    try:
        stats = memory_ingest.scan(capture)
    except Exception:
        return dict(_memory_state)
    _memory_state.update(stats)
    _memory_state["last_scan_at"] = time.time()
    return dict(_memory_state)


def _maybe_scan_memory() -> None:
    """Throttled, non-blocking memory scan. Skipped if another scan holds the
    lock or the last scan was within MEMORY_SCAN_THROTTLE_S."""
    if (time.time() - _memory_state["last_scan_at"]) < MEMORY_SCAN_THROTTLE_S:
        return
    if not _memory_lock.acquire(blocking=False):
        return
    try:
        _scan_memory()
    finally:
        _memory_lock.release()


# Phase E.B: background tool-call summarizer. Wakes every WORKER_INTERVAL_S
# seconds, drains pending tool_calls in batches per session, asks DeepSeek
# for 1–4 observations per batch, marks the batch processed.
_worker_stop = threading.Event()
_worker_lock = threading.Lock()
_worker_state = {"running": False, "sessions_done": 0, "obs_written": 0,
                 "batches_processed": 0, "failed": 0, "last_tick_at": None}
WORKER_INTERVAL_S = int(os.environ.get("SESSION_KB_WORKER_INTERVAL_S", "60"))
WORKER_MIN_AGE_S = int(os.environ.get("SESSION_KB_WORKER_MIN_AGE_S", "300"))
WORKER_BATCH = int(os.environ.get("SESSION_KB_WORKER_BATCH", "50"))


def _tool_summarizer_tick():
    """One pass of the worker. Returns count of observations written this tick.
    Factored out so tests can drive a single iteration synchronously."""
    written = 0
    if not _worker_lock.acquire(blocking=False):
        return 0
    try:
        sids = capture.sessions_with_pending_tools(
            min_age_s=WORKER_MIN_AGE_S, max_sessions=20,
        )
        for sid in sids:
            sess = capture.get_session(sid) or {"id": sid}
            batch = capture.pop_pending_tools(sid, limit=WORKER_BATCH)
            if not batch:
                continue
            try:
                obs = observe.generate_tool_observations(sess, batch)
            except Exception:
                obs = []
                _worker_state["failed"] += 1
            if obs:
                capture.append_observations(sid, obs)
                written += len(obs)
            capture.mark_tools_processed([b["id"] for b in batch])
            _worker_state["batches_processed"] += 1
            _worker_state["sessions_done"] += 1
        _worker_state["obs_written"] += written
        _worker_state["last_tick_at"] = int(time.time())
    finally:
        _worker_lock.release()
    return written


def _tool_summarizer_loop():
    _worker_state["running"] = True
    try:
        while not _worker_stop.wait(WORKER_INTERVAL_S):
            try:
                _tool_summarizer_tick()
            except Exception:
                _worker_state["failed"] += 1
    finally:
        _worker_state["running"] = False


def _refresh_and_kick() -> bool:
    """Reload the store if our DB grew; if it did, kick background indexing and
    LLM enrichment for whatever new records arrived. Every read endpoint should
    call this so freshness + enrichment stay in step on the same code path.

    Defensive on purpose: a missing/locked/corrupt DB must NOT take down the
    read endpoints. We swallow any error, fall back to the in-memory snapshot."""
    try:
        # Pick up any edited/added auto-memory files first; a write here bumps
        # memory_facts.updated_at so ensure_fresh() reloads + re-embeds below.
        _maybe_scan_memory()
        if store.ensure_fresh():
            threading.Thread(target=_index, args=(store.records,), daemon=True).start()
            threading.Thread(target=_enrich, daemon=True).start()
            return True
    except Exception:
        pass
    return False


@asynccontextmanager
async def lifespan(_app):
    # Sync auto-memory before the first load so memory facts are in the corpus
    # (and get embedded) from the start, not only after the first prompt.
    _scan_memory()
    store.reload()
    # Embed in the background so keyword search is available immediately; the first
    # run embeds everything (minutes on CPU) and caches to .cache for fast restarts.
    threading.Thread(target=_index, args=(store.records,), daemon=True).start()
    # LLM enrichment runs alongside; degrades to TF-IDF if DEEPSEEK_API_KEY is missing.
    threading.Thread(target=_enrich, daemon=True).start()
    # Phase E.B: tool-call summarizer drains the queue every WORKER_INTERVAL_S
    _worker_stop.clear()
    threading.Thread(target=_tool_summarizer_loop, daemon=True).start()
    yield
    _worker_stop.set()


app = FastAPI(title="Claude Knowledge Base", lifespan=lifespan)


def _clean(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "_blob"}


@app.get("/api/meta")
def api_meta():
    _refresh_and_kick()
    m = store.meta()
    m["indexing"] = _state["indexing"]
    m["indexed"] = len(embedder.ids)
    m["enriching"] = enricher.running
    m["enriched"] = enricher.done
    m["enrichTotal"] = enricher.total
    return m


@app.get("/api/search")
def api_search(
    q: str = "",
    project: str = "all",
    kind: str = "all",
    mode: str = Query("keyword", pattern="^(keyword|semantic)$"),
    session: str = "",
    limit: int = 250,
):
    _refresh_and_kick()

    sess = session or None
    used = mode
    if mode == "semantic" and q and embedder.matrix.size and not _state["indexing"]:
        recs = embedder.search(q, store.candidates(project, kind, sess), limit)
    else:
        if mode == "semantic":
            used = "indexing" if _state["indexing"] else "keyword"
        recs = store.keyword_search(q, project, kind, limit, sess)

    return {"mode": used, "total": len(recs), "results": [_clean(r) for r in recs]}


@app.get("/api/record/{rec_id}")
def api_record(rec_id: str):
    r = store.rec_by_id.get(rec_id)
    if not r:
        raise HTTPException(404, "record not found")
    out = _clean(r)
    out["session"] = store.sess_by_id.get(r.get("sessionId"))
    return out


@app.get("/api/graph")
def api_graph(project: str = "all"):
    """Session-overview topology (sessions + shared-file edges). The frontend's main
    overview is now the timeline (uses /api/sessions instead), so this endpoint is
    currently unused by kb.js — kept in place so a 'map' view can be reinstated as a
    toggle with a one-line render switch."""
    _refresh_and_kick()
    return store.session_graph(project)


@app.get("/api/sessions")
def api_sessions(project: str = "all"):
    s = store.sessions if project == "all" else [x for x in store.sessions if x["project"] == project]
    return {"sessions": s}


# Phase E.C: ranking helpers for the recall hook.
# Half-life is configurable so heavy users can tighten it (e.g., 14 days) and
# casual users can loosen it (e.g., 60 days). Set to 99999 to effectively
# disable the decay.
HALFLIFE_DAYS = float(os.environ.get("SESSION_KB_HALFLIFE_DAYS", "30"))
# Max boost any single session can get from file overlap, additive on top of
# the (decayed) cosine score. Keeps semantic match dominant.
FILE_BOOST_CAP = float(os.environ.get("SESSION_KB_FILE_BOOST_CAP", "0.25"))
FILE_BOOST_PER_PATH = float(os.environ.get("SESSION_KB_FILE_BOOST_PER_PATH", "0.10"))

# Durable knowledge (distilled lessons + hand-authored auto-memory) is surfaced
# in its own recall block. It's cheap to show and high-value, so its threshold
# is looser than sessions' and it is NOT time-decayed — durability is the point.
KNOWLEDGE_MIN_SCORE = float(os.environ.get("SESSION_KB_KNOWLEDGE_MIN_SCORE", "0.28"))
KNOWLEDGE_LIMIT = int(os.environ.get("SESSION_KB_KNOWLEDGE_LIMIT", "4"))

# Same regex shape as the Bash file extractor in hooks/capture.py — used to
# pull file-shaped tokens from the user's prompt at recall time.
_QUERY_FILE_RE = __import__("re").compile(
    r"[A-Za-z0-9_./~-]+\.(?:py|ts|tsx|js|jsx|kt|java|kts|md|sql|json|yaml|yml|sh|bash|html|css|toml|ini|cfg|xml|gradle|rs|go|rb)"
)


def _decay(epoch_ms, halflife_days=HALFLIFE_DAYS):
    """Returns a multiplier in (0, 1]. Older = smaller. Anything from the
    future or right now is 1.0."""
    if not epoch_ms or halflife_days >= 99999:
        return 1.0
    age_days = (time.time() * 1000 - float(epoch_ms)) / (1000 * 86400)
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / halflife_days)


def _extract_query_files(q):
    """Pull file-shaped tokens out of a prompt. Dedupes case-insensitively."""
    seen = set()
    out = []
    for m in _QUERY_FILE_RE.findall(q or ""):
        k = m.lower()
        if k not in seen:
            seen.add(k)
            out.append(m)
    return out[:8]


def _recall_knowledge(sims, min_score=KNOWLEDGE_MIN_SCORE,
                      limit=KNOWLEDGE_LIMIT, project="all"):
    """Rank durable knowledge (lessons + auto-memory facts) by raw cosine.
    No time decay — durability is the point. Cheap: walks only the few dozen
    knowledge records, reusing the already-computed `sims` row per embedded id.
    Returns compact dicts for the recall payload's `lessons` field; [] when
    nothing is indexed yet or clears the threshold (clean empty-table degrade).
    """
    scored = []
    for r in store.knowledge_records():
        # Lessons are global (project=""); memory facts are scoped — when a
        # specific project is asked for, show that project's facts + global ones.
        if project != "all" and r.get("kind") == "memory":
            rp = r.get("project") or ""
            if rp not in (project, "global", ""):
                continue
        idx = embedder.index.get(r["id"])
        if idx is None:
            continue
        score = float(sims[int(idx)])
        if score < min_score:
            continue
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    out = []
    for score, r in scored[:limit]:
        item = {
            "kind": r.get("kind"),
            "title": r.get("title") or "",
            "text": (r.get("text") or "")[:400],
            "tags": r.get("concepts") or [],
            "score": round(score, 3),
        }
        if r.get("kind") == "memory":
            item["memType"] = r.get("memType")
            item["project"] = r.get("project") or ""
            item["sourcePath"] = r.get("sourcePath") or ""
        else:
            item["evidence"] = r.get("evidenceCount", 0)
        out.append(item)
    return out


@app.get("/api/recall")
def api_recall(
    q: str = "",
    limit: int = 3,
    project: str = "all",
    exclude: str = "",
    min_score: float = 0.30,
):
    """Pre-emptive recall with three ranking signals:
      1. Cosine similarity over MiniLM embeddings (the primary signal)
      2. Time decay (half-life HALFLIFE_DAYS) so 6-month-old sessions don't
         outrank last-week's at the same cosine
      3. File-overlap boost (+0.10 per path, cap +0.25) when the query
         mentions files this session touched per session_files

    Final session score = max over its records of (cosine * decay) + file_boost.
    """
    _refresh_and_kick()
    q = (q or "").strip()
    if not q or embedder.matrix.size == 0:
        return {"sessions": [], "lessons": [], "indexing": _state["indexing"]}
    qv = embedder.embed_query(q)
    sims = embedder.matrix @ qv  # one cosine per indexed record
    # Wider over-fetch than before — decay can demote top semantic matches
    # if they're old, and boost can promote previously-below-threshold ones.
    order = np.argsort(-sims)[: max(limit * 30, 60)]
    query_files = _extract_query_files(q)

    # Walk in raw-cosine order, compute the final score per record, dedupe by
    # session keeping that session's best record. Can no longer break early
    # when raw cosine dips below min_score (boost can lift it back), so we
    # filter after the loop.
    best_per_session: dict[str, tuple[float, dict]] = {}
    for i in order:
        raw = float(sims[i])
        rec_id = embedder.ids[int(i)]
        r = store.rec_by_id.get(rec_id)
        if not r:
            continue
        if project != "all" and r.get("project") != project:
            continue
        sid = r.get("sessionId")
        if not sid or sid == exclude:
            continue
        # Per-record signal (decayed cosine), then add the session-level boost
        cosine_decayed = raw * _decay(r.get("epoch"))
        boost = 0.0
        if query_files:
            try:
                overlap = capture.session_file_overlap(sid, query_files)
            except Exception:
                overlap = 0
            if overlap:
                boost = min(FILE_BOOST_CAP, FILE_BOOST_PER_PATH * overlap)
        final = cosine_decayed + boost
        if final < min_score:
            continue
        # Keep the session's best record for the response payload
        cur = best_per_session.get(sid)
        if cur is None or final > cur[0]:
            best_per_session[sid] = (final, r)
    # Sort by final score and dedupe-by-session is already done
    ranked = sorted(best_per_session.items(), key=lambda kv: -kv[1][0])
    sessions = []
    for sid, (final, r) in ranked[:limit]:
        s = store.sess_by_id.get(sid)
        if not s:
            continue
        sessions.append({
            "id": sid,
            "label": s.get("label"),
            "title": (s.get("title") or "")[:120],
            "summary": s.get("summary") or "",
            "project": s.get("project"),
            "started": s.get("started"),
            "obsCount": s.get("obsCount", 0),
            "score": round(final, 3),
            "matchedTitle": (r.get("title") or "")[:160],
        })
    return {
        "sessions": sessions,
        "lessons": _recall_knowledge(sims, project=project),
        "indexing": _state["indexing"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — capture pipeline. The Claude Code hook scripts at hooks/capture.py
# POST to these endpoints on SessionStart / UserPromptSubmit / Stop. Runs in
# parallel with claude-mem during the dual-write period.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel  # local import to keep startup cost out of the API


class _CaptureStart(BaseModel):
    session_id: str
    project: str | None = None
    cwd: str | None = None
    started_at: int | None = None


class _CapturePrompt(BaseModel):
    session_id: str
    text: str
    project: str | None = None
    cwd: str | None = None
    ts: int | None = None


class _CaptureEnd(BaseModel):
    session_id: str
    ts: int | None = None


class _CaptureTool(BaseModel):
    session_id: str
    ts: int
    tool_name: str
    tool_input: str | None = None
    tool_response: str | None = None
    files: list[str] = []
    kind: str = "mentioned"
    project: str | None = None
    cwd: str | None = None


@app.post("/api/capture/start")
def api_capture_start(body: _CaptureStart):
    capture.session_start(body.session_id, body.project, body.cwd, body.started_at)
    return {"ok": True}


@app.post("/api/capture/prompt")
def api_capture_prompt(body: _CapturePrompt):
    text = (body.text or "").strip()
    if not text:
        return {"ok": True, "skipped": "empty"}
    capture.record_prompt(body.session_id, text, body.project, body.cwd, body.ts)
    return {"ok": True}


@app.post("/api/capture/end")
def api_capture_end(body: _CaptureEnd):
    capture.session_end(body.session_id, body.ts)
    return {"ok": True}


@app.post("/api/capture/tool")
def api_capture_tool(body: _CaptureTool):
    """Phase E: one row per PostToolUse event. The background summarizer
    (started in the lifespan) drains pending rows in batches."""
    tc_id = capture.record_tool_call(
        session_id=body.session_id,
        ts=body.ts,
        tool_name=body.tool_name,
        tool_input=body.tool_input,
        tool_response=body.tool_response,
        files=body.files or [],
        kind=body.kind or "mentioned",
        project=body.project,
        cwd=body.cwd,
    )
    return {"ok": True, "id": tc_id}


@app.get("/api/worker/status")
def api_worker_status():
    """Visibility into the tool-call summarizer loop."""
    return {
        **_worker_state,
        "interval_s": WORKER_INTERVAL_S,
        "min_age_s": WORKER_MIN_AGE_S,
        "batch": WORKER_BATCH,
    }


@app.post("/api/worker/tick")
def api_worker_tick(min_age_s: int | None = None):
    """Drive one summarizer pass synchronously. Useful for tests / when the
    user wants their session's tool calls summarized RIGHT NOW instead of
    waiting for the next 60s tick. Returns the number of observations written.
    `min_age_s=0` ignores the 'wait 5 min before summarizing a session' guard."""
    global WORKER_MIN_AGE_S
    saved = WORKER_MIN_AGE_S
    if min_age_s is not None:
        WORKER_MIN_AGE_S = max(0, int(min_age_s))
    try:
        written = _tool_summarizer_tick()
    finally:
        WORKER_MIN_AGE_S = saved
    return {"ok": True, "observations_written": written, **_worker_state}


@app.get("/api/capture/stats")
def api_capture_stats():
    """Capture-store health check: row counts in our SQLite."""
    return capture.stats()


@app.get("/api/capture/sessions")
def api_capture_sessions(limit: int = 50):
    return {"sessions": capture.list_sessions(limit)}


# ─────────────────────────────────────────────────────────────────────────────
# Phase C — observation generation. Reads captured prompts, asks DeepSeek for
# topical observations + a session label/summary, stores them on our side.
# The cache key includes the system prompt so prompt tuning is free.
# ─────────────────────────────────────────────────────────────────────────────

_observe_lock = threading.Lock()
_observe_state = {"running": False, "done": 0, "total": 0, "failed": 0}


def _observe_one(session_id: str) -> dict:
    """Synchronous generation for one session. Returns a small status dict."""
    sess = capture.get_session(session_id)
    if not sess:
        return {"ok": False, "error": "session not found"}
    prompts = capture.prompts_for(session_id)
    if not prompts:
        return {"ok": False, "error": "no prompts captured for this session"}
    obs = observe.generate_observations(sess.get("project"), prompts)
    label, summary = observe.clarify_session(sess.get("project"), prompts)
    capture.set_session_clarification(session_id, label, summary)
    n = capture.replace_observations(session_id, obs)
    return {"ok": True, "session_id": session_id, "observations": n,
            "label": label, "summary": summary}


@app.post("/api/observe/{session_id}")
def api_observe_one(session_id: str, sync: bool = False):
    """Generate observations for one session. Default fires-and-forgets in a
    background thread (so the Stop hook returns fast). Pass ?sync=true to wait
    for the result — used by smoke tests and manual triggers from the UI."""
    if sync:
        return _observe_one(session_id)
    threading.Thread(target=_observe_one, args=(session_id,), daemon=True).start()
    return {"started": True, "session_id": session_id}


def _label_from_observations(observations: list[dict],
                             project: Optional[str]) -> Optional[str]:
    """Pick the most-frequent topical token across a session's observation
    titles + tags and kebab-ify it. Used to backfill labels for Phase D sessions
    that have observations but no captured prompts (so the prompt-driven
    `clarify_session` path can't reach them)."""
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    next_ord = 0
    # Tags are pre-extracted, weight them higher (2x) since they're already topical.
    for o in observations:
        for t in (o.get("tags") or []):
            key = t.strip()
            if not key or len(key) < 3:
                continue
            if key not in counts:
                counts[key] = 0
                first_seen[key] = next_ord
                next_ord += 1
            counts[key] += 2
        # Tokens from the title with the same extractor used at import time
        for t in observe.extract_topical_tags(o.get("title")):
            if not t or len(t) < 3:
                continue
            if t not in counts:
                counts[t] = 0
                first_seen[t] = next_ord
                next_ord += 1
            counts[t] += 1
    if not counts:
        return llm._to_kebab(project) if project else None
    # Prefer multi-word / multi-token labels — they're more specific. So bias
    # against single-token all-caps acronyms when something richer is available.
    def score(item):
        tok, n = item
        rich = ("-" in tok) or ("_" in tok) or any(c.islower() for c in tok)
        return (-n, 0 if rich else 1, first_seen[tok])
    top = sorted(counts.items(), key=score)[0][0]
    return llm._to_kebab(top)


def _label_payload_for_session(session_id: str, project: Optional[str],
                                cm_request: Optional[str]) -> str:
    """Compact LLM payload for labelling a single session: claude-mem's
    `request` (when present — it's a plain-English description of what the
    session was about), plus the first 3 observation titles for added topic
    signal."""
    obs = capture.observations_for(session_id)
    titles = [o.get("title") or "" for o in obs[:3] if o.get("title")]
    parts = [f"Project: {project or 'unknown'}"]
    if cm_request:
        parts.append(f"Request: {cm_request.strip()[:400]}")
    if titles:
        parts.append("First observations:")
        for t in titles:
            parts.append(f"- {t.strip()[:160]}")
    return "\n".join(parts)[:2000]


def _backfill_labels_worker(limit: int, force: bool, use_llm: bool) -> None:
    """Walk sessions that need a label. For each: try the LLM with the rich
    request/observation payload; fall back to the cheap tag-frequency heuristic
    if the LLM call fails (no API key, network blip, cache miss)."""
    import sqlite3 as _sql
    if not _label_lock.acquire(blocking=False):
        return
    try:
        _label_state.update(running=True, done=0, failed=0)
        cm_path = os.path.expanduser("~/.claude-mem/claude-mem.db")
        cm = (_sql.connect(f"file:{cm_path}?mode=ro", uri=True)
              if os.path.exists(cm_path) else None)
        if cm is not None:
            cm.row_factory = _sql.Row
        try:
            with capture._conn() as c:
                rows = c.execute(
                    "SELECT s.id, s.project, s.label FROM sessions s "
                    "WHERE EXISTS (SELECT 1 FROM observations o WHERE o.session_id = s.id) "
                    + ("" if force else "AND s.label IS NULL ")
                    + "ORDER BY s.started_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            _label_state["total"] = len(rows)
            for r in rows:
                sid = r["id"]
                project = r["project"]
                cm_req = None
                if cm is not None:
                    summ = cm.execute(
                        "SELECT request FROM session_summaries ss "
                        "JOIN sdk_sessions s ON s.memory_session_id = ss.memory_session_id "
                        "WHERE s.content_session_id = ? ORDER BY ss.created_at_epoch DESC LIMIT 1",
                        (sid,),
                    ).fetchone()
                    cm_req = (summ["request"] if summ else None)
                label = None
                if use_llm and (cm_req or capture.observations_for(sid)):
                    payload = _label_payload_for_session(sid, project, cm_req)
                    try:
                        label = llm.label_for(payload)
                    except Exception:
                        label = None
                if not label:
                    label = _label_from_observations(
                        capture.observations_for(sid), project)
                if label:
                    capture.set_session_clarification(sid, label, None)
                    _label_state["done"] += 1
                else:
                    _label_state["failed"] += 1
        finally:
            if cm is not None:
                cm.close()
    finally:
        _label_state["running"] = False
        _label_lock.release()


_label_lock = threading.Lock()
_label_state = {"running": False, "done": 0, "total": 0, "failed": 0}


@app.post("/api/labels/backfill")
def api_backfill_labels(limit: int = 1000, force: bool = False,
                        use_llm: bool = True, sync: bool = False):
    """Derive a kebab label for sessions that lack one (Phase D leftovers).
    By default uses DeepSeek with the rich claude-mem `request` + first 3
    observation titles as payload (good labels). Falls back to a cheap
    tag-frequency heuristic when the LLM is unavailable.

    `force=true`  → re-label every session, even ones that already have one.
    `use_llm=false` → skip the LLM entirely; use heuristic only.
    `sync=true`   → block until done (useful for one-shot bulk runs)."""
    if sync:
        _backfill_labels_worker(limit, force, use_llm)
        return {"ok": True, **_label_state}
    threading.Thread(target=_backfill_labels_worker,
                     args=(limit, force, use_llm), daemon=True).start()
    return {"started": True, "limit": limit, "force": force, "use_llm": use_llm}


@app.get("/api/labels/status")
def api_labels_status():
    return dict(_label_state)


@app.get("/api/observe/{session_id}")
def api_observe_get(session_id: str):
    sess = capture.get_session(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    return {
        "session": {k: sess.get(k) for k in ("id", "project", "first_prompt",
                                              "label", "summary", "prompt_count")},
        "observations": capture.observations_for(session_id),
    }


def _backfill_worker(limit: int) -> None:
    """Walks captured sessions that don't have observations yet, generates each."""
    if not _observe_lock.acquire(blocking=False):
        return
    try:
        _observe_state.update(running=True, done=0, failed=0)
        ids = capture.sessions_needing_observations(limit)
        _observe_state["total"] = len(ids)
        for sid in ids:
            try:
                res = _observe_one(sid)
                if res.get("ok"):
                    _observe_state["done"] += 1
                else:
                    _observe_state["failed"] += 1
            except Exception:
                _observe_state["failed"] += 1
    finally:
        _observe_state["running"] = False
        _observe_lock.release()


@app.post("/api/observe/backfill")
def api_observe_backfill(limit: int = 200):
    """Generate observations for every captured session that doesn't have any
    yet. Fire-and-forget; poll status via GET /api/observe/status."""
    threading.Thread(target=_backfill_worker, args=(limit,), daemon=True).start()
    return {"started": True, "limit": limit}


@app.get("/api/observe/status")
def api_observe_status():
    return dict(_observe_state)


# ─────────────────────────────────────────────────────────────────────────────
# Phase E.D — cross-session lessons. /api/lessons/distill runs the LLM over
# the observation corpus and replaces the lessons table; /api/lessons reads
# them back. Triggered manually for now (no scheduled job).
# ─────────────────────────────────────────────────────────────────────────────

_lessons_lock = threading.Lock()
_lessons_state = {"running": False, "lessons_written": 0, "obs_scanned": 0,
                  "last_distilled_at": None, "last_error": None}


def _distill_worker(days: int):
    if not _lessons_lock.acquire(blocking=False):
        return
    try:
        _lessons_state.update(running=True, last_error=None)
        since_ms = int(time.time() * 1000) - days * 86400 * 1000
        obs = capture.observations_since(since_ms, limit=5000)
        _lessons_state["obs_scanned"] = len(obs)
        if not obs:
            return
        lessons = observe.distill_lessons(obs)
        if lessons:
            now = int(time.time() * 1000)
            for L in lessons:
                L["first_seen"] = since_ms
                L["last_seen"] = now
                L["evidence_count"] = len(L.get("source_session_ids") or [])
            n = capture.replace_lessons(lessons)
            _lessons_state["lessons_written"] = n
        _lessons_state["last_distilled_at"] = int(time.time())
    except Exception as e:
        _lessons_state["last_error"] = f"{type(e).__name__}: {e}"
    finally:
        _lessons_state["running"] = False
        _lessons_lock.release()


@app.post("/api/lessons/distill")
def api_lessons_distill(days: int = 30, sync: bool = False):
    """Run the distiller over the last `days` of observations and rewrite the
    lessons table. Pass sync=true to block until done (1 DeepSeek call, ~10s);
    default fires-and-forgets in a background thread so the caller returns
    instantly."""
    if sync:
        _distill_worker(days)
        return {"ok": True, **_lessons_state}
    threading.Thread(target=_distill_worker, args=(days,), daemon=True).start()
    return {"started": True, "days": days}


@app.get("/api/lessons")
def api_lessons(tag: str | None = None, limit: int = 50):
    return {"lessons": capture.list_lessons(tag=tag, limit=limit)}


@app.get("/api/lessons/status")
def api_lessons_status():
    return dict(_lessons_state)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-memory facts — the hand-authored ~/.claude/projects/*/memory/*.md files,
# mirrored into memory_facts and projected into the shared corpus by store.py.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/memory")
def api_memory(type: str | None = None, limit: int = 200):
    """List ingested auto-memory facts (optionally filtered by mem_type:
    user|feedback|project|reference)."""
    return {"memory": capture.list_memory_facts(mem_type=type, limit=limit)}


@app.post("/api/memory/sync")
def api_memory_sync():
    """Re-scan the auto-memory directories now (disk → memory_facts). Returns
    ingest stats. The throttled poll in _refresh_and_kick picks up changes
    automatically too; this is the manual 'right now' trigger."""
    with _memory_lock:
        stats = _scan_memory()
    return {"ok": True, **stats}


@app.get("/api/sessions/by-file")
def api_sessions_by_file(path: str, limit: int = 10):
    """Sessions that touched a given file. Used by the file-aware MCP tool
    and by the UI's file-chip drill-in."""
    if not path:
        raise HTTPException(400, "path is required")
    return {"path": path, "sessions": capture.sessions_for_file(path, limit=limit)}


@app.post("/api/legacy/import")
def api_legacy_import(limit: int = 200):
    """One-shot seed of our capture DB from claude-mem's existing data so Phase C
    has real material to generate against before new captures accumulate.
    Idempotent: sessions/prompts already imported are skipped."""
    import sqlite3 as _sql
    cm_path = os.path.expanduser("~/.claude-mem/claude-mem.db")
    if not os.path.exists(cm_path):
        raise HTTPException(404, f"claude-mem db not at {cm_path}")
    con = _sql.connect(f"file:{cm_path}?mode=ro", uri=True)
    con.row_factory = _sql.Row
    sessions_added = 0
    prompts_added = 0
    try:
        rows = con.execute(
            "SELECT content_session_id, memory_session_id, project, started_at_epoch, user_prompt "
            "FROM sdk_sessions ORDER BY started_at_epoch DESC LIMIT ?",
            (limit,),
        ).fetchall()
        for r in rows:
            sid = r["content_session_id"]
            if not sid:
                continue
            existed_before = capture.get_session(sid) is not None
            # Pull prompt texts for this session from user_prompts; fallback to user_prompt only
            prompt_rows = con.execute(
                "SELECT created_at_epoch, prompt_text FROM user_prompts "
                "WHERE content_session_id = ? ORDER BY created_at_epoch",
                (sid,),
            ).fetchall()
            prompts = [(int(p["created_at_epoch"] or 0),
                        (p["prompt_text"] or "").strip())
                       for p in prompt_rows if (p["prompt_text"] or "").strip()]
            if not prompts and r["user_prompt"]:
                prompts = [(int(r["started_at_epoch"] or 0),
                            (r["user_prompt"] or "").strip())]
            capture.import_legacy_session(
                session_id=sid,
                project=r["project"],
                started_at=int(r["started_at_epoch"] or 0),
                first_prompt=(r["user_prompt"] or (prompts[0][1] if prompts else None)),
                prompts=prompts,
            )
            if not existed_before:
                sessions_added += 1
            prompts_added += len(prompts)
    finally:
        con.close()
    return {"ok": True, "sessions_added": sessions_added,
            "prompts_added": prompts_added}


@app.post("/api/legacy/import-observations")
def api_legacy_import_observations(replace: bool = False, max_sessions: int = 1000):
    """Phase D: bulk-import claude-mem's historical observations + session
    summaries into our store. claude-mem's `concepts` field is the generic
    filler we ban (`how-it-works`, `what-changed`); we discard it and extract
    fresh topical tags from each row's rich title/subtitle/narrative.

    Idempotent: by default, a session that already has any observations in our
    table is skipped (so re-running won't duplicate, and Phase C-generated rows
    on still-active sessions are preserved). Pass ?replace=true to overwrite.
    """
    import sqlite3 as _sql
    cm_path = os.path.expanduser("~/.claude-mem/claude-mem.db")
    if not os.path.exists(cm_path):
        raise HTTPException(404, f"claude-mem db not at {cm_path}")
    con = _sql.connect(f"file:{cm_path}?mode=ro", uri=True)
    con.row_factory = _sql.Row
    summaries_set = 0
    sessions_seeded = 0
    sessions_imported = 0
    obs_imported = 0
    sessions_skipped_existing = 0
    try:
        # All claude-mem sessions that have at least one observation.
        sess_rows = con.execute(
            "SELECT DISTINCT s.content_session_id, s.memory_session_id, s.project, "
            "       s.started_at_epoch, s.user_prompt "
            "FROM sdk_sessions s "
            "JOIN observations o ON o.memory_session_id = s.memory_session_id "
            "ORDER BY s.started_at_epoch DESC LIMIT ?",
            (max_sessions,),
        ).fetchall()
        for s in sess_rows:
            sid = s["content_session_id"]
            if not sid:
                continue

            # Ensure the session shell exists on our side (idempotent INSERT OR IGNORE).
            if capture.get_session(sid) is None:
                capture.import_legacy_session(
                    session_id=sid,
                    project=s["project"],
                    started_at=int(s["started_at_epoch"] or 0),
                    first_prompt=(s["user_prompt"] or None),
                    prompts=[],  # prompts come from /api/legacy/import, not from this endpoint
                )
                sessions_seeded += 1

            # If observations already exist on our side, skip unless replace=true.
            if not replace and capture.observations_for(sid):
                sessions_skipped_existing += 1
            else:
                obs_rows = con.execute(
                    "SELECT type, title, subtitle, narrative, text, concepts, "
                    "       created_at_epoch "
                    "FROM observations WHERE memory_session_id = ? "
                    "ORDER BY created_at_epoch",
                    (s["memory_session_id"],),
                ).fetchall()
                normalized: list[dict] = []
                for o in obs_rows:
                    title = (o["title"] or "").strip()
                    subtitle = (o["subtitle"] or "").strip()
                    narrative = (o["narrative"] or "").strip()
                    text_blob = (o["text"] or "").strip()
                    if not title and not subtitle and not narrative and not text_blob:
                        continue
                    if not title:
                        # Fall back to first sentence of richest text field available
                        src = subtitle or narrative or text_blob
                        title = src.split(". ", 1)[0][:200]
                    body = narrative or subtitle or text_blob
                    tags = observe.extract_topical_tags(title, subtitle, narrative)
                    t = (o["type"] or "discovery").strip().lower()
                    if t not in ("discovery", "change", "bugfix", "decision", "refactor", "feature"):
                        t = "discovery"
                    normalized.append({
                        "type": t,
                        "title": title[:200],
                        "text": body,
                        "tags": tags,
                    })
                if normalized:
                    capture.replace_observations(sid, normalized)
                    sessions_imported += 1
                    obs_imported += len(normalized)

            # Set summary if missing on our side. Use claude-mem's `learned`
            # (richest) and fall back through the pipeline. Truncate generously
            # — these summaries are human-readable detail, not a chip label.
            existing = capture.get_session(sid) or {}
            if not existing.get("summary"):
                summ = con.execute(
                    "SELECT learned, completed, request "
                    "FROM session_summaries WHERE memory_session_id = ? "
                    "ORDER BY created_at_epoch DESC LIMIT 1",
                    (s["memory_session_id"],),
                ).fetchone()
                if summ:
                    candidate = (summ["learned"] or summ["completed"] or
                                 summ["request"] or "").strip()
                    if candidate:
                        capture.set_session_clarification(sid, None, candidate[:400])
                        summaries_set += 1
    finally:
        con.close()
    return {
        "ok": True,
        "sessions_seeded": sessions_seeded,
        "sessions_imported": sessions_imported,
        "sessions_skipped_existing": sessions_skipped_existing,
        "observations_imported": obs_imported,
        "summaries_set": summaries_set,
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/stream")
async def api_stream():
    """Server-Sent Events: emits `refresh` when claude-mem grows, else periodic `ping`."""
    async def gen():
        loop = asyncio.get_event_loop()
        yield _sse("meta", store.meta())
        while True:
            await asyncio.sleep(4)
            changed = await loop.run_in_executor(None, store.ensure_fresh)
            if changed:
                threading.Thread(target=_index, args=(store.records,), daemon=True).start()
                threading.Thread(target=_enrich, daemon=True).start()
                yield _sse("refresh", store.meta())
            else:
                yield _sse("ping", {
                    "indexing": _state["indexing"], "indexed": len(embedder.ids),
                    "enriching": enricher.running, "enriched": enricher.done,
                    "enrichTotal": enricher.total,
                })

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# StaticFiles wrapper that disables browser caching for the frontend assets.
# A long-cached kb.js can produce confusing label/render bugs after a deploy;
# the static set is tiny enough that re-fetching every time is fine.
class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp


def _bust(html: str) -> str:
    """Append ?v=<mtime> to local asset URLs so the browser can't refuse to refetch."""
    def ver(name: str) -> str:
        try:
            return str(int(os.path.getmtime(os.path.join(STATIC, name))))
        except OSError:
            return "0"
    return (
        html.replace('src="kb.js"', f'src="kb.js?v={ver("kb.js")}"')
            .replace('href="theme.css"', f'href="theme.css?v={ver("theme.css")}"')
    )


def _serve_html(name: str) -> HTMLResponse:
    body = open(os.path.join(STATIC, name), encoding="utf-8").read()
    return HTMLResponse(
        content=_bust(body),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/")
def root():
    return _serve_html("index.html")


@app.get("/kb.html")
def kb_html():
    return _serve_html("kb.html")


# Serve the rest of the frontend (kb.js, theme.css, etc). Mounted last so /api/* + the
# explicit HTML routes above win. The class also sends no-cache headers on every asset.
app.mount("/", NoCacheStaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
