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
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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


def _refresh_and_kick() -> bool:
    """Reload the store if claude-mem grew; if it did, kick background indexing and
    LLM enrichment for whatever new records arrived. Every read endpoint should call
    this so freshness + enrichment stay in step on the same code path."""
    if store.ensure_fresh():
        threading.Thread(target=_index, args=(store.records,), daemon=True).start()
        threading.Thread(target=_enrich, daemon=True).start()
        return True
    return False


@asynccontextmanager
async def lifespan(_app):
    store.reload()
    # Embed in the background so keyword search is available immediately; the first
    # run embeds everything (minutes on CPU) and caches to .cache for fast restarts.
    threading.Thread(target=_index, args=(store.records,), daemon=True).start()
    # LLM enrichment runs alongside; degrades to TF-IDF if DEEPSEEK_API_KEY is missing.
    threading.Thread(target=_enrich, daemon=True).start()
    yield


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


@app.get("/api/recall")
def api_recall(
    q: str = "",
    limit: int = 3,
    project: str = "all",
    exclude: str = "",
    min_score: float = 0.30,
):
    """Pre-emptive recall: given a query, return the top-N most semantically similar
    past sessions. Used by the UserPromptSubmit hook to surface related work to Claude
    before it sees the user's prompt.

    Algorithm: embed the query, cosine-score against every record vector, dedupe by
    session (a session's score = its best matching record's score), drop anything
    below min_score (configurable; default 0.30 keeps noise out)."""
    _refresh_and_kick()
    q = (q or "").strip()
    if not q or embedder.matrix.size == 0:
        return {"sessions": [], "indexing": _state["indexing"]}
    qv = embedder.embed_query(q)
    sims = embedder.matrix @ qv  # one cosine per indexed record
    # over-fetch a generous multiple so dedupe-by-session has room
    order = np.argsort(-sims)[: max(limit * 12, 24)]
    seen: set[str] = set()
    sessions = []
    for i in order:
        score = float(sims[i])
        if score < min_score:
            break  # sims are sorted desc; everything after is worse
        rec_id = embedder.ids[int(i)]
        r = store.rec_by_id.get(rec_id)
        if not r:
            continue
        if project != "all" and r.get("project") != project:
            continue
        sid = r.get("sessionId")
        if not sid or sid == exclude or sid in seen:
            continue
        seen.add(sid)
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
            "score": round(score, 3),
            "matchedTitle": (r.get("title") or "")[:160],
        })
        if len(sessions) >= limit:
            break
    return {"sessions": sessions, "indexing": _state["indexing"]}


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


@app.get("/api/capture/stats")
def api_capture_stats():
    """Dual-write health check: counts in our own SQLite vs claude-mem's."""
    s = capture.stats()
    s["claude_mem"] = {
        "sessions": len(store.sessions),
        "observations": sum(1 for r in store.records if r["kind"] == "observation"),
        "summaries": sum(1 for r in store.records if r["kind"] == "summary"),
    }
    return s


@app.get("/api/capture/sessions")
def api_capture_sessions(limit: int = 50):
    return {"sessions": capture.list_sessions(limit)}


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
