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

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from embeddings import Embedder
from store import Enricher, Store

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

store = Store()
embedder = Embedder()
enricher = Enricher()
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
    _refresh_and_kick()
    return store.session_graph(project)


@app.get("/api/sessions")
def api_sessions(project: str = "all"):
    s = store.sessions if project == "all" else [x for x in store.sessions if x["project"] == project]
    return {"sessions": s}


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
