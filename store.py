"""
store.py — live, in-memory view of the KB for the web backend.

Reads ~/.claude-kb/data.db (our own store, populated by Phase B capture hooks +
Phase D imports). Was previously backed by ~/.claude-mem/claude-mem.db; that
dependency was retired now that all 5,464 historical observations and their
sessions/labels/summaries live in our own SQLite.

The Store loads everything into memory on `reload()`, refreshes when the DB
grows (`ensure_fresh()`), and projects our (sessions, observations) schema into
the legacy record shape the rest of the app already speaks (kind, memId,
sessionId, files, concepts, etc.). This keeps the embedder/search/graph code
untouched.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field

import llm

DB_PATH = os.path.expanduser("~/.claude-kb/data.db")

# Worktree convention: <repo>/.claude/worktrees/<branch-or-ticket>/...
# (capture.py rewrites `project` to the parent repo at write time; this regex
# pulls the worktree NAME back out of cwd so the UI can label it.)
_WORKTREE_RE = re.compile(r"/\.claude/worktrees/([^/]+)")


def _worktree_from_cwd(cwd):
    """Return the worktree directory name when cwd points inside one, else None."""
    if not cwd:
        return None
    m = _WORKTREE_RE.search(cwd)
    return m.group(1) if m else None


def _connect_ro(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _jarray(raw):
    """Tags column is JSON text; tolerate strings, null, or already-parsed lists."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else [str(v)]
    except (json.JSONDecodeError, TypeError):
        return [raw]


def embed_text(r: dict) -> str:
    """Representative string for a record (used by the semantic indexer).
    Concepts here = our topical tags (not claude-mem's generic ones)."""
    parts = [r.get("title", ""), r.get("subtitle", ""), r.get("text", ""),
             " ".join(r.get("concepts", []))]
    return "\n".join(p for p in parts if p)[:2000]


def _llm_payload(session: dict, records: list) -> str:
    """Compact representation of a session for the enricher: prompt + obs titles."""
    parts: list[str] = []
    t = (session.get("title") or "").strip()
    if t:
        parts.append(f"Prompt: {t[:200]}")
    seen = 0
    for r in records:
        title = (r.get("title") or "").strip()
        if not title:
            continue
        parts.append(f"- {title}")
        seen += 1
        if seen >= 30:
            break
    return "\n".join(parts)[:2500]


class Enricher:
    """Background LLM pass that attaches a kebab label + 1–2 sentence summary
    to sessions that don't have them yet. Phase D already labeled+summarized
    every imported session, so in practice this only runs for live captures
    that finish before observe.py reaches them. Fire-and-forget, disk-cached."""

    def __init__(self):
        self._lock = threading.Lock()
        self.running = False
        self.total = 0
        self.done = 0

    def sync(self, sessions: list, records: list) -> None:
        if not llm.is_enabled():
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self.running = True
            by_sess: dict[str, list] = {}
            for r in records:
                m = r.get("sessionId")
                if m:
                    by_sess.setdefault(m, []).append(r)
            # Only enrich sessions that lack a stored label OR summary
            todo: list[tuple[dict, str]] = []
            for s in sessions:
                if s.get("label") and s.get("summary"):
                    continue
                payload = _llm_payload(s, by_sess.get(s["id"], []))
                if payload:
                    todo.append((s, payload))
            self.total, self.done = len(todo), 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(self._one, s, p) for s, p in todo]
                for _ in concurrent.futures.as_completed(futs):
                    self.done += 1
        finally:
            self.running = False
            self._lock.release()

    def _one(self, session: dict, payload: str) -> None:
        if not session.get("label"):
            lab = llm.label_for(payload)
            if lab:
                session["label"] = lab[:40]
        if not session.get("summary"):
            summ = llm.summary_for(payload)
            if summ:
                session["summary"] = summ


def _blob(r: dict) -> str:
    """Lowercased search blob — what keyword_search greps against."""
    return " \n ".join([r["title"], r.get("subtitle", ""), r.get("text", ""),
                        r["type"], r["project"],
                        " ".join(r.get("concepts", []))]).lower()


@dataclass
class Store:
    db_path: str = DB_PATH
    records: list = field(default_factory=list)
    sessions: list = field(default_factory=list)
    rec_by_id: dict = field(default_factory=dict)
    sess_by_id: dict = field(default_factory=dict)
    projects: list = field(default_factory=list)
    max_epoch: int = 0
    loaded_at: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _db_max_epoch(self, con) -> int:
        # Include lessons + memory_facts so a distill or a disk-synced memory
        # edit (which touch neither observations nor sessions) still trips
        # ensure_fresh() into reloading + re-embedding the corpus. Each source
        # is queried defensively so a DB missing a newer table still works.
        best = 0
        for sql in (
            "SELECT MAX(created_at) m FROM observations",
            "SELECT MAX(started_at) m FROM sessions",
            "SELECT MAX(created_at) m FROM lessons",
            "SELECT MAX(updated_at) m FROM memory_facts",
        ):
            try:
                row = con.execute(sql).fetchone()
                best = max(best, int((row["m"] if row else 0) or 0))
            except sqlite3.Error:
                continue
        return best

    def reload(self) -> None:
        if not os.path.exists(self.db_path):
            with self._lock:
                self.records, self.sessions = [], []
                self.rec_by_id, self.sess_by_id = {}, {}
                self.projects, self.max_epoch = [], 0
                self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            return
        con = _connect_ro(self.db_path)
        try:
            sessions = self._fetch_sessions(con)
            records = self._fetch_observations(con)
            # Durable cross-session knowledge joins the same corpus, so the
            # embedder/search/recall path treats lessons + auto-memory facts as
            # first-class records (kind 'lesson' / 'memory', no session attached).
            records.extend(self._fetch_lessons(con))
            records.extend(self._fetch_memory_facts(con))
            records.sort(key=lambda r: r["epoch"] or 0, reverse=True)
            for r in records:
                r["_blob"] = _blob(r)
            # Attach per-session observation counts onto the session dicts
            obs_count: dict[str, int] = {}
            for r in records:
                sid = r.get("sessionId")
                if sid:
                    obs_count[sid] = obs_count.get(sid, 0) + 1
            for s in sessions:
                s["obsCount"] = obs_count.get(s["id"], 0)
            new_max = self._db_max_epoch(con)
        finally:
            con.close()
        with self._lock:
            self.records = records
            self.sessions = sessions
            self.rec_by_id = {r["id"]: r for r in records}
            self.sess_by_id = {s["id"]: s for s in sessions}
            self.projects = sorted({r["project"] for r in records if r.get("project")})
            self.max_epoch = new_max
            self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def ensure_fresh(self) -> bool:
        """Reloads only if the DB grew. Returns True if reloaded. Tolerates a
        missing DB file (returns False without raising) so the read endpoints
        keep working on the last-loaded in-memory snapshot."""
        if not os.path.exists(self.db_path):
            return False
        try:
            con = _connect_ro(self.db_path)
            try:
                cur = self._db_max_epoch(con)
            finally:
                con.close()
        except sqlite3.Error:
            return False
        if cur != self.max_epoch or not self.records:
            self.reload()
            return True
        return False

    def _fetch_sessions(self, con) -> list:
        """Map our sessions table -> the legacy session-dict shape.
        memId == id since we don't keep a separate memory-session id.

        Phantom filter: `claude --resume` makes Claude Code mint a throwaway
        session id during the resume handshake and fire SessionStart against
        it before switching to the real resumed id. That leaves an orphan row
        with no prompts, observations, or tool calls (and the enricher then
        gives it a hallucinated summary). We skip any session that has none of
        the three — it's either that phantom or a just-born session with
        nothing to show yet. Self-correcting: the store reloads when the DB
        grows, so a real session reappears the instant it records anything."""
        rows = con.execute(
            "SELECT s.id, s.project, s.cwd, s.started_at, s.ended_at, s.status, "
            "       s.first_prompt, s.prompt_count, s.label, s.summary, "
            "       (SELECT COUNT(DISTINCT path) FROM session_files sf "
            "        WHERE sf.session_id = s.id) AS files_count "
            "FROM sessions s "
            "WHERE s.prompt_count > 0 "
            "   OR EXISTS (SELECT 1 FROM observations o WHERE o.session_id = s.id) "
            "   OR EXISTS (SELECT 1 FROM tool_calls t WHERE t.session_id = s.id) "
            "ORDER BY s.started_at DESC"
        ).fetchall()
        out = []
        for r in rows:
            title = (r["first_prompt"] or "(untitled session)").strip()
            started_iso = ""
            if r["started_at"]:
                # started_at is epoch-ms (Phase B) or epoch-seconds (Phase D);
                # heuristic split at year 3000 in seconds
                ts = int(r["started_at"])
                if ts > 10_000_000_000:  # > year 2286 in seconds → must be ms
                    ts //= 1000
                started_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts))
            out.append({
                "id": r["id"], "memId": r["id"],
                "project": r["project"], "title": title[:120],
                "started": started_iso, "status": r["status"] or "completed",
                "obsCount": 0,                            # filled by reload()
                "label": r["label"], "summary": r["summary"],
                "cwd": r["cwd"],
                "worktree": _worktree_from_cwd(r["cwd"]),
                "filesCount": int(r["files_count"] or 0),
            })
        return out

    def _fetch_observations(self, con) -> list:
        """Map our observations table -> the legacy record-dict shape.
        Our `tags` column (topical) takes the place of claude-mem's `concepts`
        (generic) — same field name on the way out so the embedder/search code
        doesn't need to change."""
        rows = con.execute(
            "SELECT o.id, o.session_id, o.ord, o.type, o.title, o.text, o.tags, "
            "       o.created_at, s.project "
            "FROM observations o LEFT JOIN sessions s ON s.id = o.session_id "
            "ORDER BY o.created_at DESC"
        ).fetchall()
        out = []
        for r in rows:
            ts = int(r["created_at"] or 0)
            if ts > 10_000_000_000:
                ts //= 1000  # normalize ms -> s for ISO formatting
            date_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) if ts else ""
            out.append({
                "id": f"obs-{r['id']}", "kind": "observation",
                "type": r["type"] or "discovery",
                "project": r["project"] or "",
                "title": (r["title"] or "(untitled)").strip(),
                "subtitle": "",                      # we don't store subtitles
                "text": r["text"] or "",
                "facts": [],                         # vestigial: claude-mem-only
                "concepts": _jarray(r["tags"]),      # OUR tags occupy this slot
                "files": [],                         # we don't track touched files
                "sessionId": r["session_id"], "memId": r["session_id"],
                "date": date_iso, "epoch": int(r["created_at"] or 0),
            })
        return out

    def _fetch_lessons(self, con) -> list:
        """Project the lessons table into the shared record shape. kind/type =
        'lesson'; no session attaches. id is 'lesson-<rowid>' — replace_lessons
        wipes+reinserts with fresh rowids, so an updated lesson re-embeds."""
        try:
            rows = con.execute(
                "SELECT id, title, text, tags, evidence_count, last_seen "
                "FROM lessons ORDER BY last_seen DESC"
            ).fetchall()
        except sqlite3.Error:
            return []
        out = []
        for r in rows:
            ts = int(r["last_seen"] or 0)
            if ts > 10_000_000_000:
                ts //= 1000
            date_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) if ts else ""
            out.append({
                "id": f"lesson-{r['id']}", "kind": "lesson", "type": "lesson",
                "project": "",
                "title": (r["title"] or "(untitled lesson)").strip(),
                "subtitle": "",
                "text": r["text"] or "",
                "facts": [],
                "concepts": _jarray(r["tags"]),
                "files": [],
                "sessionId": None, "memId": None,
                "date": date_iso, "epoch": int(r["last_seen"] or 0),
                "evidenceCount": int(r["evidence_count"] or 0),
            })
        return out

    def _fetch_memory_facts(self, con) -> list:
        """Project memory_facts (synced from ~/.claude/projects/*/memory/*.md by
        memory_ingest) into the record shape. kind/type = 'memory'. The id is
        content-hashed so an edited fact embeds as a fresh vector; the old one
        is left orphaned in the matrix (never referenced via rec_by_id)."""
        try:
            rows = con.execute(
                "SELECT content_hash, project, source_path, name, mem_type, "
                "       description, text, tags, updated_at "
                "FROM memory_facts ORDER BY updated_at DESC"
            ).fetchall()
        except sqlite3.Error:
            return []
        out = []
        for r in rows:
            ts = int(r["updated_at"] or 0)
            if ts > 10_000_000_000:
                ts //= 1000
            date_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) if ts else ""
            tags = _jarray(r["tags"])
            mem_type = r["mem_type"] or "reference"
            concepts = tags + ([mem_type] if mem_type not in tags else [])
            out.append({
                "id": f"mem-{r['content_hash']}", "kind": "memory", "type": "memory",
                "project": r["project"] or "",
                "title": (r["name"] or "(memory)").strip(),
                "subtitle": (r["description"] or "").strip(),
                "text": r["text"] or "",
                "facts": [],
                "concepts": concepts,
                "files": [],
                "sessionId": None, "memId": None,
                "date": date_iso, "epoch": int(r["updated_at"] or 0),
                "memType": mem_type,
                "sourcePath": r["source_path"] or "",
            })
        return out

    def knowledge_records(self) -> list:
        """Lesson + memory records — the durable cross-session knowledge that
        /api/recall surfaces in its own block (no time decay)."""
        return [r for r in self.records if r.get("kind") in ("lesson", "memory")]

    def _passes(self, r, project, kind, session) -> bool:
        if project and project != "all" and r["project"] != project:
            return False
        if kind == "observations" and r["kind"] != "observation":
            return False
        if kind == "summaries" and r["kind"] != "summary":
            # We have no separate summary records — they live as a column on
            # sessions. The summaries filter therefore returns empty by design.
            return False
        if session and r.get("sessionId") != session:
            return False
        return True

    def candidates(self, project, kind, session=None) -> list:
        return [r for r in self.records if self._passes(r, project, kind, session)]

    def keyword_search(self, query, project="all", kind="all", limit=250, session=None) -> list:
        recs = self.candidates(project, kind, session)
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return recs[:limit]
        scored = []
        for r in recs:
            blob, title = r["_blob"], r["title"].lower()
            score, ok = 0, True
            for t in terms:
                if t not in blob:
                    ok = False
                    break
                score += 5 if t in title else 1
            if ok:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def session_graph(self, project="all", max_sessions=40, max_share=8, max_links=80) -> dict:
        """Session-level overview: top sessions as nodes, linked by shared
        tags. We no longer track per-record file lists (claude-mem did), so the
        graph links on shared topical tags instead of shared files. Same
        intuition: connect sessions that worked on similar things."""
        sess = [s for s in self.sessions if project in ("all", None) or s["project"] == project]
        sess = sorted(sess, key=lambda s: s.get("obsCount") or 0, reverse=True)[:max_sessions]
        keep = {s["id"] for s in sess}
        tag_sessions: dict[str, set] = {}
        for r in self.records:
            sid = r.get("sessionId")
            if sid in keep:
                for t in r["concepts"]:
                    tag_sessions.setdefault(t, set()).add(sid)
        weight: dict[tuple, int] = {}
        for _, sids in tag_sessions.items():
            if 2 <= len(sids) <= max_share:
                ordered = sorted(sids)
                for i in range(len(ordered)):
                    for j in range(i + 1, len(ordered)):
                        weight[(ordered[i], ordered[j])] = weight.get((ordered[i], ordered[j]), 0) + 1
        links = sorted(
            ({"source": a, "target": b, "weight": w} for (a, b), w in weight.items()),
            key=lambda link: link["weight"], reverse=True,
        )[:max_links]
        nodes = [{"id": s["id"], "title": s["title"], "project": s["project"],
                  "obsCount": s.get("obsCount") or 0, "label": s.get("label")}
                 for s in sess]
        return {"nodes": nodes, "links": links}

    def meta(self) -> dict:
        return {
            "counts": {
                "observations": sum(1 for r in self.records if r["kind"] == "observation"),
                "lessons": sum(1 for r in self.records if r["kind"] == "lesson"),
                "memoryFacts": sum(1 for r in self.records if r["kind"] == "memory"),
                "summaries": sum(1 for s in self.sessions if s.get("summary")),
                "sessions": len(self.sessions),
            },
            "projects": self.projects,
            "loadedAt": self.loaded_at,
            "maxEpoch": self.max_epoch,
        }
