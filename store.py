"""
store.py — live, in-memory view of claude-mem for the knowledge-base backend.

Reads ~/.claude-mem/claude-mem.db READ-ONLY. Unlike the old static ingest, nothing is
written to disk: the Store loads records into memory and transparently reloads when the
DB grows (ensure_fresh), so the API is always current.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field

DB_PATH = os.path.expanduser("~/.claude-mem/claude-mem.db")


def _connect_ro(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _jarray(raw):
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else [str(v)]
    except (json.JSONDecodeError, TypeError):
        return [raw]


def embed_text(r: dict) -> str:
    """Representative string for a record (used by the semantic indexer)."""
    parts = [r.get("title", ""), r.get("subtitle", ""), r.get("text", ""),
             " ".join(r.get("facts", [])), " ".join(r.get("concepts", []))]
    return "\n".join(p for p in parts if p)[:2000]


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Generic English + KB-noise stopwords. Kept small on purpose — TF-IDF filters the rest.
_STOPWORDS = frozenset("""
the and that for with this from they have been about into your what when where which while
will been them could should would there only also some most just like then than because such
these those well much more then their there here when what whom they have what whose were
this that does into onto over upon under above below than what does did doing will would
session sessions claude task tasks investigation investigate investigated learned completed
""".split())


def _tokens(text: str) -> list[str]:
    if not text:
        return []
    out = []
    for tok in _TOKEN_RE.findall(text):
        if len(tok) < 4:
            continue
        if tok.isdigit():
            continue
        if tok.lower() in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _display_label(token: str) -> str:
    """Code identifiers (have `_` or any uppercase) stay verbatim; prose gets title-case."""
    token = token[:24]
    if "_" in token or any(c.isupper() for c in token):
        return token
    return token[:1].upper() + token[1:].lower()


def _best_token(tf: dict[str, int], df: dict[str, int], n_docs: int) -> tuple[str, float] | None:
    """Highest TF-IDF token in a single document, given corpus document-frequencies."""
    best, best_score = None, -1.0
    for tl, c in tf.items():
        d = df.get(tl, 1)
        score = c * math.log(max(n_docs, 1) / max(d, 1)) if n_docs > d else c * 0.5
        if score > best_score:
            best, best_score = tl, score
    return (best, best_score) if best else None


def _compute_session_labels(sessions: list, records: list) -> dict[str, str]:
    """One distinctive token per session (memId -> label), with fallbacks.

    Primary signal: TF-IDF over the session's observation titles + subtitles. Secondary
    signal: TF-IDF over the session's own title. Tertiary: top non-ubiquitous file
    basename. Last resort: first 8 chars of session id.
    """
    by_sess: dict[str, list] = {}
    for r in records:
        m = r.get("memId")
        if m:
            by_sess.setdefault(m, []).append(r)

    orig_case: dict[str, str] = {}                  # lowered -> first-seen original
    tf_obs: dict[str, dict[str, int]] = {}          # memId -> {token -> count} from obs titles
    for memId, recs in by_sess.items():
        tf: dict[str, int] = {}
        for r in recs:
            for t in _tokens(r.get("title", "")) + _tokens(r.get("subtitle", "")):
                tl = t.lower()
                tf[tl] = tf.get(tl, 0) + 1
                orig_case.setdefault(tl, t)
        if tf:
            tf_obs[memId] = tf

    df_obs: dict[str, int] = {}
    for tf in tf_obs.values():
        for tl in tf:
            df_obs[tl] = df_obs.get(tl, 0) + 1
    n_obs = max(1, len(tf_obs))

    labels: dict[str, str] = {}
    for memId, tf in tf_obs.items():
        pick = _best_token(tf, df_obs, n_obs)
        if pick:
            labels[memId] = _display_label(orig_case[pick[0]])

    # Fallback 1: session titles
    tf_title: dict[str, dict[str, int]] = {}
    for s in sessions:
        if s["memId"] in labels:
            continue
        tf: dict[str, int] = {}
        for t in _tokens(s.get("title", "")):
            tl = t.lower()
            tf[tl] = tf.get(tl, 0) + 1
            orig_case.setdefault(tl, t)
        if tf:
            tf_title[s["memId"]] = tf
    if tf_title:
        df_title: dict[str, int] = {}
        for tf in tf_title.values():
            for tl in tf:
                df_title[tl] = df_title.get(tl, 0) + 1
        n_title = max(1, len(tf_title))
        for memId, tf in tf_title.items():
            pick = _best_token(tf, df_title, n_title)
            if pick:
                labels[memId] = _display_label(orig_case[pick[0]])

    # Fallback 2: top file basename (no extension), filtered by corpus ubiquity
    sess_files: dict[str, list[str]] = {}
    for memId, recs in by_sess.items():
        if memId in labels:
            continue
        bases: list[str] = []
        for r in recs:
            for f in r.get("files", []):
                b = os.path.basename(f).rsplit(".", 1)[0]
                if len(b) >= 4:
                    bases.append(b)
        if bases:
            sess_files[memId] = bases
    if sess_files:
        file_df: dict[str, int] = {}
        for bases in sess_files.values():
            for b in set(bases):
                file_df[b] = file_df.get(b, 0) + 1
        ubiq = max(2, int(len(by_sess) * 0.25))
        for memId, bases in sess_files.items():
            counts: dict[str, int] = {}
            for b in bases:
                if file_df.get(b, 0) <= ubiq:
                    counts[b] = counts.get(b, 0) + 1
            if counts:
                top = max(counts.items(), key=lambda x: x[1])[0]
                labels[memId] = _display_label(top)

    # Last resort: first 8 chars of the content session id
    for s in sessions:
        if s["memId"] not in labels:
            labels[s["memId"]] = (s.get("id") or s["memId"] or "?")[:8]
    return labels


def _blob(r: dict) -> str:
    base = [os.path.basename(f) for f in r.get("files", [])]
    return " \n ".join([r["title"], r.get("subtitle", ""), r.get("text", ""), r["type"],
                        r["project"], " ".join(r.get("facts", [])),
                        " ".join(r.get("concepts", [])), " ".join(base)]).lower()


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

    # ---- loading ----
    def _db_max_epoch(self, con) -> int:
        row = con.execute(
            "SELECT MAX(m) m FROM (SELECT MAX(created_at_epoch) m FROM observations "
            "UNION ALL SELECT MAX(created_at_epoch) FROM session_summaries)"
        ).fetchone()
        return row["m"] or 0

    def reload(self) -> None:
        con = _connect_ro(self.db_path)
        try:
            sessions_by_mem = self._fetch_sessions(con)
            records = self._fetch_observations(con) + self._fetch_summaries(con)
            records.sort(key=lambda r: r["epoch"] or 0, reverse=True)
            for r in records:
                r["_blob"] = _blob(r)
            kept = {r["memId"] for r in records if r["memId"]}
            sessions = [s for s in sessions_by_mem.values() if s["memId"] in kept]
            sessions.sort(key=lambda s: s["started"] or "", reverse=True)
            # one distinctive token per session (sticks on the dict; reused by /api/sessions
            # and /api/graph). Computed here because reload() is where the data set is fixed.
            labels = _compute_session_labels(sessions, records)
            for s in sessions:
                s["label"] = labels.get(s["memId"])
            new_max = self._db_max_epoch(con)
        finally:
            con.close()
        with self._lock:
            self.records = records
            self.sessions = sessions
            self.rec_by_id = {r["id"]: r for r in records}
            self.sess_by_id = {s["id"]: s for s in sessions}
            self.projects = sorted({r["project"] for r in records})
            self.max_epoch = new_max
            self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def ensure_fresh(self) -> bool:
        """Cheap check; reloads only if the DB grew. Returns True if reloaded."""
        con = _connect_ro(self.db_path)
        try:
            cur = self._db_max_epoch(con)
        finally:
            con.close()
        if cur != self.max_epoch or not self.records:
            self.reload()
            return True
        return False

    def _fetch_sessions(self, con) -> dict:
        rows = con.execute("""
            SELECT s.content_session_id, s.memory_session_id, s.project, s.custom_title,
                   s.user_prompt, s.started_at, s.started_at_epoch, s.status,
                   (SELECT COUNT(*) FROM observations o
                      WHERE o.memory_session_id = s.memory_session_id) obs_count
            FROM sdk_sessions s ORDER BY s.started_at_epoch DESC
        """).fetchall()
        out = {}
        for r in rows:
            title = (r["custom_title"] or r["user_prompt"] or "(untitled session)").strip()
            out[r["memory_session_id"]] = {
                "id": r["content_session_id"], "memId": r["memory_session_id"],
                "project": r["project"], "title": title[:120],
                "started": r["started_at"], "status": r["status"], "obsCount": r["obs_count"],
            }
        return out

    def _fetch_observations(self, con) -> list:
        rows = con.execute("""
            SELECT o.id, o.memory_session_id, o.project, o.type, o.title, o.subtitle,
                   o.text, o.facts, o.concepts, o.files_read, o.files_modified,
                   o.created_at, o.created_at_epoch, s.content_session_id
            FROM observations o
            LEFT JOIN sdk_sessions s ON o.memory_session_id = s.memory_session_id
            ORDER BY o.created_at_epoch DESC
        """).fetchall()
        out = []
        for r in rows:
            files = list(dict.fromkeys(_jarray(r["files_read"]) + _jarray(r["files_modified"])))
            out.append({
                "id": f"obs-{r['id']}", "kind": "observation", "type": r["type"] or "discovery",
                "project": r["project"], "title": (r["title"] or "(untitled)").strip(),
                "subtitle": (r["subtitle"] or "").strip(), "text": r["text"] or "",
                "facts": _jarray(r["facts"]), "concepts": _jarray(r["concepts"]),
                "files": files, "sessionId": r["content_session_id"],
                "memId": r["memory_session_id"], "date": r["created_at"],
                "epoch": r["created_at_epoch"],
            })
        return out

    def _fetch_summaries(self, con) -> list:
        rows = con.execute("""
            SELECT ss.id, ss.memory_session_id, ss.project, ss.request, ss.investigated,
                   ss.learned, ss.completed, ss.next_steps, ss.files_read, ss.files_edited,
                   ss.created_at, ss.created_at_epoch, s.content_session_id
            FROM session_summaries ss
            LEFT JOIN sdk_sessions s ON ss.memory_session_id = s.memory_session_id
            ORDER BY ss.created_at_epoch DESC
        """).fetchall()
        out = []
        for r in rows:
            body = "\n\n".join(f"{lbl}: {r[k]}" for lbl, k in (
                ("Request", "request"), ("Investigated", "investigated"), ("Learned", "learned"),
                ("Completed", "completed"), ("Next steps", "next_steps")) if r[k])
            files = list(dict.fromkeys(_jarray(r["files_read"]) + _jarray(r["files_edited"])))
            out.append({
                "id": f"sum-{r['id']}", "kind": "summary", "type": "summary",
                "project": r["project"], "title": (r["request"] or "Session summary").strip()[:120],
                "subtitle": (r["next_steps"] or "").strip()[:160], "text": body,
                "facts": [], "concepts": [], "files": files,
                "sessionId": r["content_session_id"], "memId": r["memory_session_id"],
                "date": r["created_at"], "epoch": r["created_at_epoch"],
            })
        return out

    # ---- queries ----
    def _passes(self, r, project, kind, session) -> bool:
        if project and project != "all" and r["project"] != project:
            return False
        if kind == "observations" and r["kind"] != "observation":
            return False
        if kind == "summaries" and r["kind"] != "summary":
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
        """Session-level overview: top sessions as nodes, linked when they share a
        (reasonably specific) file. Files touched by many sessions are skipped — they
        don't tell you anything. Concepts are ignored here (claude-mem's are generic tags)."""
        sess = [s for s in self.sessions if project in ("all", None) or s["project"] == project]
        sess = sorted(sess, key=lambda s: s["obsCount"], reverse=True)[:max_sessions]
        keep = {s["id"] for s in sess}
        file_sessions: dict[str, set] = {}
        for r in self.records:
            sid = r.get("sessionId")
            if sid in keep:
                for f in r["files"]:
                    file_sessions.setdefault(f, set()).add(sid)
        weight: dict[tuple, int] = {}
        for f, sids in file_sessions.items():
            if 2 <= len(sids) <= max_share:
                s = sorted(sids)
                for i in range(len(s)):
                    for j in range(i + 1, len(s)):
                        weight[(s[i], s[j])] = weight.get((s[i], s[j]), 0) + 1
        links = sorted(
            ({"source": a, "target": b, "weight": w} for (a, b), w in weight.items()),
            key=lambda l: l["weight"], reverse=True,
        )[:max_links]
        nodes = [{"id": s["id"], "title": s["title"], "project": s["project"],
                  "obsCount": s["obsCount"], "label": s.get("label")}
                 for s in sess]
        return {"nodes": nodes, "links": links}

    def meta(self) -> dict:
        return {
            "counts": {
                "observations": sum(1 for r in self.records if r["kind"] == "observation"),
                "summaries": sum(1 for r in self.records if r["kind"] == "summary"),
                "sessions": len(self.sessions),
            },
            "projects": self.projects,
            "loadedAt": self.loaded_at,
            "maxEpoch": self.max_epoch,
        }
