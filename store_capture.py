"""
store_capture.py — our own session capture store. Lives alongside claude-mem
during the dual-write period (Phase B); becomes the source of truth in Phase D.

Schema:
  sessions(id PK, project, cwd, started_at, ended_at, status, first_prompt, prompt_count)
  prompts(id PK, session_id FK, ts, text)

The DB is at ~/.claude-kb/data.db. Schema is created on first use.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Optional

DATA_DIR = os.path.expanduser("~/.claude-kb")
DB_PATH = os.path.join(DATA_DIR, "data.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id           TEXT PRIMARY KEY,
  project      TEXT,
  cwd          TEXT,
  started_at   INTEGER NOT NULL,
  ended_at     INTEGER,
  status       TEXT DEFAULT 'active',
  first_prompt TEXT,
  prompt_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);

CREATE TABLE IF NOT EXISTS prompts (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  ts         INTEGER NOT NULL,
  text       TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_prompts_session ON prompts(session_id, ts);
"""


class CaptureStore:
    """Thread-safe SQLite-backed capture store. One process, many writer threads."""

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA foreign_keys=ON;")
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    def session_start(self, session_id: str, project: Optional[str],
                      cwd: Optional[str], started_at: Optional[int] = None) -> None:
        """Idempotent: SessionStart fires for resumes too; we INSERT OR IGNORE."""
        ts = started_at or int(time.time() * 1000)
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO sessions (id, project, cwd, started_at, status) "
                "VALUES (?, ?, ?, ?, 'active')",
                (session_id, project, cwd, ts),
            )

    def record_prompt(self, session_id: str, text: str,
                      project: Optional[str] = None, cwd: Optional[str] = None,
                      ts: Optional[int] = None) -> None:
        """Append a prompt to its session. Auto-creates the session if missing
        (e.g. UserPromptSubmit arrives before SessionStart on a resumed session)
        and stamps first_prompt the first time we see one."""
        ts = ts or int(time.time() * 1000)
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO sessions (id, project, cwd, started_at, status) "
                "VALUES (?, ?, ?, ?, 'active')",
                (session_id, project, cwd, ts),
            )
            c.execute(
                "INSERT INTO prompts (session_id, ts, text) VALUES (?, ?, ?)",
                (session_id, ts, text),
            )
            c.execute(
                "UPDATE sessions SET "
                "  first_prompt = COALESCE(first_prompt, ?), "
                "  prompt_count = prompt_count + 1 "
                "WHERE id = ?",
                (text, session_id),
            )

    def session_end(self, session_id: str, ts: Optional[int] = None) -> None:
        ts = ts or int(time.time() * 1000)
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE sessions SET ended_at = ?, status = 'completed' WHERE id = ?",
                (ts, session_id),
            )

    # --- read helpers (used by /api/capture endpoints) ---
    def stats(self) -> dict:
        with self._conn() as c:
            n_sess = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            n_prompts = c.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
            n_active = c.execute("SELECT COUNT(*) FROM sessions WHERE status='active'").fetchone()[0]
            latest = c.execute(
                "SELECT id, project, started_at, first_prompt FROM sessions "
                "ORDER BY started_at DESC LIMIT 5"
            ).fetchall()
        return {
            "db_path": self.db_path,
            "sessions": n_sess,
            "prompts": n_prompts,
            "active": n_active,
            "latest": [dict(r) for r in latest],
        }

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, project, cwd, started_at, ended_at, status, "
                "       first_prompt, prompt_count "
                "FROM sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
