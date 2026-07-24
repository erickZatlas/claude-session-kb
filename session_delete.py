"""
session_delete.py — core logic for removing a Claude Code session.

Claude Code has no built-in delete: a session exists exactly as long as its
JSONL transcript at ~/.claude/projects/<workspace>/<sid>.jsonl does. Removing
that transcript (and any .bak.* siblings left by rename-session.py) drops it
from the session picker. This module does that safely and reversibly, and can
also delete the session's rows from the claude-kb DB.

Shared by:
  - scripts/delete-session.py  (CLI)
  - app.py                     (DELETE /api/sessions/{sid})
  - tests/test_delete_session.py

The two operations are independent:
  - trash_transcript / purge_transcript  → the on-disk transcript
  - delete_kb_rows                        → the ~/.claude-kb/data.db rows

`delete_session()` is the high-level orchestrator that ties them together.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time

PROJECTS_DIR = os.path.expanduser("~/.claude/projects")
TRASH_DIR = os.path.expanduser("~/.claude-kb/trash")
KB_DB = os.path.expanduser("~/.claude-kb/data.db")
MIN_PREFIX = 8

# Child tables that hang off sessions via ON DELETE CASCADE. Listed explicitly
# so we can report per-table counts; the actual delete is a single DELETE on
# sessions with foreign_keys=ON.
KB_CHILD_TABLES = ("prompts", "observations", "tool_calls", "session_files")


class SessionDeleteError(Exception):
    """Raised for unrecoverable conditions: ambiguous/short prefix, live
    session without force, or nothing to delete. Callers turn this into a CLI
    exit or an HTTP error."""


def find_transcript(sid_or_prefix: str) -> str | None:
    """Locate the JSONL for an exact UUID or a >=8-char prefix.

    Returns the path, or None when nothing matches (the caller decides whether
    a missing transcript is fatal — for the CLI it is; for the API a KB-only
    row is still worth deleting). Raises SessionDeleteError on a too-short
    prefix or an ambiguous match, since silently picking one would be wrong."""
    if len(sid_or_prefix) < MIN_PREFIX:
        raise SessionDeleteError(
            f"session id (or prefix) must be at least {MIN_PREFIX} characters "
            f"to avoid mis-matching; got {len(sid_or_prefix)} ('{sid_or_prefix}')"
        )
    pattern = os.path.join(PROJECTS_DIR, "*", f"{sid_or_prefix}*.jsonl")
    matches = [m for m in glob.glob(pattern) if ".bak." not in os.path.basename(m)]
    if not matches:
        return None
    if len(matches) > 1:
        raise SessionDeleteError(
            "prefix matches multiple transcripts; please use the full UUID:\n  "
            + "\n  ".join(matches)
        )
    return matches[0]


def is_open_by_claude(path: str) -> tuple[bool, str]:
    """Detect whether a session is *live*. Claude Code doesn't hold the
    transcript open between writes (append+close), so lsof returns nothing —
    we instead look for:
      - any `claude` process whose argv mentions this session id;
      - a transcript modified in the last 30s (a live writer just appended).
    Either signal trips the refusal. Returns (is_live, reason).

    Our own process tree is excluded: invoking `delete-session.py <uuid>` (or
    the API server handling a request for <uuid>) puts the sid on a command
    line that also contains 'claude', which would otherwise self-trip."""
    sid = os.path.basename(path)[: -len(".jsonl")]

    own_pids = {str(os.getpid()), str(os.getppid())}
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True, text=True, timeout=4,
        )
        for line in out.stdout.splitlines():
            if "claude" not in line or sid not in line:
                continue
            pid = line.strip().split(None, 1)[0]
            # Skip ourselves, our parent shell, and any delete-session invocation.
            if pid in own_pids or "delete-session.py" in line or "session_delete" in line:
                continue
            return True, f"claude pid {pid} has this session id in its argv"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        age = time.time() - os.path.getmtime(path)
        if age < 30:
            return True, f"transcript was written {int(age)}s ago (likely live)"
    except OSError:
        pass

    return False, ""


def _session_files(path: str) -> list[str]:
    """The transcript plus any .bak.* siblings rename-session.py left behind."""
    return [path] + sorted(glob.glob(f"{path}.bak.*"))


def _unique_dest(dest: str) -> str:
    """Never overwrite an earlier trashed copy of the same session."""
    if not os.path.exists(dest):
        return dest
    n = 1
    while os.path.exists(f"{dest}.{n}"):
        n += 1
    return f"{dest}.{n}"


def trash_transcript(path: str, dry_run: bool = False) -> dict:
    """Move the transcript (+ baks) into TRASH_DIR/<workspace>/, reversibly.
    Restoring is just moving the file back to its projects/<workspace>/ dir."""
    files = _session_files(path)
    workspace = os.path.basename(os.path.dirname(path))
    dest_dir = os.path.join(TRASH_DIR, workspace)

    if dry_run:
        return {"dry_run": True, "would_move": files, "dest_dir": dest_dir}

    os.makedirs(dest_dir, exist_ok=True)
    moved = []
    for f in files:
        dest = _unique_dest(os.path.join(dest_dir, os.path.basename(f)))
        shutil.move(f, dest)
        moved.append(dest)
    return {"moved": moved, "dest_dir": dest_dir}


def purge_transcript(path: str, dry_run: bool = False) -> dict:
    """Permanently unlink the transcript (+ baks). Not reversible."""
    files = _session_files(path)
    if dry_run:
        return {"dry_run": True, "would_remove": files}
    for f in files:
        os.remove(f)
    return {"removed": files}


def delete_kb_rows(sid: str, dry_run: bool = False) -> dict:
    """Delete the session's rows from ~/.claude-kb/data.db. Child tables go via
    ON DELETE CASCADE (foreign_keys must be ON per-connection). The embedding
    cache (.cache/vectors.npy) drops the dead ids on its next compact()."""
    if not os.path.exists(KB_DB):
        return {"skipped": True, "reason": "no claude-kb db"}
    import sqlite3
    con = sqlite3.connect(KB_DB)
    try:
        con.execute("PRAGMA foreign_keys=ON;")
        counts = {
            t: con.execute(
                f"SELECT COUNT(*) FROM {t} WHERE session_id = ?", (sid,)
            ).fetchone()[0]
            for t in KB_CHILD_TABLES
        }
        counts["sessions"] = con.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?", (sid,)
        ).fetchone()[0]
        if dry_run:
            return {"dry_run": True, "would_delete": counts, "db": KB_DB}
        con.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        con.commit()
        return {"deleted": counts, "db": KB_DB}
    finally:
        con.close()


def delete_session(
    sid_or_prefix: str,
    *,
    purge: bool = False,
    sync_kb: bool = True,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Orchestrate a full session delete: transcript + (optionally) KB rows.

    Returns a result dict:
      {sid, transcript: <trash/purge result | {"missing": True}>, kb: <delete_kb_rows result | None>}

    Raises SessionDeleteError on ambiguous/short prefix, a live session (unless
    force), or when there is genuinely nothing to delete (no transcript and
    sync_kb is off)."""
    path = find_transcript(sid_or_prefix)

    if path is None:
        # No transcript. Only meaningful if we're still removing KB rows and the
        # caller gave a full id we can use directly.
        if not sync_kb:
            raise SessionDeleteError(
                f"no transcript matches '{sid_or_prefix}' under {PROJECTS_DIR}"
            )
        sid = sid_or_prefix
        kb = delete_kb_rows(sid, dry_run)
        return {"sid": sid, "transcript": {"missing": True}, "kb": kb}

    sid = os.path.basename(path)[: -len(".jsonl")]

    if not dry_run and not force:
        live, why = is_open_by_claude(path)
        if live:
            raise SessionDeleteError(
                f"{path} looks live ({why}). Close that session first, preview "
                "with dry_run, or pass force=True to override."
            )

    transcript = (
        purge_transcript(path, dry_run) if purge else trash_transcript(path, dry_run)
    )
    kb = delete_kb_rows(sid, dry_run) if sync_kb else None
    return {"sid": sid, "transcript": transcript, "kb": kb}
