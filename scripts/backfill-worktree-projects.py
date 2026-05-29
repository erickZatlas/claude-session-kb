#!/usr/bin/env python3
"""
backfill-worktree-projects.py — one-shot migration.

Older sessions were captured with `project = basename(cwd)`, so anything
started inside a worktree at `<repo>/.claude/worktrees/<name>/...` got a
bogus project equal to the worktree name (zs-6054-sweeper, agent-abcd…,
etc), polluting the KB's project dropdown.

This script rewrites those rows in place: extracts the parent repo's
basename from cwd and sets sessions.project to it. cwd is left intact so
store.py can derive the worktree label at read time.

Idempotent — a session whose project already matches the parent basename
is skipped. Running twice is a no-op.
"""
from __future__ import annotations

import os
import re
import sqlite3

DB = os.path.expanduser("~/.claude-kb/data.db")
WORKTREE_RE = re.compile(r"^(.*?)/\.claude/worktrees/[^/]+(?:/|$)")


def parent_project(cwd: str | None) -> str | None:
    if not cwd:
        return None
    m = WORKTREE_RE.match(cwd)
    return os.path.basename(m.group(1) if m else cwd)


def main() -> int:
    if not os.path.exists(DB):
        print(f"error: kb db not at {DB}")
        return 1
    con = sqlite3.connect(DB)
    try:
        rows = con.execute(
            "SELECT id, project, cwd FROM sessions "
            "WHERE cwd LIKE '%/.claude/worktrees/%'"
        ).fetchall()
        if not rows:
            print("no worktree-cwd sessions found; nothing to do.")
            return 0
        touched: list[tuple[str, str, str]] = []
        for sid, old_project, cwd in rows:
            new_project = parent_project(cwd)
            if not new_project or new_project == old_project:
                continue
            con.execute(
                "UPDATE sessions SET project = ? WHERE id = ?",
                (new_project, sid),
            )
            touched.append((sid, old_project, new_project))
        con.commit()
        print(f"scanned {len(rows)} sessions with worktree cwd")
        print(f"updated {len(touched)} rows")
        if touched:
            # Sample to make the effect obvious
            print("\nsample changes:")
            for sid, old, new in touched[:8]:
                print(f"  {sid[:8]}…  {old!r:>26}  ->  {new!r}")
            if len(touched) > 8:
                print(f"  …and {len(touched) - 8} more")
        # Summary of resulting projects
        print("\nresulting project distribution:")
        for project, n in con.execute(
            "SELECT project, COUNT(*) FROM sessions "
            "GROUP BY project ORDER BY COUNT(*) DESC LIMIT 12"
        ).fetchall():
            print(f"  {n:4d}  {project}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
