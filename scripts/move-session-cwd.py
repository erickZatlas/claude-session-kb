#!/usr/bin/env python3
"""
move-session-cwd.py — move a Claude Code session transcript to a different
project directory so `claude --resume <sid>` works from a new cwd.

Claude Code stores transcripts at ~/.claude/projects/<encoded-cwd>/<sid>.jsonl
where <encoded-cwd> is the absolute path with `/` replaced by `-`. When you
launch claude from cwd X, it only sees sessions whose transcripts live in
the encoded-X directory.

Use this when you started a session in directory A (e.g. you launched claude
from a worktree, or from the wrong folder) but want to resume it from
directory B going forward. The transcript file is moved; the in-memory KB
record's `cwd` + `project` are optionally updated to match.

SAFETY: refuses to move a live session (one whose UUID appears in a claude
process's argv, or whose transcript was written within the last 30s).
Override with --force if you know the safety check is a false positive.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time

PROJECTS = os.path.expanduser("~/.claude/projects")
KB_DB = os.path.expanduser("~/.claude-kb/data.db")


def encode_cwd(cwd: str) -> str:
    """Path → Claude's project-dir name. /home/x/y -> -home-x-y."""
    abs_path = os.path.abspath(os.path.expanduser(cwd))
    return abs_path.replace("/", "-")


def find_transcript(sid_or_prefix: str) -> str:
    """Locate the JSONL for a session by exact UUID or >=8-char prefix.
    Returns the absolute path or raises SystemExit."""
    if len(sid_or_prefix) < 8:
        sys.exit(
            f"error: session id '{sid_or_prefix}' must be at least 8 characters"
        )
    matches: list[str] = []
    for proj in os.listdir(PROJECTS):
        d = os.path.join(PROJECTS, proj)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".jsonl") and fn.startswith(sid_or_prefix):
                matches.append(os.path.join(d, fn))
    if not matches:
        sys.exit(f"error: no transcript matches '{sid_or_prefix}'")
    if len(matches) > 1:
        sys.exit(
            f"error: prefix '{sid_or_prefix}' matches multiple transcripts:\n  "
            + "\n  ".join(matches)
        )
    return matches[0]


def is_live(path: str) -> tuple[bool, str]:
    """True if the session looks live. Same heuristic as rename-session.py:
    (a) any `claude` process has the session id in its argv, or
    (b) the transcript was modified within the last 30 s."""
    sid = os.path.basename(path)[: -len(".jsonl")]
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True, text=True, timeout=4,
        )
        for line in out.stdout.splitlines():
            if "claude" in line and sid in line:
                pid = line.strip().split(None, 1)[0]
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


def sync_kb(sid: str, new_cwd: str, new_project: str | None) -> int:
    """Update the kb DB to reflect the new cwd (and project, if given)."""
    if not os.path.exists(KB_DB):
        return 0
    con = sqlite3.connect(KB_DB)
    try:
        if new_project:
            cur = con.execute(
                "UPDATE sessions SET cwd = ?, project = ? WHERE id = ?",
                (new_cwd, new_project, sid),
            )
        else:
            cur = con.execute(
                "UPDATE sessions SET cwd = ? WHERE id = ?",
                (new_cwd, sid),
            )
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("session", help="Session UUID or >=8-char prefix.")
    ap.add_argument(
        "new_cwd",
        help="Absolute path of the cwd you want to resume this session from "
        "(e.g. /home/erick/dev/claude-kb).",
    )
    ap.add_argument(
        "--sync-kb", action="store_true",
        help="Also update the claude-session-kb sessions.cwd row. Pair with "
             "--project to also rewrite sessions.project.",
    )
    ap.add_argument(
        "--project",
        help="When --sync-kb is set, also rewrite sessions.project to this "
             "value. If omitted, the project field is left alone (the KB "
             "will derive a worktree label from the new cwd at read time).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show the planned move without performing it.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Override the live-session safety check.",
    )
    args = ap.parse_args()

    old_path = find_transcript(args.session)
    sid = os.path.basename(old_path)[: -len(".jsonl")]
    new_dir = os.path.join(PROJECTS, encode_cwd(args.new_cwd))
    new_path = os.path.join(new_dir, sid + ".jsonl")

    print(f"  session:  {sid}")
    print(f"  from:     {old_path}")
    print(f"  to:       {new_path}")

    if old_path == new_path:
        sys.exit("error: source and destination are identical — nothing to do.")
    if os.path.exists(new_path):
        sys.exit(
            f"error: destination already exists. Two sessions with the same "
            f"UUID would be a collision:\n  {new_path}"
        )

    if not args.dry_run and not args.force:
        live, why = is_live(old_path)
        if live:
            sys.exit(
                f"error: {old_path}\nlooks live ({why}). /exit the Claude "
                f"session first, or re-run with --dry-run / --force."
            )

    if args.dry_run:
        print("\nDRY RUN — no changes written.")
        return

    os.makedirs(new_dir, exist_ok=True)
    shutil.move(old_path, new_path)
    print("\nmoved ✓")
    print(f"  next:    cd {args.new_cwd} && claude --resume {sid}")

    if args.sync_kb:
        n = sync_kb(sid, os.path.abspath(args.new_cwd), args.project)
        proj_msg = f", project='{args.project}'" if args.project else ""
        print(f"kb sync: updated {n} row(s) (cwd{proj_msg})")


if __name__ == "__main__":
    main()
