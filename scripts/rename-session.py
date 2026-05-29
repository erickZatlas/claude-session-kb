#!/usr/bin/env python3
"""
rename-session.py — rename a Claude Code session by editing its transcript.

Claude Code stores a session's display name as a `custom-title` record inside
the session's JSONL transcript at ~/.claude/projects/<workspace>/<sid>.jsonl.
The built-in `/rename <name>` slash command can only be typed from inside a
live session. This script does the same edit offline, idempotently, with a
backup, and refuses to touch a transcript that's currently held open by Claude
Code.

Usage:
  rename-session.py <session-id-or-prefix> <new-name> [--sync-kb] [--dry-run]

  --sync-kb   also write the new name to ~/.claude-kb/data.db's sessions.label
              so the KB web UI's card heading matches.
  --dry-run   show the planned edit without writing anything.

Examples:
  rename-session.py f3cdc10e oxi-outage-debrief
  rename-session.py f3cdc10e-8cf0-428a-b25e-329ce6346081 oxi-outage-debrief --sync-kb
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

PROJECTS_DIR = os.path.expanduser("~/.claude/projects")
KB_DB = os.path.expanduser("~/.claude-kb/data.db")
MIN_PREFIX = 8


def find_transcript(sid_or_prefix: str) -> str:
    """Locate the JSONL for an exact UUID or a >=8-char prefix. Errors out
    cleanly on miss / ambiguity rather than silently picking one."""
    pattern = os.path.join(PROJECTS_DIR, "*", f"{sid_or_prefix}*.jsonl")
    if len(sid_or_prefix) < MIN_PREFIX:
        sys.exit(
            f"error: session id (or prefix) must be at least {MIN_PREFIX} characters "
            f"to avoid mis-matching; got {len(sid_or_prefix)} ('{sid_or_prefix}')"
        )
    matches = glob.glob(pattern)
    if not matches:
        sys.exit(f"error: no transcript matches '{sid_or_prefix}' under {PROJECTS_DIR}")
    if len(matches) > 1:
        sys.exit(
            "error: prefix matches multiple transcripts; please use the full UUID:\n  "
            + "\n  ".join(matches)
        )
    return matches[0]


def is_open_by_claude(path: str) -> tuple[bool, str]:
    """Detect whether a session is *live*. Claude Code doesn't hold the
    transcript open between writes (append+close), so lsof returns nothing —
    we instead look for:
      - any `claude` process whose argv mentions this session id (an active
        --resume tab, OR a session that was started fresh with a name);
      - a transcript that was modified in the last 30s (a live writer just
        appended something).
    Either signal trips the refusal. Returns (is_live, reason)."""
    sid = os.path.basename(path)[: -len(".jsonl")]

    # 1. Any claude process holding this session id in its command line
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True, text=True, timeout=4,
        )
        for line in out.stdout.splitlines():
            if "claude" not in line:
                continue
            if sid in line:
                pid = line.strip().split(None, 1)[0]
                return True, f"claude pid {pid} has this session id in its argv"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # No ps — fall through to mtime heuristic; don't hard-refuse
        pass

    # 2. Transcript modified very recently → a writer is active
    try:
        age = time.time() - os.path.getmtime(path)
        if age < 30:
            return True, f"transcript was written {int(age)}s ago (likely live)"
    except OSError:
        pass

    return False, ""


def rewrite_transcript(path: str, new_title: str, dry_run: bool) -> dict:
    """Drop all existing `custom-title` records, append exactly one fresh one.
    Preserves trailing newline. Returns a status dict for the caller to print."""
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    sid = os.path.basename(path)[: -len(".jsonl")]
    kept: list[str] = []
    dropped: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            # Don't choke on the rare malformed line — keep verbatim
            kept.append(line)
            continue
        if obj.get("type") == "custom-title":
            dropped.append(obj.get("customTitle") or "(empty)")
        else:
            kept.append(line)

    new_record = json.dumps(
        {"type": "custom-title", "customTitle": new_title, "sessionId": sid},
        ensure_ascii=False,
    ) + "\n"

    # Ensure the file ends with a newline before appending
    if kept and not kept[-1].endswith("\n"):
        kept[-1] = kept[-1] + "\n"
    kept.append(new_record)

    if dry_run:
        return {
            "path": path,
            "would_drop": dropped,
            "would_set": new_title,
            "lines_in": len(raw_lines),
            "lines_out": len(kept),
            "dry_run": True,
        }

    # Backup with a timestamp; preserves access if anything goes wrong
    backup = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup)

    # Atomic-ish: write to a sibling then rename in place
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(kept)
    os.replace(tmp, path)

    return {
        "path": path,
        "backup": backup,
        "dropped": dropped,
        "set_to": new_title,
        "lines_in": len(raw_lines),
        "lines_out": len(kept),
    }


def sync_kb_label(sid: str, new_title: str) -> dict:
    """Mirror the new title into ~/.claude-kb/data.db's sessions.label so the
    KB web UI card heading matches. No-op when the KB DB isn't present."""
    if not os.path.exists(KB_DB):
        return {"skipped": True, "reason": "no claude-kb db"}
    import sqlite3
    con = sqlite3.connect(KB_DB)
    try:
        cur = con.execute("UPDATE sessions SET label = ? WHERE id = ?", (new_title, sid))
        con.commit()
        return {"updated_rows": cur.rowcount, "db": KB_DB, "label": new_title}
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rename a Claude Code session by editing its transcript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[0],
    )
    ap.add_argument("session_id", help="Session UUID or >=8-char prefix.")
    ap.add_argument("new_name", help="New display name for the session.")
    ap.add_argument("--sync-kb", action="store_true",
                    help="Also update ~/.claude-kb/data.db's sessions.label.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the planned edit without writing anything.")
    ap.add_argument("--force", action="store_true",
                    help="Override the live-session safety check. Useful for "
                    "undoing an immediately-prior rename (which trips the "
                    "mtime guard on our own write).")
    args = ap.parse_args()

    path = find_transcript(args.session_id)
    sid = os.path.basename(path)[: -len(".jsonl")]

    if not args.dry_run and not args.force:
        live, why = is_open_by_claude(path)
        if live:
            sys.exit(
                f"error: {path}\nlooks live ({why}). Close that Claude Code "
                "session first, re-run with --dry-run to preview, or use "
                "--force if you know the safety check is a false positive."
            )

    result = rewrite_transcript(path, args.new_name, args.dry_run)
    if args.dry_run:
        print("DRY RUN — no changes written")
    else:
        print("renamed:")
        if result["dropped"]:
            print(f"  was:    {', '.join(result['dropped'])}")
        print(f"  now:    {args.new_name}")
        print(f"  file:   {result['path']}")
        print(f"  backup: {result['backup']}")

    if args.sync_kb:
        kb = sync_kb_label(sid, args.new_name)
        if kb.get("skipped"):
            print(f"\nkb sync: skipped ({kb['reason']})")
        else:
            print(f"\nkb sync: updated {kb['updated_rows']} row(s) in {kb['db']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
