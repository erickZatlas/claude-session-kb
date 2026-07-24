#!/usr/bin/env python3
"""
delete-session.py — remove a Claude Code session so it no longer appears in
the session picker.

Claude Code has no built-in delete: the picker only supports rename, resume,
preview, and search, and a session exists exactly as long as its JSONL
transcript at ~/.claude/projects/<workspace>/<sid>.jsonl does. This is a thin
CLI over session_delete.py, which the KB web backend (app.py) also uses.

Behavior:
  - default: MOVES the transcript (+ any .bak.* siblings) to
    ~/.claude-kb/trash/<workspace>/ so the delete is reversible;
  - --purge: permanently unlinks them instead;
  - --sync-kb: also deletes the session's rows from ~/.claude-kb/data.db.
  - refuses a session that looks live (unless --force).

Usage:
  delete-session.py <session-id-or-prefix> [...] [--purge] [--sync-kb] [--dry-run] [--force]

Examples:
  delete-session.py f3cdc10e --dry-run
  delete-session.py f3cdc10e 4efa538f --sync-kb
  delete-session.py f3cdc10e --purge --sync-kb
"""
from __future__ import annotations

import argparse
import os
import sys

# Import the shared core from the repo root (this script lives in scripts/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import session_delete as sd  # noqa: E402


def _report(result: dict, args) -> None:
    t = result["transcript"]
    if t.get("missing"):
        print("no transcript on disk (already gone); removing KB rows only")
    elif args.purge:
        if args.dry_run:
            print("DRY RUN — would permanently remove:")
            for f in t["would_remove"]:
                print(f"  {f}")
        else:
            print("purged (NOT recoverable):")
            for f in t["removed"]:
                print(f"  {f}")
    else:
        if args.dry_run:
            print(f"DRY RUN — would move to {t['dest_dir']}/:")
            for f in t["would_move"]:
                print(f"  {f}")
        else:
            print("trashed (move the file back to restore):")
            for f in t["moved"]:
                print(f"  {f}")

    kb = result.get("kb")
    if kb is None:
        return
    if kb.get("skipped"):
        print(f"kb sync: skipped ({kb['reason']})")
    else:
        counts = kb.get("deleted") or kb.get("would_delete")
        rows = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        verb = "would delete" if args.dry_run else "deleted"
        print(f"kb sync: {verb} rows in {kb['db']}: {rows or 'none'}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Delete Claude Code sessions by removing their transcripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[0],
    )
    ap.add_argument("session_ids", nargs="+",
                    help="One or more session UUIDs or >=8-char prefixes.")
    ap.add_argument("--purge", action="store_true",
                    help="Permanently delete instead of moving to "
                    "~/.claude-kb/trash/ (the reversible default).")
    ap.add_argument("--sync-kb", action="store_true",
                    help="Also delete the session's rows from ~/.claude-kb/data.db.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen without touching anything.")
    ap.add_argument("--force", action="store_true",
                    help="Override the live-session safety check.")
    args = ap.parse_args()

    rc = 0
    for i, sid in enumerate(args.session_ids):
        if i:
            print()
        try:
            result = sd.delete_session(
                sid, purge=args.purge, sync_kb=args.sync_kb,
                dry_run=args.dry_run, force=args.force,
            )
        except sd.SessionDeleteError as e:
            print(f"error: {e}", file=sys.stderr)
            rc = 1
            continue
        _report(result, args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
