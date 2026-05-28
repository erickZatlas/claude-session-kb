#!/usr/bin/env python3
"""
Claude Code session-capture hook — Phase B.

One script, dispatched by hook_event_name. Wired in ~/.claude/settings.json
under SessionStart, UserPromptSubmit, and Stop. Captures the session lifecycle
+ prompts into the KB's own SQLite, running in parallel with claude-mem.

Always silent (no stdout, exit 0). Network errors are swallowed so the hook
never blocks Claude Code on a server hiccup.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

KB_BASE = "http://127.0.0.1:8000/api"
TIMEOUT_S = 1.5


def _post(path: str, payload: dict | None = None) -> None:
    """Fire-and-forget POST. Any failure is swallowed (hook stays silent).
    `path` is relative to /api (e.g. "capture/start" or "observe/<sid>")."""
    try:
        body = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            f"{KB_BASE}/{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=TIMEOUT_S).read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        pass


def _read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def main() -> None:
    inp = _read_input()
    event = inp.get("hook_event_name") or ""
    session_id = inp.get("session_id") or ""
    if not session_id:
        sys.exit(0)

    now_ms = int(time.time() * 1000)
    cwd = inp.get("cwd")
    project = os.path.basename(cwd) if cwd else None

    if event == "SessionStart":
        _post("capture/start", {
            "session_id": session_id,
            "project": project,
            "cwd": cwd,
            "started_at": now_ms,
        })
    elif event == "UserPromptSubmit":
        text = (inp.get("prompt") or "").strip()
        if text:
            _post("capture/prompt", {
                "session_id": session_id,
                "text": text,
                "project": project,
                "cwd": cwd,
                "ts": now_ms,
            })
    elif event == "Stop":
        _post("capture/end", {"session_id": session_id, "ts": now_ms})
        # Phase C: on session end, kick off observation generation. The endpoint
        # is synchronous server-side (one DeepSeek call), but failures are silent
        # and the hook short timeout means even a stalled call can't block Claude.
        _post(f"observe/{session_id}")


if __name__ == "__main__":
    main()
