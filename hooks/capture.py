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
import re
import sys
import time
import urllib.error
import urllib.request

KB_BASE = "http://127.0.0.1:8000/api"
TIMEOUT_S = 1.5

# Cheap path-shape detector for Bash output where we don't know argv structure.
_BASH_PATH_RE = re.compile(
    r"[A-Za-z0-9_./~-]+\.(?:py|ts|tsx|js|jsx|kt|java|kts|md|sql|json|yaml|yml|sh|bash|html|css|toml|ini|cfg|xml|gradle|rs|go|rb)"
)
# Truncation budgets — keep PostToolUse cheap while still preserving signal.
_TRUNC_INPUT = 2048
_TRUNC_RESPONSE_EACH_END = 500


def _truncate(s, n):
    if s is None:
        return None
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


def _head_tail(s, each):
    if s is None:
        return None
    s = str(s)
    if len(s) <= each * 2 + 16:
        return s
    return s[:each] + "\n…[truncated]…\n" + s[-each:]


def _files_from_tool(tool_name, tool_input):
    """Tiny per-tool dispatcher: pull paths out of the tool_input dict.
    Returns (files, kind) where kind ∈ {"read", "edited", "mentioned"}."""
    if not isinstance(tool_input, dict):
        return [], "mentioned"
    name = (tool_name or "").strip()
    if name in ("Read", "NotebookEdit"):
        fp = tool_input.get("file_path") or tool_input.get("notebook_path")
        return ([fp], "read") if fp else ([], "read")
    if name in ("Edit", "Write"):
        fp = tool_input.get("file_path")
        return ([fp], "edited") if fp else ([], "edited")
    if name == "MultiEdit":
        files = [tool_input.get("file_path")] if tool_input.get("file_path") else []
        # MultiEdit also has edits[] with potentially distinct files in newer schemas
        for e in (tool_input.get("edits") or []):
            fp = e.get("file_path") if isinstance(e, dict) else None
            if fp and fp not in files:
                files.append(fp)
        return (files, "edited")
    if name in ("Grep", "Glob"):
        p = tool_input.get("path")
        return ([p], "mentioned") if isinstance(p, str) and p.startswith("/") else ([], "mentioned")
    if name == "Bash":
        cmd = tool_input.get("command") or ""
        paths = list(dict.fromkeys(_BASH_PATH_RE.findall(cmd)))[:8]
        return (paths, "mentioned")
    # MCP and everything else: no files extracted by default.
    return [], "mentioned"

# Convention: worktrees live at <repo>/.claude/worktrees/<branch-or-ticket>/...
# (see ~/.claude/rules/common/patterns.md). Without this normalisation a session
# started inside a worktree would be recorded with project=<worktree-name>,
# producing dozens of bogus "projects" in the KB dropdown.
_WORKTREE_RE = re.compile(r"^(.*?)/\.claude/worktrees/[^/]+(?:/|$)")


def _project_from_cwd(cwd):
    """Return the parent repo's basename. For a worktree path, strip the
    `.claude/worktrees/<name>/...` tail first. For a normal path, just
    `os.path.basename(cwd)`."""
    if not cwd:
        return None
    m = _WORKTREE_RE.match(cwd)
    return os.path.basename(m.group(1) if m else cwd)


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
    project = _project_from_cwd(cwd)

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
    elif event == "PostToolUse":
        # Phase E: capture every tool call into the queue. The background
        # summarizer will roll batches into observations. Hook stays cheap —
        # one POST, payload truncated.
        tool_name = inp.get("tool_name") or ""
        tool_input = inp.get("tool_input") or {}
        tool_response = inp.get("tool_response")
        files, kind = _files_from_tool(tool_name, tool_input)
        try:
            input_json = json.dumps(tool_input, ensure_ascii=False)
        except (TypeError, ValueError):
            input_json = str(tool_input)
        _post("capture/tool", {
            "session_id": session_id,
            "ts": now_ms,
            "tool_name": tool_name,
            "tool_input": _truncate(input_json, _TRUNC_INPUT),
            "tool_response": _head_tail(tool_response, _TRUNC_RESPONSE_EACH_END),
            "files": files,
            "kind": kind,
            "project": project,
            "cwd": cwd,
        })
    elif event == "Stop":
        _post("capture/end", {"session_id": session_id, "ts": now_ms})
        # Phase C: on session end, kick off observation generation. The endpoint
        # is synchronous server-side (one DeepSeek call), but failures are silent
        # and the hook short timeout means even a stalled call can't block Claude.
        _post(f"observe/{session_id}")


if __name__ == "__main__":
    main()
