#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit hook — pre-emptive recall.

For every user prompt, query the local KB (/api/recall) for the most
semantically-similar past sessions and inject a short Markdown block of the
top matches as additional context. Claude sees "you've worked on this before"
without you having to ask.

Failure modes are all silent (timeout, KB down, no matches above threshold,
prompt too short to be meaningful) — the hook never blocks or noises up the
chat. If the KB isn't running, nothing happens.

Register in ~/.claude/settings.json under hooks.UserPromptSubmit.
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

KB_URL = "http://127.0.0.1:8000/api/recall"
LIMIT = 3                   # how many sessions to surface, max
MIN_PROMPT_LEN = 16         # shorter prompts aren't meaningful enough to recall on
MIN_SCORE = 0.32            # cosine threshold — below this it's noise
TIMEOUT_S = 2.0             # never block the user for more than this


def _silent_exit() -> None:
    sys.exit(0)


def _read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _fetch(prompt: str, exclude: str) -> list[dict]:
    qs = urllib.parse.urlencode({"q": prompt, "limit": LIMIT, "exclude": exclude})
    try:
        with urllib.request.urlopen(f"{KB_URL}?{qs}", timeout=TIMEOUT_S) as r:
            data = json.load(r)
    except Exception:
        return []
    return [s for s in (data.get("sessions") or []) if (s.get("score") or 0) >= MIN_SCORE]


def _render(hits: list[dict]) -> str:
    lines = [
        "### Related past sessions (auto-surfaced by claude-session-kb)",
        "",
        "_These are semantically similar sessions from your own history. Use them as context, ignore if not relevant._",
        "",
    ]
    for s in hits:
        label = s.get("label") or "(unlabeled)"
        proj = s.get("project") or "?"
        obs = s.get("obsCount", 0)
        started = (s.get("started") or "")[:10]
        score = s.get("score", 0)
        summary = s.get("summary") or s.get("matchedTitle") or s.get("title") or ""
        summary = (summary[:280] + "…") if len(summary) > 280 else summary
        lines.append(f"- **{label}** · `{proj}` · {obs} obs · {started} · score {score:.2f}")
        if summary:
            lines.append(f"  - {summary}")
        if s.get("id"):
            lines.append(f"  - To revisit: `claude --resume {s['id']}`")
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    inp = _read_input()
    prompt = (inp.get("prompt") or "").strip()
    this_session = inp.get("session_id") or ""
    if len(prompt) < MIN_PROMPT_LEN:
        _silent_exit()
    hits = _fetch(prompt, this_session)
    if not hits:
        _silent_exit()
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": _render(hits),
        }
    }))


if __name__ == "__main__":
    main()
