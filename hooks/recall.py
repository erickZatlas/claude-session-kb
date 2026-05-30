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
LESSON_LIMIT = 4            # how many durable lessons/memory facts to surface, max
MIN_PROMPT_LEN = 16         # shorter prompts aren't meaningful enough to recall on
MIN_SCORE = 0.32            # session cosine threshold — below this it's noise
LESSON_MIN_SCORE = 0.30     # lessons/memory are cheap + high-value → looser floor
TIMEOUT_S = 2.0             # never block the user for more than this


def _silent_exit() -> None:
    sys.exit(0)


def _read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _fetch(prompt: str, exclude: str) -> tuple[list[dict], list[dict]]:
    qs = urllib.parse.urlencode({"q": prompt, "limit": LIMIT, "exclude": exclude})
    try:
        with urllib.request.urlopen(f"{KB_URL}?{qs}", timeout=TIMEOUT_S) as r:
            data = json.load(r)
    except Exception:
        return [], []
    sessions = [s for s in (data.get("sessions") or [])
                if (s.get("score") or 0) >= MIN_SCORE]
    lessons = [l for l in (data.get("lessons") or [])
               if (l.get("score") or 0) >= LESSON_MIN_SCORE][:LESSON_LIMIT]
    return sessions, lessons


def _render_lessons(lessons: list[dict]) -> list[str]:
    """Durable cross-session lessons + hand-authored auto-memory. Rendered above
    the sessions block — densest, most actionable knowledge first."""
    if not lessons:
        return []
    lines = [
        "### Relevant lessons & memory (auto-surfaced by claude-session-kb)",
        "",
        "_Durable cross-session learnings + your hand-authored auto-memory. Treat as background knowledge, ignore if not relevant._",
        "",
    ]
    for l in lessons:
        title = l.get("title") or "(untitled)"
        score = l.get("score", 0)
        if l.get("kind") == "memory":
            origin = f"memory: {l.get('memType') or 'note'}"
        else:
            ev = l.get("evidence") or 0
            origin = f"lesson · {ev} sessions" if ev else "lesson"
        text = (l.get("text") or "").strip()
        text = (text[:280] + "…") if len(text) > 280 else text
        lines.append(f"- **{title}** · _{origin}_ · score {score:.2f}")
        if text:
            lines.append(f"  - {text}")
        lines.append("")
    return lines


def _render_sessions(hits: list[dict]) -> list[str]:
    if not hits:
        return []
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
    return lines


def _render(sessions: list[dict], lessons: list[dict]) -> str:
    blocks = []
    lesson_lines = _render_lessons(lessons)
    if lesson_lines:
        blocks.append("\n".join(lesson_lines).rstrip())
    session_lines = _render_sessions(sessions)
    if session_lines:
        blocks.append("\n".join(session_lines).rstrip())
    return "\n\n".join(blocks)


def main() -> None:
    inp = _read_input()
    prompt = (inp.get("prompt") or "").strip()
    this_session = inp.get("session_id") or ""
    if len(prompt) < MIN_PROMPT_LEN:
        _silent_exit()
    sessions, lessons = _fetch(prompt, this_session)
    if not sessions and not lessons:
        _silent_exit()
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": _render(sessions, lessons),
        }
    }))


if __name__ == "__main__":
    main()
