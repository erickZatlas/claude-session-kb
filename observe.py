"""
observe.py — Phase C: generate topical observations + session clarification
from a captured session.

Inputs: a session record (project, first_prompt) + its list of prompt texts.
Output: 3–7 observations, each with type / title / text / domain-specific tags.
Plus a kebab label + 1–2 sentence summary for the session itself (reuses llm.py
where possible so we don't duplicate the prompt + cache machinery).

Topical-tag invariant: tags are domain identifiers (`AWAITING_CHECKIN`,
`OperaPostCharge`, `kebab-labels`, `webhook-fix`), NEVER claude-mem's generic
`how-it-works` / `what-changed` / `pattern` filler. The system prompt makes
this explicit and the cache key includes that prompt so any tuning
auto-invalidates.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import llm  # reuse the DeepSeek client, cache, and post-processing

log = logging.getLogger("observe")

OBSERVATIONS_SYSTEM = (
    "You read a developer's Claude Code session and extract 3 to 7 topical "
    "observations. Each observation captures ONE specific thing the user "
    "learned, decided, built, or fixed — not what they merely wanted to do.\n\n"
    "Return a JSON ARRAY of objects, each with these exact keys:\n"
    "  - type: one of \"discovery\" | \"change\" | \"bugfix\" | \"decision\" | "
    "\"refactor\" | \"feature\"\n"
    "  - title: one specific line (max ~120 chars)\n"
    "  - text: 1–2 sentences expanding the title\n"
    "  - tags: an array of 2–5 DOMAIN-SPECIFIC tokens — acronyms, identifiers, "
    "file names, library/tool names, kebab-case concepts. "
    "NEVER use generic tags like \"how-it-works\", \"what-changed\", \"pattern\", "
    "\"general\", \"session\", \"work\", \"task\". Prefer things like "
    "AWAITING_CHECKIN, OperaPostCharge, kebab-labels, FastAPI, MiniLM, OXI, ZIF.\n\n"
    "Output ONLY the JSON array. No prose around it, no markdown fence."
)

_MODEL = llm.MODEL  # use the same DeepSeek model the rest of the app uses

# Register our system prompt so llm._call("observations", ...) can hash + invoke it.
llm.register_system("observations", OBSERVATIONS_SYSTEM)


def _build_payload(project: Optional[str], prompts: list[str], cap: int = 30) -> str:
    """Compact representation of the session for the model. Front-load the
    earliest prompts (they tend to anchor the topic), then sample later ones."""
    seen = prompts[:cap]
    lines = [f"Project: {project or 'unknown'}", "", "Session prompts (in order):"]
    for i, p in enumerate(seen, 1):
        # Trim absurdly long single prompts; full text not needed for topic extraction
        snippet = p.strip().replace("\r", "").replace("\n", " ⏎ ")
        if len(snippet) > 600:
            snippet = snippet[:600].rstrip() + "…"
        lines.append(f"{i}. {snippet}")
    return "\n".join(lines)[:8000]


def _strip_fence(s: str) -> str:
    """Models sometimes wrap JSON in ```json ... ``` despite being told not to."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


_GENERIC_TAGS = frozenset({
    "how-it-works", "what-changed", "pattern", "general", "session", "work",
    "task", "tasks", "code", "data", "thing", "stuff", "info", "things",
})


def _clean_tags(raw) -> list[str]:
    """Drop generic tags, enforce length, dedupe (case-insensitive)."""
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s or len(s) < 2 or len(s) > 40:
            continue
        if s.lower() in _GENERIC_TAGS:
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= 8:
            break
    return out


def _normalize(obs: dict) -> Optional[dict]:
    """Coerce one observation dict to our shape; drop garbage."""
    if not isinstance(obs, dict):
        return None
    title = (obs.get("title") or "").strip()
    if not title:
        return None
    t = (obs.get("type") or "discovery").strip().lower()
    if t not in ("discovery", "change", "bugfix", "decision", "refactor", "feature"):
        t = "discovery"
    return {
        "type": t,
        "title": title[:200],
        "text": (obs.get("text") or "").strip(),
        "tags": _clean_tags(obs.get("tags")),
    }


def generate_observations(project: Optional[str], prompts: list[str]) -> list[dict]:
    """Returns a list of normalized observation dicts, or [] on any failure."""
    if not prompts:
        return []
    payload = _build_payload(project, prompts)
    # _call already hashes (model, kind, system, payload) so any prompt tweak
    # invalidates stale cached observations automatically.
    raw = llm._call("observations", payload, max_tokens=900)
    if not raw:
        return []
    try:
        parsed = json.loads(_strip_fence(raw))
    except json.JSONDecodeError:
        log.warning("observations: invalid JSON from model, dropping: %r", raw[:200])
        return []
    if not isinstance(parsed, list):
        return []
    cleaned: list[dict] = []
    for item in parsed:
        n = _normalize(item)
        if n:
            cleaned.append(n)
        if len(cleaned) >= 8:
            break
    return cleaned


def clarify_session(project: Optional[str], prompts: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Returns (kebab_label, summary) for the session, via the existing
    llm.label_for / llm.summary_for so we share the same cache + prompt
    invalidation rules with the older claude-mem-backed enrichment path."""
    if not prompts:
        return (None, None)
    payload = _build_payload(project, prompts, cap=20)
    label = llm.label_for(payload)
    summary = llm.summary_for(payload)
    return (label, summary)
