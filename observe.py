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
    "trade-off", "tradeoff", "tradeoffs",
})

# Used by the Phase D bulk import. We deliberately do NOT call DeepSeek for the
# 5,000+ historical rows; topical tags are extracted client-side from the rich
# title/subtitle/narrative claude-mem already produced. Cheap, deterministic,
# avoids burning a few dollars on a one-shot migration.
_EXTRACT_PATTERNS = (
    # CamelCase / PascalCase identifiers (e.g. OperaPostCharge, ChargeEvent)
    re.compile(r"\b([A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z0-9_]+)+)\b"),
    # All-caps acronyms with optional digits (e.g. OXI, ZIF, PUEGH, ZS-5953, AWS).
    # Stopword-checked below so SHOUTED English (RESIZE, READY, ABOUT) gets dropped.
    re.compile(r"\b([A-Z]{2,}(?:[_\-][A-Z0-9]+)*)\b"),
    # kebab-case multi-word (e.g. opera-cloud, no-show-fallback) — at least one hyphen
    re.compile(r"\b([a-z]+(?:-[a-z]+){1,3})\b"),
    # numeric IDs of 4+ digits (PMS ids, ticket numbers, ports, channel ids)
    re.compile(r"\b(\d{4,})\b"),
    # snake_case / dotted identifiers / filenames — require an explicit `_` or `.`
    # in the body so plain English nouns ("window", "title", "session") can't sneak
    # in. Matches store_capture.py, channel_property_id, app.py.
    re.compile(r"\b([a-z][a-z0-9]*(?:[_.][a-z0-9_]+)+(?:\.[a-z]+)?)\b"),
)

# Lowercase stopwords applied to every match (regardless of pattern). Keeps
# domain words while dropping plain English. Generous on purpose — false
# negatives (filtering a real token) are recoverable; false positives
# (tagging "current") pollute search.
_TAG_STOPWORDS = frozenset({
    # function words
    "this", "that", "with", "from", "into", "their", "have", "been",
    "the", "and", "but", "for", "you", "are", "was", "were", "they",
    "we", "my", "our", "your", "his", "her", "its", "all", "any",
    "to", "in", "on", "at", "of", "as", "by", "or", "if", "is", "be",
    "do", "did", "done", "what", "when", "where", "why", "how", "who",
    "via", "use", "used", "make", "made", "set", "got", "has", "had",
    # SHOUTED English the all-caps pattern would otherwise catch
    "resize", "ready", "about", "while", "should", "would", "could",
    "must", "will", "shall", "true", "false", "null", "yes", "open",
    "close", "edit", "save", "load", "read", "write", "start", "stop",
    # generic project nouns
    "session", "sessions", "work", "task", "tasks", "info", "data",
    "code", "thing", "things", "stuff", "phase", "phases", "step", "steps",
    "feature", "features", "bug", "bugs", "fix", "fixes", "change", "changes",
    "title", "subtitle", "body", "header", "footer", "window", "bar", "mode",
    "current", "previous", "next", "old", "new", "first", "last", "final",
    "result", "results", "output", "input", "value", "values", "list", "lists",
    "object", "objects", "item", "items", "field", "fields", "row", "rows",
    "table", "tables", "file", "files", "line", "lines", "page", "pages",
    "user", "users", "system", "systems", "config", "settings",
    "name", "names", "type", "types", "kind", "kinds", "form", "forms",
    "way", "ways", "case", "cases", "part", "parts", "side", "sides",
    "issue", "issues", "problem", "problems", "solution", "solutions",
    "approach", "method", "methods", "tool", "tools", "test", "tests",
    "review", "reviews", "report", "reports", "comment", "comments",
    "support", "supported", "available", "enabled", "disabled",
    "main", "core", "base", "default", "common", "general", "specific",
    "simple", "complex", "small", "large", "big", "tiny", "huge",
    "good", "bad", "best", "worst", "high", "low", "fast", "slow",
    "make-up", "true-positive", "false-positive",  # multi-word noise
})


def extract_topical_tags(*texts: Optional[str], max_tags: int = 5) -> list[str]:
    """Extract domain-specific tags from raw text using regex patterns. Used for
    Phase D bulk import — preserves claude-mem's historical observations with
    decent tags without paying for thousands of LLM calls. The kebab/snake
    branches are stopword-filtered so we don't tag every paragraph with
    'the-and-but'; CamelCase, all-caps and numeric ID matches pass through."""
    counts: dict[str, int] = {}
    order: dict[str, int] = {}
    next_ord = 0
    for chunk in texts:
        if not chunk:
            continue
        for i, pat in enumerate(_EXTRACT_PATTERNS):
            for m in pat.findall(chunk):
                token = m.strip()
                if not token or len(token) < 3 or len(token) > 40:
                    continue
                key = token
                # Stopword filter applies only to lowercase patterns (kebab + snake).
                if i >= 2 and token.lower() in _TAG_STOPWORDS:
                    continue
                if token.lower() in _GENERIC_TAGS:
                    continue
                if key not in counts:
                    counts[key] = 0
                    order[key] = next_ord
                    next_ord += 1
                counts[key] += 1
    # Sort by frequency desc, then by first-seen order for deterministic output
    ranked = sorted(counts.items(),
                    key=lambda kv: (-kv[1], order[kv[0]]))
    seen_lower: set[str] = set()
    out: list[str] = []
    for tok, _ in ranked:
        lo = tok.lower()
        if lo in seen_lower:
            continue
        seen_lower.add(lo)
        out.append(tok)
        if len(out) >= max_tags:
            break
    return out


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
