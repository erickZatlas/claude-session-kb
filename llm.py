"""
llm.py — optional DeepSeek-powered clarification of session labels + summaries.

The whole module is lazy and degrades gracefully: if DEEPSEEK_API_KEY is missing or any
request fails, every function returns None and the rest of the app keeps running on the
TF-IDF fallback. Results are cached to .cache/llm.json so we never re-spend tokens for
unchanged sessions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from typing import Optional

log = logging.getLogger("llm")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, ".cache")
CACHE_PATH = os.path.join(CACHE_DIR, "llm.json")

_client = None
_client_lock = threading.Lock()
_warned = False

_cache: dict[str, str] = {}
_cache_loaded = False
_cache_lock = threading.Lock()


def is_enabled() -> bool:
    """Cheap check used by callers to decide whether to even attempt enrichment."""
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _load_cache() -> None:
    global _cache, _cache_loaded
    with _cache_lock:
        if _cache_loaded:
            return
        if os.path.exists(CACHE_PATH):
            try:
                _cache = json.load(open(CACHE_PATH))
            except Exception:
                _cache = {}
        _cache_loaded = True


def _save_cache() -> None:
    """Atomic write so concurrent runs can't corrupt the file."""
    with _cache_lock:
        os.makedirs(CACHE_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, prefix=".llm-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(_cache, f, ensure_ascii=False)
            os.replace(tmp, CACHE_PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _ensure_client():
    global _client, _warned
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            if not _warned:
                log.warning("DEEPSEEK_API_KEY not set — LLM clarification disabled.")
                _warned = True
            return None
        from openai import OpenAI  # lazy import, in case openai isn't installed yet
        _client = OpenAI(base_url=BASE_URL, api_key=key)
        return _client


# Prompts are module constants so the cache key includes them — any prompt tweak
# automatically invalidates stale entries instead of returning the old answer.
LABEL_SYSTEM = (
    "Return a kebab-case label that captures the topic of this work session — "
    "between 1 and 3 lowercase words joined by single hyphens. Prefer 2–3 words "
    "when two concepts are worth keeping (one specific + one generic = good); "
    "a single word is fine if one concept dominates. Real English words or known "
    "acronyms (lowercased: oxi, zif, pms, tls, bem, ihg). Correct typos — never "
    "echo a misspelled token. "
    "Examples: stuck-charges, opera-cloud, ihg-awaiting-checkin, no-show-fallback, "
    "bimester, payment-reconciliation, oxi-outage, claude-mem-kb. "
    "Output only the label."
)
SUMMARY_SYSTEM = (
    "Summarize what this work session was about in 1–2 short, plain sentences "
    "(under 30 words total). No preamble, no headings — just the sentences."
)
_SYSTEM = {"label": LABEL_SYSTEM, "summary": SUMMARY_SYSTEM}


def register_system(kind: str, prompt: str) -> None:
    """Extension hook for other modules (e.g. observe.py) to register their own
    system prompts. The cache key hashes the prompt text, so registering or
    changing one automatically invalidates only its own cached entries."""
    _SYSTEM[kind] = prompt


def _hash(kind: str, payload: str) -> str:
    h = hashlib.sha256()
    h.update(MODEL.encode()); h.update(b"\0")
    h.update(kind.encode()); h.update(b"\0")
    h.update(_SYSTEM[kind].encode("utf-8")); h.update(b"\0")
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def _call(kind: str, payload: str, max_tokens: int) -> Optional[str]:
    global _warned
    if not payload.strip():
        return None
    _load_cache()
    key = _hash(kind, payload)
    if key in _cache:
        return _cache[key]
    client = _ensure_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": _SYSTEM[kind]},
                      {"role": "user", "content": payload}],
            temperature=0,
            max_tokens=max_tokens,
        )
        out = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
        if out:
            with _cache_lock:
                _cache[key] = out
            _save_cache()
            return out
    except Exception as e:
        if not _warned:
            log.warning("DeepSeek call failed: %s — LLM disabled.", e)
            _warned = True
    return None


def get_cached(kind: str, payload: str) -> Optional[str]:
    """Read-only cache peek — used by reload() to apply enrichments atomically.
    Labels are reduced to a single word at read time, so display invariants hold
    regardless of how the entry was written."""
    if not payload:
        return None
    _load_cache()
    raw = _cache.get(_hash(kind, payload))
    if raw is None:
        return None
    return _to_kebab(raw) if kind == "label" else raw


def _to_kebab(text: Optional[str]) -> Optional[str]:
    """Normalize whatever the model returns into kebab-case (1–3 lowercase tokens
    joined by hyphens, cap ~24 chars; never cut mid-word). Defensive on read AND
    write so display invariants hold regardless of how the cache entry was first
    stored."""
    if not text:
        return None
    s = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    if not s:
        return None
    parts = [p for p in s.split("-") if p][:3]
    out = "-".join(parts)
    # if too long, drop trailing words instead of cutting mid-word
    while len(out) > 24 and "-" in out:
        out = out.rsplit("-", 1)[0]
    return out[:24].rstrip("-") or None


def label_for(payload: str) -> Optional[str]:
    """Return a kebab-case topic label for the session (or None)."""
    return _to_kebab(_call("label", payload, max_tokens=16))


def summary_for(payload: str) -> Optional[str]:
    """Return a 1–2 sentence summary (≤ ~30 words) of the session (or None)."""
    return _call("summary", payload, max_tokens=80)
