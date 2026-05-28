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
    "Return EXACTLY ONE plain word that captures the topic of this work session — "
    "a single noun, scannable at a glance. No spaces, no hyphens, NO CamelCase "
    "compounds (do not glue two ideas together — pick the more important one). "
    "It MUST be a real English word or a well-known acronym (OXI, ZIF, PMS, TLS, BEM, IHG). "
    "If the user's prompt contains obvious typos, correct them — e.g. 'reservals' → "
    "'Reversals', 'updaate' → 'Update'. Never echo a misspelled token. "
    "Examples: Bimester, Charges, Opera, Webhook, Refactoring, Sandbox, Outage, Reversals. "
    "Output only that one word."
)
SUMMARY_SYSTEM = (
    "Summarize what this work session was about in 1–2 short, plain sentences "
    "(under 30 words total). No preamble, no headings — just the sentences."
)
_SYSTEM = {"label": LABEL_SYSTEM, "summary": SUMMARY_SYSTEM}


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
    return _one_word(raw) if kind == "label" else raw


def _one_word(text: Optional[str]) -> Optional[str]:
    """Defensive: keep only the first whitespace-separated token, strip punctuation."""
    if not text:
        return None
    head = text.strip().split()[0] if text.strip() else ""
    head = re.sub(r"^[^\w]+|[^\w]+$", "", head)  # trim leading/trailing punctuation
    return head[:24] or None


def label_for(payload: str) -> Optional[str]:
    """Return a single-word topic for the session (or None)."""
    return _one_word(_call("label", payload, max_tokens=12))


def summary_for(payload: str) -> Optional[str]:
    """Return a 1–2 sentence summary (≤ ~30 words) of the session (or None)."""
    return _call("summary", payload, max_tokens=80)
