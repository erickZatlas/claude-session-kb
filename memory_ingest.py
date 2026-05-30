"""
memory_ingest.py — sync the hand-authored auto-memory into our KB.

Claude Code keeps a per-project memory directory at
``~/.claude/projects/<dir-slug>/memory/*.md`` — one durable fact per file, with
YAML-ish frontmatter (``name`` / ``description`` / ``metadata.type``) and the
fact itself as the markdown body. ``MEMORY.md`` is just the index and is skipped.

This module reads those files and mirrors each into the ``memory_facts`` table
(see store_capture.py). store.py then projects them into the shared record
corpus so the embedder/search/recall code surfaces them like any observation.

Dependency-free on purpose (no PyYAML): the frontmatter shape is small and
regular, so a tiny hand parser keeps the local-only / degrade-cleanly contract.
Any unreadable or malformed file is skipped silently rather than failing the scan.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re

import observe  # reuse the same topical-tag extractor used for observations

MEMORY_GLOB = os.path.expanduser("~/.claude/projects/*/memory/*.md")

_MEM_TYPES = ("user", "feedback", "project", "reference")
_FM_LINE = re.compile(r"^(\s*)([A-Za-z0-9_-]+):\s*(.*)$")


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Return (meta, body). meta is a flat dict plus a nested 'metadata' dict.
    Tolerates a missing/garbled frontmatter block — then meta is {} and the
    whole text is the body."""
    if not raw.startswith("---"):
        return {}, raw.strip()
    # Split off the frontmatter between the first two '---' fences.
    parts = raw.split("\n")
    if parts[0].strip() != "---":
        return {}, raw.strip()
    end = None
    for i in range(1, len(parts)):
        if parts[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, raw.strip()
    meta: dict = {}
    parent: str | None = None
    for line in parts[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _FM_LINE.match(line)
        if not m:
            continue
        indent, key, val = len(m.group(1)), m.group(2), _strip_quotes(m.group(3))
        if indent == 0:
            if val == "":
                parent = key            # opens a nested block, e.g. "metadata:"
                meta.setdefault(key, {})
            else:
                parent = None
                meta[key] = val
        elif parent is not None:
            meta[parent][key] = val
    body = "\n".join(parts[end + 1:]).strip()
    return meta, body


def _project_for_dir(dir_slug: str, known: list[str]) -> str:
    """Map a memory directory slug to a real project name. The slug is the cwd
    with separators flattened to '-' (e.g. '-home-erick-dev-claude-kb'), which
    can't be reversed unambiguously, so we suffix-match against the project
    names we already know from the sessions table and take the longest match.
    Falls back to 'global' (the bare ~/.claude/projects/-home-erick case)."""
    best = ""
    for p in known:
        if p and dir_slug.endswith(p) and len(p) > len(best):
            best = p
    return best or "global"


def _fact_from_file(path: str, known: list[str]) -> dict | None:
    try:
        raw = open(path, encoding="utf-8").read()
        mtime = int(os.path.getmtime(path) * 1000)
    except OSError:
        return None
    meta, body = _parse_frontmatter(raw)
    if not body:
        return None
    md = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    name = (meta.get("name") or os.path.splitext(os.path.basename(path))[0]).strip()
    description = (meta.get("description") or "").strip()
    mem_type = (md.get("type") or "").strip().lower()
    if mem_type not in _MEM_TYPES:
        mem_type = "reference"
    dir_slug = os.path.basename(os.path.dirname(os.path.dirname(path)))
    project = _project_for_dir(dir_slug, known)
    # Extract from description + body only (NOT name — it's the filename slug,
    # which would tag every fact with its own ugly snake_case basename).
    tags = observe.extract_topical_tags(description, body)
    if mem_type not in tags:
        tags = [mem_type] + tags
    content_hash = hashlib.sha1(body.encode("utf-8")).hexdigest()
    return {
        "id": f"{project}::{name}",
        "project": project,
        "source_path": path,
        "name": name,
        "mem_type": mem_type,
        "description": description,
        "text": body,
        "tags": tags[:8],
        "content_hash": content_hash,
        "mtime": mtime,
    }


def scan(capture) -> dict:
    """Ingest every memory file into capture's memory_facts table. Skips
    MEMORY.md, no-ops on unchanged files, prunes facts whose source file was
    deleted. Returns a small stats dict. Never raises — a broken file or a
    missing memory dir just yields zeros."""
    stats = {"scanned": 0, "inserted": 0, "updated": 0, "unchanged": 0, "pruned": 0}
    try:
        known = capture.distinct_projects()
    except Exception:
        known = []
    seen_ids: list[str] = []
    for path in glob.glob(MEMORY_GLOB):
        if os.path.basename(path) == "MEMORY.md":
            continue
        fact = _fact_from_file(path, known)
        if not fact:
            continue
        stats["scanned"] += 1
        seen_ids.append(fact["id"])
        try:
            res = capture.upsert_memory_fact(fact)
            stats[res] = stats.get(res, 0) + 1
        except Exception:
            pass
    try:
        stats["pruned"] = capture.prune_memory_facts(seen_ids)
    except Exception:
        pass
    return stats
