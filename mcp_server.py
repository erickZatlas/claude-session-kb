#!/usr/bin/env python3
"""
mcp_server.py — stdio MCP server that wraps the local claude-session-kb HTTP
API as one-call tools, so Claude Code can query its own session history
without curl + bash.

Two tools exposed:
  - search_my_sessions(query, limit=5, project=null, min_score=0.45)
      Semantic recall over past sessions. Calls GET /api/recall on
      localhost:8000 and formats the result as readable text + JSON.
  - get_session(session_id)
      Returns one session's stored label, summary, and observations (with
      topical tags). Calls GET /api/observe/{sid}.

The server itself is intentionally thin: all the real work — embedding,
cosine search, observation generation — lives in the FastAPI app at
http://127.0.0.1:8000. If the app isn't running, the tools return a clear
error instead of failing opaquely.

Registered in ~/.claude.json under mcpServers.session-kb.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as t

KB_BASE = os.environ.get("SESSION_KB_URL", "http://127.0.0.1:8000")
TIMEOUT_S = float(os.environ.get("SESSION_KB_TIMEOUT", "8"))

app = Server("session-kb")


def _http_get(path: str, query: dict) -> dict:
    """Synchronous GET that returns parsed JSON or raises with a useful message."""
    qs = urllib.parse.urlencode({k: v for k, v in query.items() if v not in (None, "")})
    url = f"{KB_BASE}{path}" + (f"?{qs}" if qs else "")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"kb server unreachable at {KB_BASE} ({e.reason}). "
            f"Start it with: cd ~/dev/claude-kb && python3 app.py"
        ) from None
    except Exception as e:  # noqa: BLE001 — surface anything else verbatim
        raise RuntimeError(f"kb request failed: {type(e).__name__}: {e}") from None


@app.list_tools()
async def list_tools() -> list[t.Tool]:
    return [
        t.Tool(
            name="search_my_sessions",
            description=(
                "Semantic search over the user's own past Claude Code sessions. "
                "Returns sessions ranked by cosine similarity over a 384-dim "
                "MiniLM embedding of each observation's title+text+tags. "
                "Use this when the user asks 'have I done this before?', "
                "'how did I solve X last time?', or when you want to ground a "
                "current task in prior work."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query (the topic or question to find).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max sessions to return (default 5).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Filter by project directory name "
                            "(e.g. 'my-api-service'). Omit for all projects."
                        ),
                    },
                    "min_score": {
                        "type": "number",
                        "description": (
                            "Minimum cosine similarity (0-1). Default 0.45. "
                            "Below 0.40 results are usually noise."
                        ),
                        "default": 0.45,
                    },
                },
                "required": ["query"],
            },
        ),
        t.Tool(
            name="get_session",
            description=(
                "Fetch the stored summary + topical observations for one session "
                "(by Claude Code session UUID). Use after search_my_sessions to "
                "drill into a specific candidate, or when the user pastes a "
                "session UUID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Claude Code session UUID.",
                    },
                },
                "required": ["session_id"],
            },
        ),
        t.Tool(
            name="find_sessions_by_file",
            description=(
                "Find past sessions that touched a specific file (read or "
                "edited it via PostToolUse capture). Use when the user is "
                "about to work on file X and you want to ground in prior "
                "work on the same file. Accepts an absolute path OR a "
                "bare basename (matches via suffix)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Full path or basename, e.g. 'app.py' or '/home/erick/dev/claude-kb/app.py'.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max sessions to return (default 10).",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["path"],
            },
        ),
        t.Tool(
            name="list_lessons",
            description=(
                "Cross-session 'lessons' distilled by /api/lessons/distill — "
                "durable facts that recur across many sessions (project "
                "conventions, architectural facts, recurring bug patterns). "
                "Use when grounding a task in long-standing project knowledge "
                "rather than one specific past session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Filter by a domain tag (e.g. 'JWT', 'FastAPI', 'kebab-labels'). Omit for all.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lessons to return (default 20).",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
            },
        ),
        t.Tool(
            name="list_memory_facts",
            description=(
                "List the user's hand-authored auto-memory facts — the durable "
                "one-fact-per-file notes under ~/.claude/projects/*/memory/*.md "
                "(who the user is, feedback/preferences, project constraints, "
                "external references). Use to ground in long-standing, "
                "explicitly-recorded user knowledge. These also surface "
                "automatically in search_my_sessions, but this lists them "
                "directly, optionally filtered by type."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Filter by memory type: user|feedback|project|reference. Omit for all.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max facts to return (default 50).",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
            },
        ),
    ]


def _fmt_search(payload: dict) -> str:
    """Render /api/recall response as readable text (with a tail JSON dump for
    the model to parse if it needs structured data). Surfaces both the matching
    sessions AND the durable knowledge block (distilled lessons + auto-memory)."""
    sessions = payload.get("sessions") or []
    lessons = payload.get("lessons") or []
    if not sessions and not lessons:
        return "No matching sessions or lessons found above the score threshold."
    lines: list[str] = []

    if lessons:
        lines.append(f"{len(lessons)} relevant lesson(s)/memory fact(s):")
        for l in lessons:
            if l.get("kind") == "memory":
                origin = f"memory:{l.get('memType') or 'note'}"
            else:
                ev = l.get("evidence") or 0
                origin = f"lesson·{ev} sessions" if ev else "lesson"
            tags = ", ".join(l.get("tags") or []) or "(none)"
            text = (l.get("text") or "").strip()
            lines.append(
                f"\n- [{origin}] {l.get('title') or '(untitled)'}  ·  score {l.get('score', 0.0):.3f}\n"
                f"  {text[:240]}" + ("…" if len(text) > 240 else "") + "\n"
                f"  tags: {tags}"
            )
        lines.append("")

    if sessions:
        lines.append(f"{len(sessions)} session(s):")
        for s in sessions:
            label = s.get("label") or "(no label)"
            score = s.get("score", 0.0)
            proj = s.get("project") or "?"
            summary = (s.get("summary") or "").strip()
            sid = s.get("id") or ""
            obs = s.get("obsCount") or 0
            lines.append(
                f"\n- {label}  ·  {proj}  ·  {obs} obs  ·  score {score:.3f}\n"
                f"  {summary[:220]}" + ("…" if len(summary) > 220 else "") + "\n"
                f"  session_id: {sid}"
            )

    lines.append("\n--- raw json ---\n" + json.dumps(
        {"sessions": sessions, "lessons": lessons}, ensure_ascii=False))
    return "\n".join(lines)


def _fmt_session(payload: dict) -> str:
    sess = payload.get("session") or {}
    obs = payload.get("observations") or []
    lines = [
        f"label:    {sess.get('label')}",
        f"project:  {sess.get('project')}",
        f"prompts:  {sess.get('prompt_count')}",
        f"summary:  {sess.get('summary')}",
        f"first_prompt: {(sess.get('first_prompt') or '')[:200]}",
        "",
        f"observations ({len(obs)}):",
    ]
    for o in obs:
        tags = ", ".join(o.get("tags") or []) or "(none)"
        lines.append(f"  [{o.get('type'):9}] {o.get('title')}")
        if o.get("text"):
            lines.append(f"             {o['text'][:180]}")
        lines.append(f"             tags: {tags}")
    return "\n".join(lines)


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[t.TextContent]:
    if name == "search_my_sessions":
        query = (arguments.get("query") or "").strip()
        if not query:
            return [t.TextContent(type="text", text="error: 'query' is required")]
        params = {
            "q": query,
            "limit": int(arguments.get("limit") or 5),
            "min_score": float(arguments.get("min_score") or 0.45),
        }
        if arguments.get("project"):
            params["project"] = arguments["project"]
        try:
            data = _http_get("/api/recall", params)
        except RuntimeError as e:
            return [t.TextContent(type="text", text=f"error: {e}")]
        return [t.TextContent(type="text", text=_fmt_search(data))]

    if name == "get_session":
        sid = (arguments.get("session_id") or "").strip()
        if not sid:
            return [t.TextContent(type="text", text="error: 'session_id' is required")]
        try:
            data = _http_get(f"/api/observe/{sid}", {})
        except RuntimeError as e:
            return [t.TextContent(type="text", text=f"error: {e}")]
        return [t.TextContent(type="text", text=_fmt_session(data))]

    if name == "find_sessions_by_file":
        path = (arguments.get("path") or "").strip()
        if not path:
            return [t.TextContent(type="text", text="error: 'path' is required")]
        try:
            data = _http_get("/api/sessions/by-file", {
                "path": path,
                "limit": int(arguments.get("limit") or 10),
            })
        except RuntimeError as e:
            return [t.TextContent(type="text", text=f"error: {e}")]
        rows = data.get("sessions") or []
        if not rows:
            return [t.TextContent(type="text", text=f"No sessions touched {path}.")]
        lines = [f"{len(rows)} session(s) touched {path}:"]
        for r in rows:
            lines.append(
                f"\n- {r.get('label') or '(no label)'}  ·  {r.get('project') or '?'}"
                f"  ·  {r.get('kind')}  ·  count {r.get('count')}\n"
                f"  session_id: {r.get('session_id')}"
            )
        lines.append("\n--- raw json ---\n" + json.dumps(rows, ensure_ascii=False))
        return [t.TextContent(type="text", text="\n".join(lines))]

    if name == "list_lessons":
        params: dict = {"limit": int(arguments.get("limit") or 20)}
        if arguments.get("tag"):
            params["tag"] = arguments["tag"]
        try:
            data = _http_get("/api/lessons", params)
        except RuntimeError as e:
            return [t.TextContent(type="text", text=f"error: {e}")]
        rows = data.get("lessons") or []
        if not rows:
            hint = " (run `POST /api/lessons/distill` to populate)" if not arguments.get("tag") else ""
            return [t.TextContent(type="text", text=f"No lessons found{hint}.")]
        lines = [f"{len(rows)} lesson(s):"]
        for L in rows:
            tags = ", ".join(L.get("tags") or []) or "(none)"
            sids = (L.get("source_session_ids") or [])[:3]
            lines.append(
                f"\n- {L.get('title')}\n  {L.get('text')}\n  tags: {tags}\n"
                f"  evidence: {L.get('evidence_count')} sessions, e.g. {sids}"
            )
        lines.append("\n--- raw json ---\n" + json.dumps(rows, ensure_ascii=False))
        return [t.TextContent(type="text", text="\n".join(lines))]

    if name == "list_memory_facts":
        params: dict = {"limit": int(arguments.get("limit") or 50)}
        if arguments.get("type"):
            params["type"] = arguments["type"]
        try:
            data = _http_get("/api/memory", params)
        except RuntimeError as e:
            return [t.TextContent(type="text", text=f"error: {e}")]
        rows = data.get("memory") or []
        if not rows:
            hint = " (run `POST /api/memory/sync` to ingest)" if not arguments.get("type") else ""
            return [t.TextContent(type="text", text=f"No memory facts found{hint}.")]
        lines = [f"{len(rows)} memory fact(s):"]
        for m in rows:
            tags = ", ".join(m.get("tags") or []) or "(none)"
            desc = (m.get("description") or "").strip()
            text = (m.get("text") or "").strip()
            lines.append(
                f"\n- [{m.get('mem_type') or 'note'}] {m.get('name') or '(unnamed)'}"
                f"  ·  {m.get('project') or '?'}\n"
                + (f"  {desc[:180]}\n" if desc else "")
                + f"  {text[:240]}" + ("…" if len(text) > 240 else "") + "\n"
                f"  tags: {tags}"
            )
        lines.append("\n--- raw json ---\n" + json.dumps(rows, ensure_ascii=False))
        return [t.TextContent(type="text", text="\n".join(lines))]

    return [t.TextContent(type="text", text=f"error: unknown tool '{name}'")]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
