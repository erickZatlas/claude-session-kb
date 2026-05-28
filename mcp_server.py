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
                            "(e.g. 'zatlas-pms-middleware'). Omit for all projects."
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
    ]


def _fmt_search(payload: dict) -> str:
    """Render /api/recall response as readable text (with a tail JSON dump for
    the model to parse if it needs structured data)."""
    sessions = payload.get("sessions") or []
    if not sessions:
        return "No matching sessions found above the score threshold."
    lines = [f"{len(sessions)} session(s):"]
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
    lines.append("\n--- raw json ---\n" + json.dumps(sessions, ensure_ascii=False))
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

    return [t.TextContent(type="text", text=f"error: unknown tool '{name}'")]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
