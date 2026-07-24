---
name: delete-session
description: Delete/remove an old Claude Code session so it disappears from the session picker (Claude Code has no built-in delete). Use when the user wants to remove, delete, or clean up past sessions. Examples — "delete session XXXX", "remove my old XYZ session", "clean up these sessions from the picker".
---

# Delete Session

## When to trigger

- The user asks to delete or remove a Claude Code session by ID, prefix, or phrase that maps to one.
- The user wants to "clean up" the session picker / resume list.
- The user pastes a `claude --resume <uuid>` line and asks to get rid of that session.

## When NOT to use this skill

- For the **current** live session: the script refuses live sessions. Tell the user to close it first (or that deleting the session they're in makes no sense).
- If the user just wants old sessions to **age out automatically**: point them at `cleanupPeriodDays` in `~/.claude/settings.json` instead (default 30-day retention).
- If the concern is **sensitive content never being written**: `CLAUDE_CODE_SKIP_PROMPT_HISTORY` suppresses transcript writes entirely; `--no-session-persistence` does it for a single `claude -p` run.

## How to run it

```bash
python3 ~/dev/claude-kb/scripts/delete-session.py <session-id-or-prefix> [...] [--purge] [--sync-kb] [--dry-run] [--force]
```

Defaults:

- Prefix matching: a partial UUID (≥ 8 chars) is enough, as long as it's unambiguous. Multiple session ids can be passed in one call.
- **Reversible by default**: the transcript (plus any `.bak.*` siblings left by `rename-session.py`) is MOVED to `~/.claude-kb/trash/<workspace>/`. Moving the file back restores the session.
- Refuses any session that looks live (a `claude` process has the UUID in its argv, or the transcript was written within the last 30 seconds).

Flags:

- `--purge` — permanently unlink instead of trashing. **Confirm with the user before using this**; it is not recoverable.
- `--sync-kb` — also delete the session's rows from `~/.claude-kb/data.db` (sessions + cascaded prompts/observations/tool_calls/session_files). **Use this by default** when the session appears in the KB, otherwise the KB web UI keeps showing a card for a session that can no longer be resumed. The embedding cache compacts itself on the next sync.
- `--dry-run` — preview without touching anything. Use whenever the user is unsure which session a prefix matches.
- `--force` — override the live-session safety check. Only when the user explicitly confirms it's a false positive.

## Steps for the assistant

1. **Identify the session(s).** Ask for the UUID/prefix if not given. If the user gave only a topic ("that old auth session"), use `search_my_sessions` (MCP tool) or curl `localhost:8000/api/recall?q=<topic>` to find candidates and confirm with the user before deleting anything.
2. **Preview first when there's any ambiguity** — a delete acts on whatever the prefix matches:
   ```bash
   python3 ~/dev/claude-kb/scripts/delete-session.py <prefix> --dry-run
   ```
   You can also show the session's first prompt so the user can confirm it's the right one:
   ```bash
   head -c 400 ~/.claude/projects/*/<prefix>*.jsonl
   ```
3. **Run for real with `--sync-kb`** by default:
   ```bash
   python3 ~/dev/claude-kb/scripts/delete-session.py <prefix> --sync-kb
   ```
4. **Only add `--purge` if the user explicitly asked for permanent deletion** — the default trash is the right call otherwise. Mention that trash lives at `~/.claude-kb/trash/` and can be emptied later.
5. **Report what happened**: which files were trashed/purged, where they went, and how many KB rows were deleted.

## Failure modes & what to do

- **"no transcript matches"** — the prefix doesn't exist under `~/.claude/projects/*`. Double-check the UUID or list candidates with `ls ~/.claude/projects/*/<prefix>*.jsonl`.
- **"prefix matches multiple transcripts"** — disambiguate; the script prints the candidates.
- **"looks live"** — the session is probably open in another tab. Ask the user to close it; pass `--force` only if they insist it's a false positive.
- **Restore request** — move the file back from `~/.claude-kb/trash/<workspace>/<sid>.jsonl` to `~/.claude/projects/<workspace>/`. KB rows deleted by `--sync-kb` are re-created only if the session is re-ingested; mention that.
