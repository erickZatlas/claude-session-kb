---
name: rename-session
description: Rename an old Claude Code session by editing its transcript file. Use when the user wants to rename a CLOSED past session (the built-in `/rename` slash command only works on live sessions). Examples — "rename session XXXX", "rename my old XYZ session to abc", "fix the name of my old kb session".
---

# Rename Session

## When to trigger

- The user asks to rename a Claude Code session by ID, prefix, or phrase that maps to one.
- The user references "renaming an old session", "renaming a previous session", or "renaming a session I closed".
- The user pastes a `claude --resume <uuid>` line and asks to rename that session.

## When NOT to use this skill

- For the **current** session the user is in: just tell them to type `/rename <name>` — it's the built-in slash command and the script will refuse a live session anyway.
- For **bulk renaming** many sessions in one go: this skill is one-at-a-time; explain that and offer to loop.

## Naming format (match the `tabtitle` convention)

Same shape as tab titles: short, lowercase, hyphen-separated slugs — 2-3 words max.

Examples: `zif-trace`, `paraty-acceptance`, `ihg-posting-incident`, `no-show-fallback`, `claude-knowledge-base`.

- Lowercase only. Hyphens, not spaces.
- 2-3 words. Skip filler ("investigation", "check", "review", "fix") unless it adds meaning the noun phrase lacks.
- Subject + scope. Pick the most identifiable noun (system, hotel code, ticket subject) plus a one-word qualifier.
- **Never use ticket numbers (ZS-XXXX), PR numbers (#XXX), or other opaque IDs** unless the user explicitly asked for one.

If the user didn't pick a name, suggest one from the session's existing content (you can look it up with the `search_my_sessions` MCP tool or `GET /api/observe/<sid>` against `localhost:8000`).

## How to run it

The actual edit lives in a script in the claude-kb repo:

```bash
python3 ~/dev/claude-kb/scripts/rename-session.py <session-id-or-prefix> <new-name> [--sync-kb] [--dry-run]
```

Defaults:

- Prefix matching: a partial UUID (≥ 8 chars) is enough, as long as it's unambiguous.
- The script collapses any existing `custom-title` records and appends exactly one fresh one (idempotent — fixes the dup buildup that Claude Code's `/rename` produces).
- Refuses any session that looks live: either a `claude` process has the UUID in its argv, or the transcript was written within the last 30 seconds.
- Backs up the transcript to `<file>.bak.<unix-ts>` before writing.

Useful flags:

- `--sync-kb` — also write the new name to `~/.claude-kb/data.db`'s `sessions.label` so the KB web-UI card heading matches. **Use this by default** when renaming a session that already appears in the KB; you can skip it for old sessions the KB hasn't seen.
- `--dry-run` — preview without writing. Use this whenever the user is unsure.
- `--force` — override the live-session safety check. Use ONLY when restoring a name immediately after a prior rename (the mtime guard fires on our own write) or when the user explicitly asks to override.

## Steps for the assistant

1. **Identify the session.** Ask the user for the UUID/prefix if they didn't paste one. If they gave only a topic ("the oxi outage session from last week"), use `search_my_sessions` (MCP tool) or curl `localhost:8000/api/recall?q=<topic>` to find candidates and confirm with the user before proceeding.
2. **Propose the new name** if the user hasn't given one. Use the format above; offer 2-3 options and let them pick.
3. **Always dry-run first** if there's any ambiguity ("looks like this is the one — preview the change?"):
   ```bash
   python3 ~/dev/claude-kb/scripts/rename-session.py <prefix> <new-name> --dry-run
   ```
4. **Run for real with `--sync-kb`** by default (so the KB card matches):
   ```bash
   python3 ~/dev/claude-kb/scripts/rename-session.py <prefix> <new-name> --sync-kb
   ```
5. **Report what happened**: old name(s) dropped, new name set, file path, KB rows updated.
6. If the rename is for a session the user is actively reviewing **in a tab**, remind them they may also want `wezterm cli set-tab-title <new-name>` to keep the WezTerm tab title in sync (the `tabtitle` skill handles that).

## Failure modes & what to do

- **"no transcript matches"** — the prefix doesn't exist under `~/.claude/projects/*`. Ask the user to double-check the UUID, or list candidates with `ls ~/.claude/projects/*/<prefix>*.jsonl`.
- **"prefix matches multiple transcripts"** — disambiguate; the script prints the candidates.
- **"looks live"** — the session is probably open in another tab. Ask the user to close it. If they insist it's not live (false positive from mtime), pass `--force`.
- **"session id … must be at least 8 characters"** — too-short prefix risks mis-matching. Get more characters from the user.
