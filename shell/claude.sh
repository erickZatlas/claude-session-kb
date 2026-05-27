# Claude Code per-task session helpers.
#
# Process: one kebab "slug" per task, shared across three places so a task is
# identifiable everywhere — the Claude session name (-n), the wezterm tab title,
# and (optionally) a nested git worktree.
#
#   ctask <slug> [-w] [claude args...]   start a NEW named session for a task
#   cresume <slug> [claude args...]      reopen a session by name (picker pre-filtered)
#   cont [claude args...]                continue most recent session in this dir
#   cfind <phrase>                       fallback: find a session by its CONTENT

# Set the wezterm tab title (persists after Claude exits; no-op outside wezterm).
_ct_tab_title() {
  [ -n "$1" ] || return 0
  command -v wezterm >/dev/null 2>&1 && wezterm cli set-tab-title "$1" 2>/dev/null
  return 0
}

# ctask <slug> [-w] [claude args...]
#   -w : also create/enter a nested worktree .claude/worktrees/<repo>-<slug>
#        on branch <slug> (off main), so `cont` later lands on this task.
ctask() {
  if [ -z "$1" ]; then
    echo "usage: ctask <slug> [-w] [claude args...]" >&2
    return 1
  fi
  local slug="$1"; shift
  if [ "$1" = "-w" ]; then
    shift
    local top dir
    top="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "ctask -w: not in a git repo" >&2; return 1; }
    dir="${top}/.claude/worktrees/$(basename "$top")-${slug}"
    if [ -d "$dir" ]; then
      echo "ctask: worktree already exists, entering ${dir}"
    elif git show-ref --verify --quiet "refs/heads/${slug}"; then
      git worktree add "$dir" "$slug" || return 1
    else
      git worktree add -b "$slug" "$dir" main || return 1
    fi
    cd "$dir" || return 1
  fi
  _ct_tab_title "$slug"
  claude -n "$slug" "$@"
}

# cresume <slug> [claude args...] — reopen by name (opens picker pre-filtered to <slug>).
cresume() {
  if [ -z "$1" ]; then
    echo "usage: cresume <slug> [claude args...]" >&2
    return 1
  fi
  local slug="$1"; shift
  _ct_tab_title "$slug"
  claude --resume "$slug" "$@"
}

# cont [claude args...] — continue the most recent session in the current directory.
cont() { claude --continue "$@"; }

# cfind <phrase> — fallback for when you forget a session's name (or it aged out):
# grep the current repo's transcripts for <phrase> and list date / uuid / first prompt.
cfind() {
  if [ -z "$1" ]; then
    echo "usage: cfind <phrase>" >&2
    return 1
  fi
  local root proj
  root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  proj="$HOME/.claude/projects/$(printf '%s' "$root" | sed 's#/#-#g')"
  if [ ! -d "$proj" ]; then
    echo "no session dir for ${root}" >&2
    return 1
  fi
  grep -rl -- "$1" "$proj"/*.jsonl 2>/dev/null | while read -r f; do
    python3 - "$f" <<'PY'
import json, os, sys, datetime
f = sys.argv[1]
first = None
with open(f) as fh:
    for line in fh:
        try:
            o = json.loads(line)
        except Exception:
            continue
        if first is None and o.get("type") == "user":
            c = o.get("message", {}).get("content")
            if isinstance(c, str):
                t = c
            elif isinstance(c, list):
                t = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            else:
                t = ""
            t = t.strip()
            if t and not t.startswith("<"):
                first = t[:80]
ts = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
print(f"{ts}  {os.path.basename(f)[:8]}  {first or '(no prompt)'}")
PY
  done | sort -r
  echo "→ resume with: claude --resume <uuid>   (or cresume <name>)"
}
