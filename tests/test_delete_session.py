"""Tests for session_delete.py — trash/purge a session transcript and
optionally its KB rows. Shared by the CLI (scripts/delete-session.py) and the
web API (DELETE /api/sessions/{sid})."""
from __future__ import annotations

import os
import sqlite3

import pytest

import session_delete as sd

SID = "f3cdc10e-8cf0-428a-b25e-329ce6346081"
OTHER_SID = "aaaa1111-2222-3333-4444-555566667777"


@pytest.fixture
def projects(tmp_path, monkeypatch):
    """A fake ~/.claude/projects with one project dir and two transcripts."""
    proj = tmp_path / "projects" / "-home-erick-dev-claude-kb"
    proj.mkdir(parents=True)
    (proj / f"{SID}.jsonl").write_text('{"type":"user"}\n')
    (proj / f"{OTHER_SID}.jsonl").write_text('{"type":"user"}\n')
    # Backdate mtimes so the <30s "looks live" heuristic doesn't fire on files
    # we just wrote — these fixtures represent old, closed sessions.
    old = 1700000000
    for f in proj.glob("*.jsonl"):
        os.utime(f, (old, old))
    monkeypatch.setattr(sd, "PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(sd, "TRASH_DIR", str(tmp_path / "trash"))
    return proj


@pytest.fixture
def kb_db(tmp_path, monkeypatch):
    """A KB data.db with the real schema and rows for two sessions."""
    db = tmp_path / "data.db"
    import store_capture
    con = sqlite3.connect(db)
    con.executescript(store_capture._SCHEMA)
    now = 1700000000
    for sid in (SID, OTHER_SID):
        con.execute(
            "INSERT INTO sessions (id, project, started_at) VALUES (?, ?, ?)",
            (sid, "claude-kb", now),
        )
        con.execute(
            "INSERT INTO prompts (session_id, ts, text) VALUES (?, ?, ?)",
            (sid, now, "hello"),
        )
        con.execute(
            "INSERT INTO observations (session_id, ord, type, title, created_at) "
            "VALUES (?, 0, 'discovery', 'obs', ?)",
            (sid, now),
        )
        con.execute(
            "INSERT INTO tool_calls (session_id, ts, tool_name) VALUES (?, ?, 'Bash')",
            (sid, now),
        )
        con.execute(
            "INSERT INTO session_files (session_id, path, kind, first_seen, last_seen) "
            "VALUES (?, '/x.py', 'read', ?, ?)",
            (sid, now, now),
        )
    con.commit()
    con.close()
    monkeypatch.setattr(sd, "KB_DB", str(db))
    return db


# ---- find_transcript ----

def test_find_rejects_short_prefix(projects):
    with pytest.raises(sd.SessionDeleteError):
        sd.find_transcript("f3cd")


def test_find_returns_none_on_no_match(projects):
    assert sd.find_transcript("deadbeef") is None


def test_find_rejects_ambiguous(projects):
    twin = SID[:8] + "-ffff-ffff-ffff-ffffffffffff"
    (projects / f"{twin}.jsonl").write_text("{}\n")
    with pytest.raises(sd.SessionDeleteError):
        sd.find_transcript(SID[:8])


def test_find_by_prefix(projects):
    assert sd.find_transcript(SID[:8]).endswith(f"{SID}.jsonl")


def test_find_ignores_bak_files(projects):
    """A .bak.* sibling must not count as a second match."""
    (projects / f"{SID}.jsonl.bak.123").write_text("{}\n")
    assert sd.find_transcript(SID).endswith(f"{SID}.jsonl")


# ---- trash (default) ----

def test_trash_moves_transcript_and_baks(projects):
    bak = projects / f"{SID}.jsonl.bak.1700000000"
    bak.write_text("{}\n")
    path = str(projects / f"{SID}.jsonl")

    result = sd.trash_transcript(path, dry_run=False)

    assert not os.path.exists(path)
    assert not bak.exists()
    trashed = result["moved"]
    assert len(trashed) == 2
    for dest in trashed:
        assert os.path.exists(dest)
        assert "-home-erick-dev-claude-kb" in dest
    assert (projects / f"{OTHER_SID}.jsonl").exists()


def test_trash_dry_run_moves_nothing(projects):
    path = str(projects / f"{SID}.jsonl")
    result = sd.trash_transcript(path, dry_run=True)
    assert os.path.exists(path)
    assert result["dry_run"] is True
    assert len(result["would_move"]) == 1


def test_trash_collision_gets_suffix(projects):
    path = str(projects / f"{SID}.jsonl")
    first = sd.trash_transcript(path, dry_run=False)["moved"][0]
    (projects / f"{SID}.jsonl").write_text('{"type":"user","restored":true}\n')
    second = sd.trash_transcript(path, dry_run=False)["moved"][0]
    assert first != second
    assert os.path.exists(first) and os.path.exists(second)


# ---- purge ----

def test_purge_removes_files(projects):
    bak = projects / f"{SID}.jsonl.bak.1700000000"
    bak.write_text("{}\n")
    path = str(projects / f"{SID}.jsonl")

    result = sd.purge_transcript(path, dry_run=False)

    assert not os.path.exists(path)
    assert not bak.exists()
    assert len(result["removed"]) == 2
    assert (projects / f"{OTHER_SID}.jsonl").exists()


def test_purge_dry_run_removes_nothing(projects):
    path = str(projects / f"{SID}.jsonl")
    result = sd.purge_transcript(path, dry_run=True)
    assert os.path.exists(path)
    assert result["dry_run"] is True


# ---- KB rows ----

def _counts(db, sid):
    con = sqlite3.connect(db)
    try:
        return {
            t: con.execute(
                f"SELECT COUNT(*) FROM {t} WHERE {'id' if t == 'sessions' else 'session_id'} = ?",
                (sid,),
            ).fetchone()[0]
            for t in ("sessions", "prompts", "observations", "tool_calls", "session_files")
        }
    finally:
        con.close()


def test_kb_delete_cascades(projects, kb_db):
    result = sd.delete_kb_rows(SID, dry_run=False)
    assert result["deleted"]["sessions"] == 1
    gone = _counts(kb_db, SID)
    assert all(v == 0 for v in gone.values()), gone
    kept = _counts(kb_db, OTHER_SID)
    assert all(v == 1 for v in kept.values()), kept


def test_kb_delete_dry_run(projects, kb_db):
    result = sd.delete_kb_rows(SID, dry_run=True)
    assert result["dry_run"] is True
    assert result["would_delete"]["sessions"] == 1
    assert _counts(kb_db, SID)["sessions"] == 1


def test_kb_delete_missing_db(projects, monkeypatch, tmp_path):
    monkeypatch.setattr(sd, "KB_DB", str(tmp_path / "nope.db"))
    result = sd.delete_kb_rows(SID, dry_run=False)
    assert result["skipped"] is True


# ---- delete_session orchestrator ----

def test_delete_session_trashes_and_syncs_kb(projects, kb_db):
    result = sd.delete_session(SID, sync_kb=True)
    assert result["sid"] == SID
    assert len(result["transcript"]["moved"]) == 1
    assert result["kb"]["deleted"]["sessions"] == 1
    assert not os.path.exists(str(projects / f"{SID}.jsonl"))
    assert _counts(kb_db, SID)["sessions"] == 0


def test_delete_session_purge(projects, kb_db):
    result = sd.delete_session(SID, purge=True, sync_kb=True)
    assert "removed" in result["transcript"]
    assert not os.path.exists(str(projects / f"{SID}.jsonl"))


def test_delete_session_no_kb(projects):
    result = sd.delete_session(SID, sync_kb=False)
    assert result["kb"] is None
    assert not os.path.exists(str(projects / f"{SID}.jsonl"))


def test_delete_session_missing_transcript_kb_only(projects, kb_db):
    """A full id with no transcript still clears KB rows when sync_kb is on."""
    ghost = "bbbb2222-3333-4444-5555-666677778888"
    con = sqlite3.connect(kb_db)
    con.execute("INSERT INTO sessions (id, project, started_at) VALUES (?, 'x', 1)", (ghost,))
    con.commit()
    con.close()
    result = sd.delete_session(ghost, sync_kb=True)
    assert result["transcript"]["missing"] is True
    assert result["kb"]["deleted"]["sessions"] == 1


def test_delete_session_missing_transcript_no_kb_raises(projects):
    with pytest.raises(sd.SessionDeleteError):
        sd.delete_session("cccc3333-4444-5555-6666-777788889999", sync_kb=False)


def test_delete_session_live_refuses(projects, monkeypatch):
    monkeypatch.setattr(sd, "is_open_by_claude", lambda p: (True, "pid 999 live"))
    with pytest.raises(sd.SessionDeleteError):
        sd.delete_session(SID, sync_kb=False)
    assert os.path.exists(str(projects / f"{SID}.jsonl"))


def test_delete_session_force_overrides_live(projects, monkeypatch):
    monkeypatch.setattr(sd, "is_open_by_claude", lambda p: (True, "pid 999 live"))
    result = sd.delete_session(SID, sync_kb=False, force=True)
    assert not os.path.exists(str(projects / f"{SID}.jsonl"))
    assert "moved" in result["transcript"]
