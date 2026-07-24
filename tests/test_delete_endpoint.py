"""Integration tests for DELETE /api/sessions/{session_id}.

Monkeypatches session_delete's paths to a tmp sandbox and stubs the store /
embedder side effects so the endpoint never touches the real ~/.claude-kb.
"""
from __future__ import annotations

import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

import app as app_mod
import session_delete as sd
import store_capture

SID = "f3cdc10e-8cf0-428a-b25e-329ce6346081"


@pytest.fixture
def client(tmp_path, monkeypatch):
    proj = tmp_path / "projects" / "-home-erick-dev-claude-kb"
    proj.mkdir(parents=True)
    (proj / f"{SID}.jsonl").write_text('{"type":"user"}\n')
    old = 1700000000
    os.utime(proj / f"{SID}.jsonl", (old, old))

    db = tmp_path / "data.db"
    con = sqlite3.connect(db)
    con.executescript(store_capture._SCHEMA)
    con.execute("INSERT INTO sessions (id, project, started_at) VALUES (?, 'claude-kb', ?)",
                (SID, old))
    con.execute("INSERT INTO prompts (session_id, ts, text) VALUES (?, ?, 'hi')", (SID, old))
    con.commit()
    con.close()

    monkeypatch.setattr(sd, "PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(sd, "TRASH_DIR", str(tmp_path / "trash"))
    monkeypatch.setattr(sd, "KB_DB", str(db))

    # Stub the post-delete side effects so we don't touch the real store/cache.
    reloaded = {"count": 0}
    monkeypatch.setattr(app_mod.store, "reload", lambda: reloaded.__setitem__("count", reloaded["count"] + 1))
    monkeypatch.setattr(app_mod.store, "rec_by_id", {})
    monkeypatch.setattr(app_mod.embedder, "compact", lambda live: 0)

    c = TestClient(app_mod.app)
    c._proj = proj
    c._db = db
    c._reloaded = reloaded
    return c


def _session_rows(db):
    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", (SID,)).fetchone()[0]
    finally:
        con.close()


def test_delete_trashes_and_removes_kb_rows(client):
    r = client.delete(f"/api/sessions/{SID}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["transcript"] == "trashed"
    assert body["kbDeleted"]["sessions"] == 1

    assert not os.path.exists(str(client._proj / f"{SID}.jsonl"))
    assert _session_rows(client._db) == 0
    assert client._reloaded["count"] == 1


def test_delete_purge(client):
    r = client.delete(f"/api/sessions/{SID}?purge=true")
    assert r.status_code == 200, r.text
    assert r.json()["transcript"] == "purged"
    # Purged files don't land in trash.
    assert not (client._proj.parent.parent / "trash").exists()


def test_delete_live_session_conflicts(client, monkeypatch):
    monkeypatch.setattr(sd, "is_open_by_claude", lambda p: (True, "pid 999 has this session id in its argv"))
    r = client.delete(f"/api/sessions/{SID}")
    assert r.status_code == 409
    assert "looks live" in r.json()["detail"]
    # Nothing removed.
    assert os.path.exists(str(client._proj / f"{SID}.jsonl"))
    assert _session_rows(client._db) == 1


def test_delete_force_overrides_live(client, monkeypatch):
    monkeypatch.setattr(sd, "is_open_by_claude", lambda p: (True, "live"))
    r = client.delete(f"/api/sessions/{SID}?force=true")
    assert r.status_code == 200, r.text
    assert not os.path.exists(str(client._proj / f"{SID}.jsonl"))


def test_delete_short_prefix_is_bad_request(client):
    r = client.delete("/api/sessions/f3cd")
    assert r.status_code == 400
