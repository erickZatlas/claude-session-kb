"""Recall ranking helpers in app.py: decay, query-file extraction, knowledge
ranking (threshold, gap-trim, project boost). app.store/app.embedder are
monkeypatched with light fakes so no model or DB is needed."""
import time
import types

import numpy as np

import app


def test_decay_bounds():
    assert app._decay(0) == 1.0
    assert app._decay(None) == 1.0
    assert app._decay((time.time() + 1000) * 1000) == 1.0      # future → 1.0
    one_hl = (time.time() - app.HALFLIFE_DAYS * 86400) * 1000  # one half-life ago
    assert abs(app._decay(one_hl, app.HALFLIFE_DAYS) - 0.5) < 0.05


def test_extract_query_files_dedup_and_filter():
    out = app._extract_query_files("edit app.py and App.PY then store.py, ignore notes.txt")
    assert "app.py" in out
    assert sum(1 for x in out if x.lower() == "app.py") == 1   # case-insensitive dedup
    assert "store.py" in out
    assert all(not x.endswith(".txt") for x in out)            # .txt not in extension set


def _patch(monkeypatch, recs, index):
    monkeypatch.setattr(app, "store", types.SimpleNamespace(knowledge_records=lambda: recs))
    monkeypatch.setattr(app, "embedder", types.SimpleNamespace(index=index))


def test_knowledge_threshold_and_gap_trim(monkeypatch):
    recs = [
        {"id": "k1", "kind": "lesson", "title": "strong", "text": "", "concepts": [], "evidenceCount": 3},
        {"id": "k2", "kind": "lesson", "title": "weak", "text": "", "concepts": []},
        {"id": "k3", "kind": "lesson", "title": "noise", "text": "", "concepts": []},
    ]
    _patch(monkeypatch, recs, {"k1": 0, "k2": 1, "k3": 2})
    sims = np.array([0.80, 0.50, 0.20])  # k3 below floor; k2 within floor but >gap below top
    out = app._recall_knowledge(sims, min_score=0.28, limit=4,
                                project="all", boost_project="")
    assert [o["title"] for o in out] == ["strong"]


def test_knowledge_project_boost(monkeypatch):
    recs = [
        {"id": "m1", "kind": "memory", "title": "mine", "text": "", "concepts": [],
         "memType": "project", "project": "claude-kb", "sourcePath": ""},
        {"id": "m2", "kind": "memory", "title": "other", "text": "", "concepts": [],
         "memType": "project", "project": "zatlas", "sourcePath": ""},
    ]
    _patch(monkeypatch, recs, {"m1": 0, "m2": 1})
    sims = np.array([0.50, 0.52])  # m2 slightly higher raw
    # Without boost m2 wins; with +0.05 boost on claude-kb, m1 (0.55) wins.
    out = app._recall_knowledge(sims, min_score=0.28, limit=4, boost_project="claude-kb")
    assert out[0]["title"] == "mine"


def test_knowledge_project_scoping(monkeypatch):
    recs = [
        {"id": "m1", "kind": "memory", "title": "kb", "text": "", "concepts": [],
         "memType": "feedback", "project": "claude-kb", "sourcePath": ""},
        {"id": "m2", "kind": "memory", "title": "zat", "text": "", "concepts": [],
         "memType": "feedback", "project": "zatlas", "sourcePath": ""},
        {"id": "g1", "kind": "memory", "title": "glob", "text": "", "concepts": [],
         "memType": "user", "project": "global", "sourcePath": ""},
    ]
    _patch(monkeypatch, recs, {"m1": 0, "m2": 1, "g1": 2})
    sims = np.array([0.60, 0.60, 0.60])
    out = app._recall_knowledge(sims, min_score=0.28, limit=10, project="claude-kb")
    titles = {o["title"] for o in out}
    assert "zat" not in titles            # other project's memory excluded
    assert {"kb", "glob"} <= titles       # this project + global kept
