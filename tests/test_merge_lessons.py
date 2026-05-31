"""merge_lessons: durable survival, evidence union, refresh, insert."""
import store_capture


def _cs(tmp_path):
    return store_capture.CaptureStore(str(tmp_path / "kb.db"))


def test_merge_inserts_new(tmp_path):
    cs = _cs(tmp_path)
    n = cs.merge_lessons([
        {"title": "pool-exhaustion", "text": "v1", "tags": ["SQL"],
         "source_session_ids": ["s1", "s2"]},
    ])
    assert n == 1
    rows = cs.list_lessons()
    assert len(rows) == 1
    assert rows[0]["evidence_count"] == 2


def test_merge_refreshes_and_unions_evidence(tmp_path):
    cs = _cs(tmp_path)
    cs.merge_lessons([{"title": "pool-exhaustion", "text": "v1", "tags": ["SQL"],
                       "source_session_ids": ["s1", "s2"]}])
    cs.merge_lessons([{"title": "Pool-Exhaustion", "text": "v2 updated", "tags": ["SQL", "pool"],
                       "source_session_ids": ["s2", "s9"]}])  # case-insensitive key
    rows = cs.list_lessons()
    assert len(rows) == 1, "same title (case-insensitive) must update, not duplicate"
    L = rows[0]
    assert L["text"] == "v2 updated"
    assert sorted(L["source_session_ids"]) == ["s1", "s2", "s9"]
    assert L["evidence_count"] == 3


def test_merge_preserves_undrived_lesson(tmp_path):
    cs = _cs(tmp_path)
    cs.merge_lessons([
        {"title": "durable", "text": "keep me", "tags": [], "source_session_ids": ["s1"]},
        {"title": "transient", "text": "old", "tags": [], "source_session_ids": ["s2"]},
    ])
    # A later run re-derives only one + a brand-new one; the other must survive.
    cs.merge_lessons([
        {"title": "transient", "text": "new", "tags": [], "source_session_ids": ["s3"]},
        {"title": "fresh", "text": "f", "tags": [], "source_session_ids": ["s4"]},
    ])
    titles = sorted(r["title"] for r in cs.list_lessons())
    assert titles == ["durable", "fresh", "transient"]
    durable = next(r for r in cs.list_lessons() if r["title"] == "durable")
    assert durable["text"] == "keep me"
