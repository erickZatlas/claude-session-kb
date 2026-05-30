"""Store projects observations + lessons + memory_facts into one corpus."""
import store
import store_capture


def _seed(tmp_path):
    db = str(tmp_path / "kb.db")
    cs = store_capture.CaptureStore(db)
    cs.session_start("s1", "claude-kb", "/home/erick/dev/claude-kb", None)
    cs.replace_observations("s1", [
        {"type": "discovery", "title": "obs one", "text": "body", "tags": ["OXI"]},
    ])
    cs.merge_lessons([
        {"title": "lesson-a", "text": "L body", "tags": ["t"], "source_session_ids": ["s1"]},
    ])
    cs.upsert_memory_fact({
        "id": "claude-kb::m1", "project": "claude-kb", "source_path": "/x.md",
        "name": "m1", "mem_type": "feedback", "description": "d", "text": "mem body",
        "tags": ["feedback"], "content_hash": "abc123def", "mtime": 123,
    })
    return db


def test_projects_all_three_kinds(tmp_path):
    st = store.Store(db_path=_seed(tmp_path))
    st.reload()
    kinds = {r["kind"] for r in st.records}
    assert {"observation", "lesson", "memory"} <= kinds


def test_content_hashed_ids(tmp_path):
    st = store.Store(db_path=_seed(tmp_path))
    st.reload()
    lesson = next(r for r in st.records if r["kind"] == "lesson")
    mem = next(r for r in st.records if r["kind"] == "memory")
    assert lesson["id"].startswith("lesson-") and len(lesson["id"]) > len("lesson-")
    assert mem["id"] == "mem-abc123def"   # mem-<content_hash>


def test_kind_filters_and_knowledge_records(tmp_path):
    st = store.Store(db_path=_seed(tmp_path))
    st.reload()
    assert all(r["kind"] == "lesson" for r in st.candidates("all", "lessons"))
    assert all(r["kind"] == "memory" for r in st.candidates("all", "memory"))
    assert all(r["kind"] == "observation" for r in st.candidates("all", "observations"))
    assert {r["kind"] for r in st.knowledge_records()} <= {"lesson", "memory"}


def test_meta_counts(tmp_path):
    st = store.Store(db_path=_seed(tmp_path))
    st.reload()
    counts = st.meta()["counts"]
    assert counts["observations"] == 1
    assert counts["lessons"] == 1
    assert counts["memoryFacts"] == 1
