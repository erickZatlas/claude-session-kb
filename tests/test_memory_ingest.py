"""memory_ingest: frontmatter parsing, project resolution, fact building."""
import memory_ingest as mi


def test_parse_frontmatter_wellformed():
    raw = ('---\nname: foo\ndescription: "bar baz"\n'
           'metadata:\n  node_type: memory\n  type: feedback\n---\n\nBody here.')
    meta, body = mi._parse_frontmatter(raw)
    assert meta["name"] == "foo"
    assert meta["description"] == "bar baz"          # quotes stripped
    assert meta["metadata"]["type"] == "feedback"    # nested block
    assert body == "Body here."


def test_parse_frontmatter_missing():
    meta, body = mi._parse_frontmatter("just a plain body, no fences")
    assert meta == {}
    assert body == "just a plain body, no fences"


def test_project_for_dir_suffix_match():
    known = ["claude-kb", "zatlas-pms-middleware", "zatlas-mono"]
    assert mi._project_for_dir("-home-erick-dev-claude-kb", known) == "claude-kb"
    assert mi._project_for_dir(
        "-home-erick-dev-projects-zatlas-zatlas-pms-middleware", known
    ) == "zatlas-pms-middleware"
    assert mi._project_for_dir("-home-erick", known) == "global"  # no match → global


def test_fact_from_file(tmp_path):
    d = tmp_path / "-home-erick-dev-claude-kb" / "memory"
    d.mkdir(parents=True)
    f = d / "feedback_thing.md"
    f.write_text('---\nname: feedback_thing\ndescription: "d"\n'
                 'metadata:\n  type: feedback\n---\n\n'
                 'We ALWAYS RESIZE, e.g. use OXI and BEM here.')
    fact = mi._fact_from_file(str(f), ["claude-kb"])
    assert fact["id"] == "claude-kb::feedback_thing"
    assert fact["project"] == "claude-kb"
    assert fact["mem_type"] == "feedback"
    assert fact["name"] == "feedback_thing"
    # mem_type prepended; filename slug NOT used as a tag; shouted English dropped
    assert "feedback" in fact["tags"]
    assert "feedback_thing" not in fact["tags"]
    assert "ALWAYS" not in fact["tags"] and "RESIZE" not in fact["tags"]
    assert "OXI" in fact["tags"] and "BEM" in fact["tags"]
    assert len(fact["content_hash"]) == 40  # sha1 hex


def test_fact_from_file_defaults_unknown_type(tmp_path):
    d = tmp_path / "-home-erick" / "memory"
    d.mkdir(parents=True)
    f = d / "note.md"
    f.write_text("---\nname: note\n---\n\nplain fact body")
    fact = mi._fact_from_file(str(f), ["claude-kb"])
    assert fact["mem_type"] == "reference"   # unknown/missing type → reference
    assert fact["project"] == "global"
