"""observe.extract_topical_tags: drop SHOUTED English, keep real acronyms."""
import observe


def test_drops_shouted_english_keeps_acronyms():
    tags = observe.extract_topical_tags(
        "The source is AHEAD and we ALWAYS RESIZE, e.g. OXI ZIF BEM OperaPostCharge"
    )
    for bad in ("AHEAD", "ALWAYS", "RESIZE", "e.g"):
        assert bad not in tags, f"{bad} should be filtered"
    for good in ("OXI", "ZIF", "BEM", "OperaPostCharge"):
        assert good in tags, f"{good} should be kept"


def test_multiple_fields_and_cap():
    tags = observe.extract_topical_tags(
        "kebab-case-thing", "snake_case_thing", "CamelCaseThing and ZIF"
    )
    assert "kebab-case-thing" in tags
    assert "CamelCaseThing" in tags
    assert "ZIF" in tags
    assert len(tags) <= 5  # extractor caps the tag list (max_tags default)
