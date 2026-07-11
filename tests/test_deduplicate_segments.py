import pytest

from textplease.utils.deduplicate_segments import deduplicate_segments


def _seg(text: str, start: str = "00:00:00.000", end: str = "00:00:01.000") -> dict[str, str]:
    return {"text": text, "start_time": start, "end_time": end}


def test_removes_boundary_overlap():
    segs = [_seg("the quick brown fox"), _seg("brown fox jumps over")]
    out = deduplicate_segments(segs, overlap_words=15)
    assert [s["text"] for s in out] == ["the quick brown fox", "jumps over"]


def test_no_overlap_left_unchanged():
    segs = [_seg("alpha beta"), _seg("gamma delta")]
    out = deduplicate_segments(segs, overlap_words=5)
    assert [s["text"] for s in out] == ["alpha beta", "gamma delta"]


def test_empty_list_returns_empty():
    assert deduplicate_segments([]) == []


def test_missing_key_raises():
    with pytest.raises(KeyError):
        deduplicate_segments([_seg("hello"), {"text": "world"}])


def test_input_not_mutated():
    segs = [_seg("a b c"), _seg("b c d")]
    deduplicate_segments(segs, overlap_words=5)
    assert segs[1]["text"] == "b c d"
