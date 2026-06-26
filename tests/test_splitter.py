"""Tests for the lossless memory size-guard splitter (mnemos/splitter.py)."""

import re

from mnemos.splitter import (
    split_content, split_is_lossless, needs_split, _nonblank_lines,
    topic_sort, split_preserves_all_lines,
)


def _cml_blob(n_facts, prefix="F"):
    """Build a realistic CML-style blob: blank-line-separated blocks of facts."""
    blocks = []
    for b in range(n_facts // 5):
        lines = [f"## Topic {b}"]
        for i in range(5):
            lines.append(f"{prefix}:fact {b}.{i}; some detail about subject {b} item {i} with enough text to add length")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def test_short_content_unchanged():
    c = "F:a single small fact; nothing to split"
    assert split_content(c) == [c]


def test_none_and_empty():
    assert split_content(None) == [""]
    assert split_content("") == [""]


def test_needs_split():
    assert not needs_split("x" * 100, threshold=4000)
    assert needs_split("x" * 5000, threshold=4000)


def test_oversized_splits_into_multiple():
    blob = _cml_blob(200)
    assert len(blob) > 4000
    chunks = split_content(blob, threshold=4000, target=2800)
    assert len(chunks) > 1


def test_chunks_respect_target_or_are_single_line():
    blob = _cml_blob(300)
    chunks = split_content(blob, threshold=4000, target=2800)
    for c in chunks:
        # Each chunk is within target, unless it is a single verbatim line
        # that itself exceeds target (we never break inside a line).
        assert len(c) <= 2800 or len(_nonblank_lines(c)) == 1


def test_lossless_property():
    blob = _cml_blob(300)
    chunks = split_content(blob, threshold=4000, target=2800)
    assert split_is_lossless(blob, chunks)
    # Every fact line preserved exactly, in order.
    assert _nonblank_lines(blob) == [
        ln for c in chunks for ln in _nonblank_lines(c)
    ]


def test_single_giant_line_kept_intact():
    giant = "F:" + ("verbatim detail; " * 400)  # one line, > target, no newlines
    assert len(giant) > 2800
    chunks = split_content(giant, threshold=2000, target=2800)
    # A single un-splittable line must survive verbatim in exactly one chunk.
    assert any(giant in c for c in chunks)
    assert split_is_lossless(giant, chunks)


def test_no_fact_lost_across_many_blocks():
    blob = _cml_blob(1000)  # large, many splits
    chunks = split_content(blob, threshold=4000, target=2800)
    assert len(chunks) >= 5
    assert split_is_lossless(blob, chunks)
    assert sum(len(_nonblank_lines(c)) for c in chunks) == len(_nonblank_lines(blob))


def test_dated_sections_blob():
    blob = "\n\n".join(
        f"## Update: 2026-0{m}-15, topic {m}\nF:detail {m}a; xxxxxxxxxx\nF:detail {m}b; yyyyyyyyyy\nW:watch {m}; zzzzzzzz"
        for m in range(1, 9)
    ) * 8
    chunks = split_content(blob, threshold=4000, target=2800)
    assert split_is_lossless(blob, chunks)


def test_split_preserves_all_lines_multiset():
    orig = "F:a\nF:b\n\nF:c"
    assert split_preserves_all_lines(orig, ["F:c", "F:a\nF:b"])      # reordered ok
    assert not split_preserves_all_lines(orig, ["F:a\nF:b"])         # missing c
    assert not split_preserves_all_lines(orig, ["F:a\nF:a\nF:b\nF:c"])  # dup a


def test_topic_sort_groups_and_lossless():
    blob = _cml_blob(300)
    def router(blocks):
        return [int(re.search(r"Topic (\d+)", b).group(1)) % 4 for b in blocks]
    groups = topic_sort(blob, router, threshold=4000, target=2800)
    assert split_preserves_all_lines(blob, [c for _, c in groups])
    for _, c in groups:
        assert len(c) <= 2800 or len(_nonblank_lines(c)) == 1
    assert len(set(t for t, _ in groups)) > 1  # multiple topics emerged


def test_topic_sort_fallback_on_bad_router():
    blob = _cml_blob(300)
    g1 = topic_sort(blob, lambda b: [0], threshold=4000, target=2800)  # wrong length
    assert all(t == 0 for t, _ in g1)
    assert split_preserves_all_lines(blob, [c for _, c in g1])

    def boom(b):
        raise ValueError("no llm")
    g2 = topic_sort(blob, boom, threshold=4000, target=2800)           # router raises
    assert split_preserves_all_lines(blob, [c for _, c in g2])


def test_topic_sort_oversized_topic_subsplit():
    blob = _cml_blob(600)
    groups = topic_sort(blob, lambda b: [0] * len(b), threshold=4000, target=2800)
    assert len(groups) > 1  # the single oversized topic was flat-split further
    for _, c in groups:
        assert len(c) <= 2800 or len(_nonblank_lines(c)) == 1
    assert split_preserves_all_lines(blob, [c for _, c in groups])


def test_topic_sort_trailing_whitespace_tail_no_fallback():
    # A content line carrying trailing whitespace, routed to be the TAIL of its
    # topic, must NOT trip a .strip() that alters it and forces the flat
    # fallback. Regression for the bug the giant-sort workflow surfaced.
    b0 = "## A\nF:a1\nF:a2"
    b1 = "## B\nF:b1\nF:b2 trailing ws   "  # trailing whitespace on last line
    b2 = "## C\nF:c1"
    blob = "\n\n".join([b0, b1, b2])
    groups = topic_sort(blob, lambda blocks: [0, 1, 2], threshold=10, target=100000)
    assert len({t for t, _ in groups}) == 3            # truly sorted, not flat fallback
    assert split_preserves_all_lines(blob, [c for _, c in groups])
    assert any("F:b2 trailing ws   " in c for _, c in groups)  # verbatim, ws intact
