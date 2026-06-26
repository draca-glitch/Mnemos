"""
Memory size-guard splitter for Mnemos.

Keeps individual memories atomic by splitting oversized content into smaller
chunks, losslessly: every non-blank original line lands in exactly one chunk,
in original order, nothing paraphrased or dropped. CML memories (one fact per
line, blank-line-separated blocks) split cleanly along block and line
boundaries.

Design note (load-bearing principle: a memory system should store and retrieve,
not think endlessly): the bulk splitter is pure mechanical text work with no
LLM and no DB access. LLM-assisted topical clustering for giant or prose
blobs is a separate Phase 1 concern, layered on top, used only on the handful
of very large memories. This module stays dependency-free (stdlib only) so the
store hot path can import it without pulling in consolidation or numpy.
"""

import os

SPLIT_THRESHOLD = int(os.environ.get("MNEMOS_SPLIT_THRESHOLD", "4000"))
SPLIT_TARGET = int(os.environ.get("MNEMOS_SPLIT_TARGET", "2800"))


def split_enabled():
    """Whether auto-splitting is on. Default on; set MNEMOS_SPLIT_ENABLED=0 to disable."""
    return os.environ.get("MNEMOS_SPLIT_ENABLED", "1").lower() not in ("0", "false", "no", "")


def _blocks(content):
    """Group lines into blocks at blank-line boundaries.

    Each block is a list of consecutive non-blank lines (a paragraph or a CML
    block such as a heading plus the facts under it). Blank lines are treated
    purely as separators and are not preserved as content.
    """
    blocks = []
    cur = []
    for line in content.split("\n"):
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def split_content(content, threshold=None, target=None):
    """Split content into atomic chunks, losslessly.

    Returns a list of chunk strings. If content is within `threshold` (or empty
    or None), returns a single-element list with the content unchanged.
    Otherwise it packs whole blocks (then, for an oversized single block, whole
    lines) into chunks of at most `target` characters. It never breaks inside a
    line, so every fact stays verbatim and every non-blank line appears in
    exactly one chunk, in original order. Blank separator lines are normalized.

    Verify the guarantee with split_is_lossless(original, chunks).
    """
    threshold = SPLIT_THRESHOLD if threshold is None else threshold
    target = SPLIT_TARGET if target is None else target
    if not content or len(content) <= threshold:
        return [content if content is not None else ""]

    chunks = []
    cur = []          # list of lines accumulated for the current chunk
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            # Drop only leading/trailing blank separator lines; content lines
            # are kept verbatim so the lossless invariant holds byte for byte.
            lines = cur
            while lines and lines[0].strip() == "":
                lines.pop(0)
            while lines and lines[-1].strip() == "":
                lines.pop()
            if lines:
                chunks.append("\n".join(lines))
            cur = []
            cur_len = 0

    for block in _blocks(content):
        block_text = "\n".join(block)
        block_len = len(block_text) + 2  # account for the blank-line separator

        if block_len > target:
            # Oversized single block: flush what we have, then pack its lines.
            flush()
            for line in block:
                add = len(line) + 1
                if cur and cur_len + add > target:
                    flush()
                cur.append(line)
                cur_len += add
            flush()
            continue

        if cur and cur_len + block_len > target:
            flush()
        if cur:
            cur.append("")  # blank line between blocks within a chunk
            cur_len += 1
        cur.extend(block)
        cur_len += block_len

    flush()
    return chunks or [content]


def _nonblank_lines(text):
    return [ln for ln in (text or "").split("\n") if ln.strip()]


def split_is_lossless(original, chunks):
    """True iff every non-blank line of `original` appears across `chunks`
    exactly once, in the original order. This is the safety invariant: the
    splitter must never drop, duplicate, reorder, or paraphrase a fact.
    """
    orig = _nonblank_lines(original)
    rebuilt = []
    for c in chunks:
        rebuilt.extend(_nonblank_lines(c))
    return orig == rebuilt


def needs_split(content, threshold=None):
    threshold = SPLIT_THRESHOLD if threshold is None else threshold
    return bool(content) and len(content) > threshold


def split_preserves_all_lines(original, chunks):
    """True iff every non-blank line of `original` appears across `chunks`
    exactly once, ignoring order. This is the lossless invariant for a
    TOPIC-SORTED split, which deliberately reorders blocks (so the stricter
    in-order split_is_lossless does not apply). Multiset equality: nothing
    dropped, nothing duplicated, nothing paraphrased.
    """
    from collections import Counter
    return Counter(_nonblank_lines(original)) == Counter(
        ln for c in chunks for ln in _nonblank_lines(c)
    )


def topic_sort(content, propose_fn, threshold=None, target=None):
    """Sort content into topically coherent atomic chunks, losslessly.

    Splits content into CML blocks, then calls propose_fn(blocks) where
    `blocks` is a list of block strings. The router (typically an LLM such as
    Opus) returns a list of topic ids, one per block, in block order. The
    router ONLY routes; it never rewrites content. Blocks are then grouped by
    topic (preserving first-seen topic order and within-topic block order),
    and each topic becomes one chunk, flat-split further if it still exceeds
    target. Returns a list of (topic_id, chunk_text).

    Safety: if the router is unavailable, returns a malformed assignment, or
    the result is not a perfect lossless cover of the original lines, this
    falls back to the plain order-preserving split_content. A fact is never
    dropped, duplicated, or paraphrased, verified by split_preserves_all_lines.
    """
    threshold = SPLIT_THRESHOLD if threshold is None else threshold
    target = SPLIT_TARGET if target is None else target
    if not content or len(content) <= threshold:
        return [(0, content if content is not None else "")]

    block_lists = _blocks(content)
    blocks = ["\n".join(b) for b in block_lists]

    assign = None
    try:
        assign = propose_fn(blocks)
    except Exception:
        assign = None

    def _flat_fallback():
        return [(0, c) for c in split_content(content, threshold, target)]

    if not (isinstance(assign, list) and len(assign) == len(blocks)
            and all(isinstance(a, int) for a in assign)):
        return _flat_fallback()

    order = []
    groups = {}
    for blk, tid in zip(blocks, assign):
        if tid not in groups:
            groups[tid] = []
            order.append(tid)
        groups[tid].append(blk)

    result = []
    for tid in order:
        # Join blocks verbatim. NO .strip(): stripping would alter a content
        # line that carries trailing whitespace if it lands at the tail of a
        # topic, breaking the lossless multiset check and silently forcing the
        # flat fallback. Blocks already have no blank edges (see _blocks).
        text = "\n\n".join(groups[tid])
        if len(text) <= target:
            if text:
                result.append((tid, text))
        else:
            for part in split_content(text, threshold=target, target=target):
                if part:
                    result.append((tid, part))

    if not result or not split_preserves_all_lines(content, [c for _, c in result]):
        return _flat_fallback()
    return result
