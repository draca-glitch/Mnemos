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
import re

# Aligned to the e5-large embedder window (~512 tokens, ~2000-2500 chars):
# content past that is truncated out of the embedding vector, so larger targets
# silently hurt vector recall. 2400 keeps a whole chunk inside the window.
SPLIT_THRESHOLD = int(os.environ.get("MNEMOS_SPLIT_THRESHOLD", "4000"))
SPLIT_TARGET = int(os.environ.get("MNEMOS_SPLIT_TARGET", "2400"))

_SENT_BOUNDARY = re.compile(r"(?<=[.!?;])\s+")


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


def split_content(content, threshold=None, target=None, hard=False):
    """Split content into atomic chunks, losslessly.

    Returns a list of chunk strings. If content is within `threshold` (or empty
    or None), returns a single-element list with the content unchanged.
    Otherwise it packs whole blocks (then, for an oversized single block, whole
    lines) into chunks of at most `target` characters. It never breaks inside a
    line, so every fact stays verbatim and every non-blank line appears in
    exactly one chunk, in original order. Blank separator lines are normalized.

    With hard=True, a single line that itself exceeds `target` (the only thing
    the line splitter cannot otherwise break) is split on sentence boundaries
    as a last resort. That trades strict line-losslessness for sentence-level
    losslessness on that line; verify hard splits with
    split_preserves_all_sentences instead of split_is_lossless.

    Verify a normal (non-hard) split with split_is_lossless(original, chunks).
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
            # In hard mode, a single line longer than target is split on
            # sentence boundaries (last resort, sentence-level lossless).
            flush()
            for line in block:
                segs = _split_long_line(line, target) if hard else [line]
                for seg in segs:
                    add = len(seg) + 1
                    if cur and cur_len + add > target:
                        flush()
                    cur.append(seg)
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


# A new CML statement begins at a canonical prefix (F:/D:/C:/L:/P:/W:/R:) not
# immediately followed by a path separator (so a Windows drive letter "C:\"
# or a "D:/path" URL is not a candidate). A candidate only counts as a real
# boundary when it sits at the start of the blob or after a statement
# terminator (';' or '.'): CML chains statements with ';', so a prefix letter
# appearing after a mere space, '(', '"' or other punctuation is interior
# content ("free space on C: drive", "(P:prefer HE-AAC)"), and splitting
# there shreds the fact. The loss-guard cannot catch that, it only proves
# content survives, not that split placement is sane.
_CML_STATEMENT_CANDIDATE = re.compile(r"[FDCLPWR]:(?![\\/])")


def _statement_starts(content):
    """Offsets where a new CML statement genuinely begins."""
    starts = []
    for m in _CML_STATEMENT_CANDIDATE.finditer(content):
        pos = m.start()
        if pos == 0:
            starts.append(pos)
            continue
        before = content[:pos].rstrip()
        if before and before[-1] in ";.":
            starts.append(pos)
    return starts


def _sep_free(text):
    """Content with separators (whitespace and ';') stripped. Two CML
    renderings of the same facts differ only in separators, so this is the
    invariant the mechanical exploder must preserve. Periods are content,
    not separators: stripping them would blind the guard to a split that
    relocates or drops one."""
    return re.sub(r"[\s;]+", "", text or "")


def explode_cml_chain(content):
    """Reformat one single-line, prefix-chained CML blob into multi-line
    one-fact-per-line CML, mechanically (stdlib only, no LLM, no DB).

    Splits BEFORE each canonical prefix (F:/D:/C:/L:/P:/W:/R:) that starts a new
    statement, i.e. one at the start of the blob or right after a ';' or '.'
    terminator. A ';' not followed by such a prefix is intra-fact sub-clause
    chaining and stays put, and a prefix glued to interior punctuation or a
    bare drive letter ("on C: drive") is interior content, so a single fact
    is never shredded.

    Loss-guarded: the only characters that may change are separators. If the
    separator-free content of the result does not equal the input's, or the
    input is empty, already multi-line, or yields fewer than two statements, the
    input is returned unchanged. It never drops or paraphrases a fact; an
    ambiguous blob is left as-is for the LLM path.
    """
    if not content or "\n" in content:
        return content
    cuts = [s for s in _statement_starts(content) if s > 0]
    if not cuts:
        return content
    bounds = [0] + cuts + [len(content)]
    parts = []
    for a, b in zip(bounds, bounds[1:]):
        seg = re.sub(r";+\s*$", "", content[a:b].strip()).strip()
        if seg:
            parts.append(seg)
    if len(parts) < 2:
        return content
    exploded = "\n".join(parts)
    if _sep_free(exploded) != _sep_free(content):
        return content
    return exploded


def explode_cml_lines(text):
    """Apply explode_cml_chain to each physical line of a multi-line text.

    The chain exploder is deliberately a no-op on multi-line input, so a
    size-split child that inherited a packed multi-statement line from its
    parent would stay un-atomic forever without this per-line pass. Same
    loss guard per line; a line that cannot be safely exploded is kept."""
    if not text or "\n" not in text:
        return explode_cml_chain(text) if text else text
    return "\n".join(explode_cml_chain(ln) for ln in text.split("\n"))


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


def _split_long_line(line, target):
    """Last-resort split of a single over-target line on sentence/clause
    boundaries (. ! ? ;), packing sentences into pieces of at most `target`
    chars. Sentences are kept verbatim; a lone sentence over target is emitted
    whole. Sentence-level lossless: the sentence sequence is preserved (only
    inter-sentence whitespace is normalized).
    """
    if len(line) <= target:
        return [line]
    pieces, cur, cur_len = [], [], 0
    for s in _SENT_BOUNDARY.split(line):
        add = len(s) + 1
        if cur and cur_len + add > target:
            pieces.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += add
    if cur:
        pieces.append(" ".join(cur))
    return pieces or [line]


def _sentences(text):
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in _SENT_BOUNDARY.split(line):
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                out.append(s)
    return out


def split_preserves_all_sentences(original, chunks):
    """Sentence-level lossless check for HARD splits (which may break an
    over-target single line on sentence boundaries, so the strict line check
    does not apply). Multiset of whitespace-normalized sentences must match:
    nothing dropped, duplicated, or paraphrased.
    """
    from collections import Counter
    return Counter(_sentences(original)) == Counter(
        s for c in chunks for s in _sentences(c)
    )


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
