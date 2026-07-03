"""Mechanical merge: union of atomic lines, dedup by bidirectional entailment.

The zero-LLM replacement for the phase-2 MERGE step (v10.17.0). Selection,
never generation: every output line is an input line verbatim, so fact
preservation is provable by construction. A line is dropped only when it is
a bidirectional-entailment duplicate (both directions >= MECH_MERGE_TAU) of
a kept line; the newer memory's phrasing survives (recency semantics, an
update supersedes a stale wording of the same fact).

Evidence: benchmarks/merge-bench. 24/25 exact recovery on ground-truth
overlap splits; the single false duplicate was a short enumerated list
line (hence the exact-only floor for short lines); the single semantic
false-duplicate on production pairs scored 0.851 (hence tau 0.90, where
every true duplicate observed scored ~1.0).

Guards, all of which fail toward keeping too much rather than losing facts:
- exact-match fast path (no NLI cost for identical lines)
- lexical prefilter: lines sharing no content words cannot be duplicates;
  a false skip costs compression, never information
- short lines (< MECH_MERGE_MIN_LINE_CHARS) dedup by exact match only
- NLI unavailable: return None, the caller skips the cluster loudly
  (no silent fallback to the LLM in mechanical mode)
"""

import re

from .. import nli
from ..constants import MECH_MERGE_TAU, MECH_MERGE_MIN_LINE_CHARS
from ..splitter import explode_cml_chain

_WORD = re.compile(r"[a-za-åäö0-9_./#-]+", re.IGNORECASE)
_PREFILTER_JACCARD = 0.10


def _lines_of(content):
    text = explode_cml_chain(content)
    out = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln and not ln.startswith("---"):
            out.append(ln)
    return out


def _words(line):
    return {w.lower() for w in _WORD.findall(line) if len(w) > 2}


def _prefilter_pass(la, lb):
    wa, wb = _words(la), _words(lb)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= _PREFILTER_JACCARD


def mechanical_merge_cluster(cluster_ids, mem_by_id, tau=None,
                             min_line_chars=None):
    """Merge a cluster into one CML text without an LLM.

    Same call surface as merge_cluster (returns the merged content string
    or None). None means the merge cannot run (NLI backend unavailable or
    fewer than two members with content); the caller skips the cluster.
    The result flows through apply_merge unchanged, so the size guard,
    lossless splitter, atomic transaction and lineage row are shared with
    the LLM path.
    """
    if tau is None:
        tau = MECH_MERGE_TAU
    if min_line_chars is None:
        min_line_chars = MECH_MERGE_MIN_LINE_CHARS
    if not nli.is_available():
        return None

    members = [mem_by_id[mid] for mid in cluster_ids if mem_by_id.get(mid)]
    members = [m for m in members if (m.get("content") or "").strip()]
    if len(members) < 2:
        return None
    members.sort(key=lambda m: m.get("created_at") or "")

    pool = []
    for m in members:
        for ln in _lines_of(m["content"]):
            pool.append(ln)

    kept = []
    # Newest-first so the newer phrasing is kept and older duplicates drop.
    for cand in reversed(pool):
        duplicate = False
        for k in kept:
            if k == cand:
                duplicate = True
                break
            if len(cand) < min_line_chars or len(k) < min_line_chars:
                continue
            if not _prefilter_pass(cand, k):
                continue
            score = nli.bidirectional_entailment(cand, k)
            if score is not None and score >= tau:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)

    kept.reverse()
    return "\n".join(kept) if kept else None
