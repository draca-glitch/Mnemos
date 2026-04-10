"""
FTS5 query builder for Mnemos.

Cleans user queries and builds FTS5 MATCH expressions. Supports AND-default
mode (high precision) with OR-fallback for recall when AND yields no hits.
Handles Swedish stemming via simple suffix stripping.
"""

import re

# Common stop words (English + Swedish), removed before FTS query
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with",
    "och", "att", "det", "som", "en", "är", "på", "för", "med", "har",
    "till", "den", "av", "i", "om", "ett",
}

# Swedish suffixes - strip to allow partial matching
SWEDISH_SUFFIXES = ["arna", "erna", "orna", "arnas", "ernas", "ornas",
                    "ar", "er", "or", "en", "et", "ens", "ets", "s"]


def _strip_swedish_suffix(word):
    """Strip common Swedish noun/verb suffixes to enable stem matching."""
    if len(word) < 5:
        return word
    for suffix in SWEDISH_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def clean_fts_query(raw: str, mode: str = "AND") -> str:
    """Clean and tokenize a query for FTS5 MATCH.

    mode='AND' produces high-precision matches; 'OR' produces high-recall.
    Each token becomes a quoted exact match OR a stem prefix match.
    """
    if not raw or not raw.strip():
        return ""

    # Tokenize: alphanumeric + Unicode word chars
    tokens = re.findall(r"\w+", raw.lower(), re.UNICODE)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]

    if not tokens:
        return ""

    parts = []
    for t in tokens:
        stem = _strip_swedish_suffix(t)
        if stem != t and len(stem) >= 3:
            parts.append(f'("{t}" OR {stem}*)')
        else:
            parts.append(f'"{t}"')

    if mode == "AND":
        return " AND ".join(parts)
    else:
        return " OR ".join(parts)


def fts_dedup(store, content: str, top_n: int = 5, threshold: float = 0.6):
    """Find existing memories with high keyword overlap with new content.

    Returns list of {id, content, project, similarity} dicts.
    Used by core dedup before storing a new memory.
    """
    tokens = set(t for t in re.findall(r"\w+", content.lower(), re.UNICODE)
                 if t not in STOP_WORDS and len(t) > 3)
    if not tokens:
        return []

    fts_query = clean_fts_query(content, mode="OR")
    if not fts_query:
        return []

    candidate_ids = store.search_fts(fts_query, limit=top_n * 3, and_mode=False)
    if not candidate_ids:
        return []

    memories = store.get_memories_by_ids(candidate_ids[:top_n * 3])
    results = []
    for mid, mem in memories.items():
        mem_tokens = set(
            t for t in re.findall(r"\w+", (mem.content or "").lower(), re.UNICODE)
            if t not in STOP_WORDS and len(t) > 3
        )
        if not mem_tokens:
            continue
        overlap = len(tokens & mem_tokens)
        sim = overlap / max(len(tokens), len(mem_tokens))
        if sim >= threshold:
            results.append({
                "id": mid,
                "content": mem.content,
                "project": mem.project,
                "similarity": round(sim, 3),
            })
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:top_n]
