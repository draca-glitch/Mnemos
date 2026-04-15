"""
Cross-encoder reranker for Mnemos.

Uses Jina Reranker v2 (multilingual). Scores query↔document pairs for
high-precision reranking after first-stage hybrid retrieval. ~50ms per
batch of 20 documents on CPU.
"""

from .constants import RERANKER_MODEL, FASTEMBED_CACHE

_instance = None


def _get_reranker():
    global _instance
    if _instance is None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _instance = TextCrossEncoder(model_name=RERANKER_MODEL, cache_dir=FASTEMBED_CACHE)
        except ImportError:
            raise ImportError(
                "Reranker requires fastembed[rerank]. Install with: "
                "pip install 'fastembed[rerank]'"
            )
    return _instance


def rerank(query: str, documents: list) -> list:
    """Rerank documents by cross-encoder relevance to query.

    documents: list of dicts with at least {"text": ..., "id": ...}
    Returns the same list, sorted by score (highest first), with `_rerank_score` set.
    """
    if not documents:
        return []
    try:
        model = _get_reranker()
        texts = [d.get("text", "") for d in documents]
        scores = list(model.rerank(query, texts))
    except Exception as e:
        import sys
        print(f"Reranker error: {e}", file=sys.stderr)
        return documents

    for doc, score in zip(documents, scores):
        doc["_rerank_score"] = float(score)
    return sorted(documents, key=lambda d: d.get("_rerank_score", 0), reverse=True)


def rrf_merge(*ranked_lists, k=60):
    """Reciprocal Rank Fusion of multiple ranked ID lists.

    score(id) = sum(1 / (k + rank_in_list_i)) for each list
    """
    scores = {}
    for ids in ranked_lists:
        for rank, mid in enumerate(ids):
            scores[mid] = scores.get(mid, 0) + 1.0 / (k + rank + 1)
    return [mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
