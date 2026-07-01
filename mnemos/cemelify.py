"""
Cemelify: rewrite content into compact CML notation via the configured LLM.

Used in two places:

  1. Opt-in store-time hook: when MNEMOS_CEMELIFY_ON_IMPORT=1,
     core.store_memory() pipes content through cemelify() before
     persistence, so memories land already in CML form.
  2. Phase 0.5 of the Nyx cycle (on by default when an LLM is configured):
     scans active memories that either don't start with a CML prefix
     (F:/D:/C:/L:/P:/W:/R:) or are longer than 800 chars, and rewrites
     each in place.

The LLM call goes through mnemos.consolidation.llm.chat(), so all the env
routing (MNEMOS_LLM_API_URL / _API_KEY / _MODEL, per-phase overrides, the
v10.4.0 OpenAI default) is inherited.

On any failure (LLM unconfigured, network error, empty response, exception)
cemelify() returns the original content. That contract is the design point:
opting into MNEMOS_CEMELIFY_ON_IMPORT must not turn LLM into a hard
dependency for store_memory().
"""

from typing import Optional


CML_CEMELIFY_SYSTEM = (
    "You are a CML (Compressed Memory Language) cemelifier. Rewrite the "
    "user's content into compact CML: one fact per line, each line starting "
    "with a canonical prefix. When the source holds several distinct facts, "
    "emit several lines separated by newlines; never collapse them onto one "
    "line. Within a single line you may chain tightly related sub-clauses "
    "with ';'.\n"
    "  F: Fact (verifiable atomic claim)\n"
    "  D: Decision\n"
    "  C: Contact\n"
    "  L: Learning\n"
    "  P: Preference\n"
    "  W: Warning\n"
    "  R: Restriction (hard rule or limit)\n"
    "Preserve every concrete fact, number, name, and identifier from the "
    "source. Drop filler, redundant context, and meta-commentary. If the "
    "source already starts with a CML prefix, return it unchanged. Reply "
    "with the CML lines only, no explanation, no quoting."
)


def cemelify(content: str, max_tokens: int = 512) -> str:
    """Rewrite content into CML form via the configured LLM.

    Returns the cemelified string on success, or the original content on any
    failure. Never raises.
    """
    if not content or not content.strip():
        return content

    # Lazy import to keep cemelify safely importable in odd test contexts and
    # to mirror the lazy-import pattern already used in core.py.
    try:
        from .consolidation.llm import chat
    except Exception:
        return content

    messages = [
        {"role": "system", "content": CML_CEMELIFY_SYSTEM},
        {"role": "user", "content": content},
    ]
    try:
        # Cemelify items are small (single memory rewrite, output ~512 tokens),
        # so cap the per-call read timeout tighter than the consolidation
        # default. One slow candidate must not consume the read window meant
        # for hierarchical-merge prompts. The LLM_WALL_BUDGET ceiling in
        # llm.py still applies across retries.
        response = chat(messages, max_tokens=max_tokens, temperature=0.2, timeout=90)
    except Exception:
        return content

    if not response:
        return content
    cemelified = response.strip()
    if not cemelified:
        return content
    return cemelified


def _needs_cemelify(content: Optional[str]) -> bool:
    """Phase 0.5 filter: candidate only when content lacks a CML prefix on its
    first line. Idempotent: content already carrying a CML prefix is left as-is.
    Re-rewriting already-CML memories risks corrupting preserved facts (number/ID
    drift) for no normalization gain."""
    if not content:
        return False
    first_line = content.strip().split("\n")[0]
    starts_with_cml = any(
        first_line.startswith(p) for p in ("F:", "D:", "C:", "L:", "P:", "W:", "R:")
    )
    return not starts_with_cml
