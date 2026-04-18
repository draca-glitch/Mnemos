"""
Cemelify: rewrite content into compact CML notation via the configured LLM.

Used in two places:

  1. Opt-in store-time hook: when MNEMOS_CEMELIFY_ON_IMPORT=1,
     core.store_memory() pipes content through cemelify() before
     persistence, so memories land already in CML form.
  2. Phase 0.5 of the Nyx cycle (on by default when an LLM is configured):
     scans active memories that either don't start with a CML prefix
     (F:/D:/C:/L:/P:/W:) or are longer than 800 chars, and rewrites
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
    "user's content into a single compact CML line using one of the canonical "
    "prefixes:\n"
    "  F: Fact (verifiable atomic claim)\n"
    "  D: Decision\n"
    "  C: Constraint or Caveat\n"
    "  L: Lesson or Learning\n"
    "  P: Preference\n"
    "  W: Warning\n"
    "Preserve every concrete fact, number, name, and identifier from the "
    "source. Drop filler, redundant context, and meta-commentary. If the "
    "source already starts with a CML prefix, return it unchanged. Reply "
    "with the CML line only, no explanation, no quoting."
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
        response = chat(messages, max_tokens=max_tokens, temperature=0.2)
    except Exception:
        return content

    if not response:
        return content
    cemelified = response.strip()
    if not cemelified:
        return content
    return cemelified


def _needs_cemelify(content: Optional[str]) -> bool:
    """Phase 0.5 filter: candidate when content lacks a CML prefix on its
    first line OR exceeds 800 chars (likely prose that slipped through)."""
    if not content:
        return False
    first_line = content.strip().split("\n")[0]
    starts_with_cml = any(
        first_line.startswith(p) for p in ("F:", "D:", "C:", "L:", "P:", "W:")
    )
    if not starts_with_cml:
        return True
    if len(content) > 800:
        return True
    return False
