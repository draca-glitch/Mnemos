"""
Mnemos consolidation: the Nyx cycle for memory merging and synthesis.

6 phases modeled on brain sleep:
  Phase 1: Triage      - detect new memories, decide surge mode
  Phase 2: Dedup       - merge near-duplicates and same-topic memories
  Phase 3: Weave       - find cross-category connections (memory_links)
  Phase 4: Contradict  - detect temporal evolution, mark superseded facts
  Phase 5: Synthesize  - generate cross-domain insights via LLM
  Phase 6: Bookkeep    - decay, cleanup orphans, prune stale links

Phases 2-5 require an LLM; phases 1 and 6 are pure SQL. Mnemos supports any OpenAI-compatible API and
has no default model: pick the one your existing API key supports.
  MNEMOS_LLM_API_URL    endpoint (default: OpenAI's chat completions)
  MNEMOS_LLM_API_KEY    API key (required for LLM phases)
  MNEMOS_LLM_MODEL      model name (required for LLM phases, no default)
  MNEMOS_LLM_FAST_MODEL faster model for triage (optional, falls back to MODEL)

Examples:
  # OpenAI
  export MNEMOS_LLM_API_URL=https://api.openai.com/v1/chat/completions
  export MNEMOS_LLM_API_KEY=sk-...
  export MNEMOS_LLM_MODEL=gpt-4o-mini

  # Local Ollama
  export MNEMOS_LLM_API_URL=http://localhost:11434/v1/chat/completions
  export MNEMOS_LLM_API_KEY=ollama
  export MNEMOS_LLM_MODEL=qwen2.5:14b

  # OpenRouter
  export MNEMOS_LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
  export MNEMOS_LLM_API_KEY=sk-or-...
  export MNEMOS_LLM_MODEL=anthropic/claude-3.5-sonnet
"""

from .orchestrator import run_nyx_cycle

__all__ = ["run_nyx_cycle"]
