"""
Mnemos consolidation: the dream cycle for memory merging and synthesis.

5 phases modeled on brain sleep:
  Phase 0: Triage      - detect new memories, decide surge mode
  Phase 1: Dedup       - merge near-duplicates and same-topic memories
  Phase 2: Weave       - find cross-category connections (memory_links)
  Phase 3: Contradict  - detect temporal evolution, mark superseded facts
  Phase 4: Synthesize  - generate cross-domain insights via LLM
  Phase 5: Bookkeep    - decay, cleanup orphans, prune stale links

Phases 1-4 require an LLM. Mnemos supports any OpenAI-compatible API:
  MNEMOS_LLM_API_URL    endpoint (default: OpenAI's chat completions)
  MNEMOS_LLM_API_KEY    API key
  MNEMOS_LLM_MODEL      model name (default: gpt-4o-mini)
  MNEMOS_LLM_FAST_MODEL faster model for triage (default: same)

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

from .orchestrator import run_dream_cycle

__all__ = ["run_dream_cycle"]
