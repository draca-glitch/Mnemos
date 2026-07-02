"""
Generic LLM client for Mnemos consolidation.

Supports any OpenAI-compatible chat completions API:
  - OpenAI (api.openai.com)
  - OpenRouter (openrouter.ai)
  - Ollama (localhost:11434/v1)
  - Anthropic via OpenAI-compat proxies
  - DigitalOcean Gradient
  - Together.ai, Groq, Fireworks, etc.

Configuration via environment variables (all required for consolidation
phases that need an LLM; Mnemos has no opinion on which provider you use):
  MNEMOS_LLM_API_URL    chat completions endpoint (defaults to api.openai.com)
  MNEMOS_LLM_API_KEY    API key (Bearer token), required, no default
  MNEMOS_LLM_MODEL      model name for consolidation phases, required, no default
  MNEMOS_LLM_FAST_MODEL faster/cheaper model for Phase 1 triage (optional;
                        falls back to MNEMOS_LLM_MODEL if unset)

Per-phase model routing (optional, all fall back to MNEMOS_LLM_MODEL if unset):
  MNEMOS_LLM_MODEL_MERGE        Phase 2 Dedup merge (recommended: gpt-4o-mini)
  MNEMOS_LLM_MODEL_WEAVE        Phase 3 cross-category link classification
  MNEMOS_LLM_MODEL_CONTRADICT   Phase 4 temporal evolution classification
                                (recommended: Sonnet-class, affects memory state)
  MNEMOS_LLM_MODEL_SYNTHESIZE   Phase 5 cross-domain insight generation
                                (recommended: Opus-class, quality > cost)

Per-phase API URL overrides (optional, rare, mainly for mixing providers):
  MNEMOS_LLM_API_URL_<PHASE>    same pattern as MODEL overrides. Useful if you
                                want local Ollama for cheap phases and a remote
                                API for Synthesize, for example.

If either MNEMOS_LLM_API_KEY or MNEMOS_LLM_MODEL is unset, all chat() calls
return None and consolidation phases that depend on LLM will be skipped
automatically with a warning. Phase 6 (Bookkeeping) always runs since it
is pure SQL. The base memory store and search work without any LLM
configuration; only the optional Nyx cycle needs one.
"""

import json
import os
import time
import urllib.request
import urllib.error


DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = None
DEFAULT_FAST_MODEL = None

# Per-call read timeout (seconds). 60s was too tight for reasoning-class
# models on large hierarchical-merge prompts: a slow-but-completing call
# would time out across all retries -> chat() returns None -> phases.py
# raw-concatenation fallback (the 2026-05-11 merge-quality degradation).
# Env-tunable so operators can adjust per provider without a code change.
LLM_TIMEOUT = int(os.environ.get("MNEMOS_LLM_TIMEOUT", "240"))


# Per-chat() wall-clock budget (seconds). Caps total time spent across all
# 3 retries + their exponential backoffs. Without this, three 240s read
# timeouts plus backoffs can burn ~726s on a single hung call, which on
# 2026-05-27 sank a midweek Nyx run by eating budget that the cemelify
# loop needed. 480s gives one full retry-with-backoff cycle but stops the
# call before a third 240s wait. Env-tunable per provider.
LLM_WALL_BUDGET = int(os.environ.get("MNEMOS_LLM_WALL_BUDGET", "480"))


def _get_config(phase=None):
    """Read LLM config from environment. Returns dict, may have empty values.

    If `phase` is given (e.g. "MERGE", "WEAVE", "CONTRADICT", "SYNTHESIZE"),
    per-phase env var overrides are consulted first: MNEMOS_LLM_MODEL_<PHASE>,
    MNEMOS_LLM_API_URL_<PHASE>, and MNEMOS_LLM_API_KEY_<PHASE>. Each falls back
    to the non-scoped MNEMOS_LLM_MODEL / _API_URL / _API_KEY if unset. The
    per-phase key lets one phase (e.g. MERGE) target a cloud provider that
    needs a real token while the other phases stay on a local endpoint whose
    key is a throwaway; the secret never reaches the local endpoint.

    `phase_model_explicit` lets chat() tell an explicit per-phase override
    apart from a fallback, so the phase override can win over the fast
    flag (otherwise haiku_chat(phase=...) would ignore the phase model).
    """
    phase_suffix = f"_{phase.upper()}" if phase else ""
    phase_model = os.environ.get(f"MNEMOS_LLM_MODEL{phase_suffix}") if phase else None
    phase_url = os.environ.get(f"MNEMOS_LLM_API_URL{phase_suffix}") if phase else None
    phase_key = os.environ.get(f"MNEMOS_LLM_API_KEY{phase_suffix}") if phase else None
    phase_omit_temp = os.environ.get(f"MNEMOS_LLM_OMIT_TEMPERATURE{phase_suffix}") if phase else None
    omit_temp = (phase_omit_temp if phase_omit_temp is not None
                 else os.environ.get("MNEMOS_LLM_OMIT_TEMPERATURE", "0"))
    url = phase_url or os.environ.get("MNEMOS_LLM_API_URL", DEFAULT_API_URL)

    # v10.4.0: default model preset for the OpenAI endpoint only. If the
    # user left MNEMOS_LLM_API_URL at the OpenAI default AND MNEMOS_LLM_MODEL
    # is unset, fall back to gpt-4o-mini (recommended per the consolidation
    # quality bench in docs/benchmarks.md). No default for non-OpenAI URLs:
    # provider model naming is too heterogeneous to guess.
    env_model = os.environ.get("MNEMOS_LLM_MODEL")
    openai_default = "gpt-4o-mini" if (not env_model and "api.openai.com" in url) else None

    return {
        "url": url,
        "key": phase_key or os.environ.get("MNEMOS_LLM_API_KEY", ""),
        "model": phase_model or env_model or openai_default or DEFAULT_MODEL,
        "fast_model": os.environ.get("MNEMOS_LLM_FAST_MODEL")
                      or env_model or openai_default or DEFAULT_FAST_MODEL,
        "phase_model_explicit": bool(phase_model),
        "omit_temperature": str(omit_temp).lower() in ("1", "true", "yes", "on"),
    }


def is_configured() -> bool:
    """Whether an LLM API is configured (both key AND model). Used to skip
    phases that need it."""
    cfg = _get_config()
    return bool(cfg["key"]) and bool(cfg["model"])


def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def chat(messages, max_tokens=1024, temperature=0.3, fast=False, phase=None, timeout=None):
    """Call an OpenAI-compatible chat completions endpoint.

    Returns the response content string, or None on any failure (no LLM
    configured, network error, parse error, etc.). Never raises.

    Args:
        messages: list of {"role": ..., "content": ...} dicts
        max_tokens: max response tokens
        temperature: sampling temperature
        fast: use the fast/cheap model instead of the main model
        phase: optional phase name ("MERGE", "WEAVE", "CONTRADICT",
               "SYNTHESIZE"). When set, MNEMOS_LLM_MODEL_<PHASE> and
               MNEMOS_LLM_API_URL_<PHASE> env vars are consulted first,
               falling back to the global MNEMOS_LLM_MODEL / _API_URL.
        timeout: optional per-call read timeout (seconds) overriding the
               module-level LLM_TIMEOUT for this single call. Use a
               tighter value for small/fast paths (e.g. Phase 0.5
               Cemelify) so one slow candidate cannot consume budget
               meant for hierarchical-merge prompts. None = LLM_TIMEOUT.
               The per-chat() wall-clock ceiling LLM_WALL_BUDGET still
               applies on top, regardless of this override.
    """
    cfg = _get_config(phase=phase)
    if not cfg["key"]:
        return None
    # Per-phase override wins over the fast flag: if the caller set
    # MNEMOS_LLM_MODEL_<PHASE> they meant it, even for haiku_chat paths.
    if cfg.get("phase_model_explicit"):
        model = cfg["model"]
    else:
        model = cfg["fast_model"] if fast else cfg["model"]
    if not model:
        return None

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    # Some newer models (e.g. Anthropic Sonnet 5 on the OpenAI-compat endpoint)
    # reject `temperature` as deprecated and 400 the whole call. Omit it for such
    # endpoints via MNEMOS_LLM_OMIT_TEMPERATURE[_<PHASE>].
    if not cfg.get("omit_temperature"):
        payload["temperature"] = temperature
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "authorization": f"Bearer {cfg['key']}",
        "content-type": "application/json",
    }

    call_timeout = timeout if timeout is not None else LLM_TIMEOUT
    start = time.monotonic()
    for attempt in range(3):
        if time.monotonic() - start > LLM_WALL_BUDGET:
            _log(f"LLM API wall-budget exceeded ({LLM_WALL_BUDGET}s), giving up")
            return None
        try:
            req = urllib.request.Request(cfg["url"], data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=call_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content") or ""
                    return content.strip() if content else None
        except urllib.error.HTTPError as e:
            if attempt < 2 and e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            _log(f"LLM API HTTPError: {e.code} {e.reason}")
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            _log(f"LLM API error: {e}")
            return None
        except Exception as e:
            _log(f"LLM API unexpected error: {e}")
            return None
    return None


# Backwards-compat aliases for the Nyx cycle phase code.
# All three now accept the optional `phase` parameter to route to
# MNEMOS_LLM_MODEL_<PHASE> when set, falling back to the global model.
def haiku_chat(messages, max_tokens=256, temperature=0.3, phase=None, timeout=None):
    return chat(messages, max_tokens=max_tokens, temperature=temperature, fast=True, phase=phase, timeout=timeout)


def sonnet_chat(messages, max_tokens=1024, temperature=0.3, phase=None, timeout=None):
    return chat(messages, max_tokens=max_tokens, temperature=temperature, fast=False, phase=phase, timeout=timeout)


def opus_chat(messages, max_tokens=2048, temperature=0.3, phase=None, timeout=None):
    return chat(messages, max_tokens=max_tokens, temperature=temperature, fast=False, phase=phase, timeout=timeout)
