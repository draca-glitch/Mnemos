"""
Generic LLM client for Mnemos consolidation.

Supports any OpenAI-compatible chat completions API:
  - OpenAI (api.openai.com)
  - OpenRouter (openrouter.ai)
  - Ollama (localhost:11434/v1)
  - Anthropic via OpenAI-compat proxies
  - DigitalOcean Gradient
  - Together.ai, Groq, Fireworks, etc.

Configuration via environment variables:
  MNEMOS_LLM_API_URL    chat completions endpoint
  MNEMOS_LLM_API_KEY    API key (Bearer token)
  MNEMOS_LLM_MODEL      model name for consolidation phases (Phase 1-4)
  MNEMOS_LLM_FAST_MODEL faster/cheaper model for Phase 0 triage (optional)

If no API key is configured, all chat() calls return None and consolidation
phases that depend on LLM will be skipped automatically. Phase 5 (Bookkeeping)
always runs since it's pure SQL.
"""

import json
import os
import time
import urllib.request
import urllib.error


DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_FAST_MODEL = "gpt-4o-mini"


def _get_config():
    """Read LLM config from environment. Returns dict, may have empty values."""
    return {
        "url": os.environ.get("MNEMOS_LLM_API_URL", DEFAULT_API_URL),
        "key": os.environ.get("MNEMOS_LLM_API_KEY", ""),
        "model": os.environ.get("MNEMOS_LLM_MODEL", DEFAULT_MODEL),
        "fast_model": os.environ.get("MNEMOS_LLM_FAST_MODEL")
                      or os.environ.get("MNEMOS_LLM_MODEL", DEFAULT_FAST_MODEL),
    }


def is_configured() -> bool:
    """Whether an LLM API is configured. Used to skip phases that need it."""
    return bool(_get_config()["key"])


def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def chat(messages, max_tokens=1024, temperature=0.3, fast=False):
    """Call an OpenAI-compatible chat completions endpoint.

    Returns the response content string, or None on any failure (no LLM
    configured, network error, parse error, etc.). Never raises.

    Args:
        messages: list of {"role": ..., "content": ...} dicts
        max_tokens: max response tokens
        temperature: sampling temperature
        fast: use the fast/cheap model instead of the main model
    """
    cfg = _get_config()
    if not cfg["key"]:
        return None

    payload = {
        "model": cfg["fast_model"] if fast else cfg["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "authorization": f"Bearer {cfg['key']}",
        "content-type": "application/json",
    }

    for attempt in range(3):
        try:
            req = urllib.request.Request(cfg["url"], data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
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


# Backwards-compat aliases for the dream cycle phase code
def haiku_chat(messages, max_tokens=256, temperature=0.3):
    return chat(messages, max_tokens=max_tokens, temperature=temperature, fast=True)


def sonnet_chat(messages, max_tokens=1024, temperature=0.3):
    return chat(messages, max_tokens=max_tokens, temperature=temperature, fast=False)


def opus_chat(messages, max_tokens=2048, temperature=0.3):
    return chat(messages, max_tokens=max_tokens, temperature=temperature, fast=False)
