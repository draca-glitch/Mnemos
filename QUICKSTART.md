# Mnemos Quickstart

Five-minute walkthrough. For the full story (architecture, benchmarks, CML, Nyx cycle, storage backends) read [README.md](README.md) after this.

## Install

```bash
pip install mnemos
```

Python 3.11+ required. CPU-only, no GPU. First `mnemos` invocation downloads the embedding model (e5-large, ~1 GB) into `~/.cache/fastembed/`.

## Register with your AI client

Claude Code users can register Mnemos in one command:

```bash
claude mcp add -s user mnemos mnemos-mcp
```

For Cursor, ChatGPT Desktop, Gemini, or any other MCP-compatible client, add the equivalent entry to that client's MCP config:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "mnemos-mcp"
    }
  }
}
```

Restart the client. Six tools appear: four hot-path CRUD (`memory_store`, `memory_search`, `memory_get`, `memory_update`) plus two maintenance tools (`memory_bulk_rewrite`, `memory_list_tags`).

**If you use Claude Code**, also disable the built-in automemory so the two systems do not compete:

```json
// ~/.claude/settings.json
{ "autoMemoryEnabled": false }
```

## Store and search from the CLI

```bash
# Store a memory
mnemos add --project dev "F:Mnemos uses sqlite-vec for vectors"

# Search it
mnemos search "vector storage" --project dev

# Get a memory by id
mnemos get 1

# Health check (add --migrate to apply safe repairs behind an auto-backup)
mnemos doctor
```

## CML in thirty seconds

Mnemos stores memories in CML (Compressed Memory Language), a token-minimal format:

- Type prefixes: `D:` decision, `C:` contact, `F:` fact, `L:` learning, `P:` preference, `W:` warning
- Relation symbols: `→` leads to, `∵` because, `∴` therefore, `⚠` caveat, `@` at/context, `✓` confirmed, `✗` wrong
- One line per fact, chain with `;`

Example: `D:use sqlite-vec over pgvector ∵ embedded single-user @2026-02-13`

Stores prose too, but the Nyx cycle will cemelify it over time. See [README §CML](README.md#cml-token-minimal-memory-format) for details.

## Consolidation: the two-tier Nyx cycle

Nyx keeps the store tidy over time. Since v10.17 it runs in two tiers.

**Nightly (zero-LLM, no setup).** SQL triage, mutual-top-k candidacy, an NLI cluster gate, mechanical line-union merges (facts are *selected and unioned*, never rewritten by a model), and an NLI contradiction finder. No API key, no cloud, no local LLM:

```bash
mnemos consolidate --execute        # runs the zero-LLM core; auto-skips LLM phases if none configured
```

**Weekly (LLM-driven, optional).** With an OpenAI-compatible endpoint set, the *same command* additionally runs semantic weave, generative merge, and a contradiction judge that drains the nightly queue. Add `--nyx` for cross-domain synthesis (phase 5):

```bash
export MNEMOS_LLM_API_URL="https://api.openai.com/v1/chat/completions"
export MNEMOS_LLM_API_KEY="sk-..."
export MNEMOS_LLM_MODEL="gpt-4o-mini"

mnemos consolidate --execute          # now includes LLM weave / merge / judge
mnemos consolidate --execute --nyx    # + cross-domain synthesis
```

Any OpenAI-compatible endpoint works (OpenAI, Anthropic, Ollama, OpenRouter, Groq, etc.). Skip the weekly tier entirely; the nightly tier still self-maintains the store every night with no LLM at all.

## Optional: sharper store decisions (NLI)

By default, store-time dedup and contradiction checks reuse the cross-encoder ("same topic?"). The optional NLI layer swaps in natural-language-inference models that answer the question dedup and contradiction detection actually need ("same claim, or the opposite claim?"), and it also powers the nightly Nyx cluster gate and contradiction finder:

```bash
pip install mnemos[nli]
python scripts/export_nli_onnx.py         # one-time model export
export MNEMOS_DEDUP_CONFIRM=nli
export MNEMOS_CONTRADICT_MODE=nli
export MNEMOS_NYX_CONTRADICT_FINDER=nli
```

CPU-only ONNX like the rest of the pipeline; English content routes to an English checkpoint, everything else to a multilingual XNLI one. Benchmarked on real production memories: contradiction AUC 0.94 vs 0.69, dedup false-blocks 1 vs 16+. Full rationale in [README §Key properties](README.md#key-properties).

## Next

- Full features and rationale: [README.md](README.md)
- Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Copy-paste agent instructions for your `CLAUDE.md` / Cursor rules / custom system prompt: [docs/agent-instructions.md](docs/agent-instructions.md)
- Reproducible benchmarks: [benchmarks/README.md](benchmarks/README.md)
- Version history: [CHANGELOG.md](CHANGELOG.md)
