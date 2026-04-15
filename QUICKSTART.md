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

Restart the client. Four tools appear: `memory_store`, `memory_search`, `memory_get`, `memory_update`.

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

# Health check
mnemos doctor
```

## CML in thirty seconds

Mnemos stores memories in CML (Compressed Memory Language), a token-minimal format:

- Type prefixes: `D:` decision, `C:` contact, `F:` fact, `L:` learning, `P:` preference, `W:` warning
- Relation symbols: `→` leads to, `∵` because, `∴` therefore, `⚠` caveat, `@` at/context, `✓` confirmed, `✗` wrong
- One line per fact, chain with `;`

Example: `D:use sqlite-vec over pgvector ∵ embedded single-user @2026-02-13`

Stores prose too, but the Nyx cycle will cemelify it over time. See [README §CML](README.md#cml-token-minimal-memory-format) for details.

## Optional: weekly consolidation

The Nyx cycle merges duplicates, detects contradictions, and synthesizes cross-domain insights. Requires an LLM endpoint:

```bash
export MNEMOS_LLM_API_URL="https://api.openai.com/v1/chat/completions"
export MNEMOS_LLM_API_KEY="sk-..."
export MNEMOS_LLM_MODEL="gpt-4o-mini"

mnemos consolidate --execute
```

Any OpenAI-compatible endpoint works (OpenAI, Anthropic via proxy, Ollama, OpenRouter, DigitalOcean Gradient, Groq, etc.). Skip entirely if you want a static memory store with no adaptive layer.

## Next

- Full features and rationale: [README.md](README.md)
- Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Reproducible benchmarks: [benchmarks/README.md](benchmarks/README.md)
- Version history: [CHANGELOG.md](CHANGELOG.md)
