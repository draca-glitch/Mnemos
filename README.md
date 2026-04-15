<p align="center">
  <img src="assets/mnemos-logo.svg" alt="Mnemos" width="220">
</p>

# Mnemos

> **The last memory you'll ever need.**
>
> A persistent memory system for AI agents.
> Named after Mnemosyne (Greek: μνήμη, *memory*).

**Runs on any computer. Works with any AI.**

A local, CPU-only memory store for Claude Code, Cursor, ChatGPT Desktop, Gemini, or any MCP-compatible AI client. Hybrid retrieval (BM25 + vectors + cross-encoder rerank), no LLM in the search path, four MCP tools, one SQLite file by default.

## Quick install

```bash
git clone https://github.com/draca-glitch/Mnemos.git
cd Mnemos
python3 -m venv venv && source venv/bin/activate
pip install -e .

# Register with Claude Code
claude mcp add -s user mnemos $(pwd)/venv/bin/mnemos serve
```

**First run downloads ~800 MB of ONNX models** (FastEmbed `multilingual-e5-large` embedder + Jina Reranker v2) into `~/.cache/fastembed/`. One-time cost, no GPU, no API key. Subsequent runs are offline.

**Add the agent instructions to your AI client.** Having the MCP tools available is not enough; the agent needs to be told *when* to search, *when* to store, and *what format* to use. Copy the ready-made block from [docs/agent-instructions.md](docs/agent-instructions.md) into your `~/.claude/CLAUDE.md` (Claude Code), your project's `CLAUDE.md`, `.cursorrules` (Cursor), or your Claude Desktop custom instructions. Without this, the model has the tools but will not use them proactively.

**If you use Claude Code, disable its built-in automemory** so the two stores don't compete:

```json
// ~/.claude/settings.json
{ "autoMemoryEnabled": false }
```

Restart your AI client. The 4 MCP tools (`memory_store`, `memory_search`, `memory_get`, `memory_update`) are now available and the agent knows how to use them.

For other clients (Cursor, ChatGPT Desktop, Gemini), CLI usage, hooks, and the optional Nyx consolidation cycle, see [docs/usage.md](docs/usage.md). Full five-minute walkthrough in [QUICKSTART.md](QUICKSTART.md).

## Key properties

- **CPU-only**: no GPU required. Embeddings and reranking via ONNX models that run on a regular laptop, NAS, Raspberry Pi 4+, or budget VPS. Even the optional consolidation LLM can run locally on CPU in "slow mode" since the Nyx cycle is a background job and quality matters more than speed.
- **MCP-native**: works with any MCP-compatible AI client out of the box.
- **CLI-friendly**: a full `mnemos` command-line tool ships alongside the MCP server, so you can store, search, ingest, and consolidate from any shell, script, or cron job, with or without an AI client attached.
- **100% local**: no API calls, no telemetry, no cloud dependencies. Your memory stays on your machine.
- **Pluggable backbones**: storage (SQLite by default; Qdrant for HNSW at 25K+ memories; Postgres planned), embedder (any FastEmbed-compatible via `MNEMOS_EMBED_MODEL`), reranker (any cross-encoder via `MNEMOS_RERANKER_MODEL`), consolidation LLM (any OpenAI-compatible endpoint, or skip entirely).

## Benchmarks

Mnemos is evaluated on two public long-conversation memory benchmarks (LongMemEval and LoCoMo) plus an end-to-end QA pass with abstention. In the canonical CML configuration, the same retrieval pipeline posts:

- **99.15% R@5 on LongMemEval** (`hybrid+rerank --cml`, 470 non-abstention questions)
- **86.1% R@5 on LoCoMo** (`hybrid+rerank --cml`, top-K capped at 10, 446 adversarial questions excluded per the same convention LongMemEval applies to abstention)
- **96.7% abstention accuracy** on LongMemEval's 30 abstention questions

All runs are first-run, no parameter tuning, no preprocessing of test data, no LLM in the retrieval path. Result JSON files in [`benchmarks/`](benchmarks/) so you can verify the numbers yourself. Per-mode methodology, comparison against MemPalace and the wider field, end-to-end QA, and consolidation-quality numbers in [docs/benchmarks.md](docs/benchmarks.md) and [docs/comparison.md](docs/comparison.md).

## The 4 MCP tools

```python
memory_store(project, content, tags?, importance?, type?, subcategory?,
             valid_from?, valid_until?, verified?, layer?)
memory_search(query, project?, subcategory?, type?, layer?,
              valid_only?, search_mode?, limit?)
memory_get(id)
memory_update(id, [any field])
```

That's the entire surface. No `navigate_to_wing`, no `open_room`, no `list_halls`. Just CRUD plus search. Hierarchy is metadata, not architecture. Why this matters in [docs/philosophy.md](docs/philosophy.md#why-4-tools-and-not-45).

## Architecture (visual summary)

```
                         ┌─────────────────┐
                         │  Claude Code    │
                         │   (or any MCP   │
                         │     client)     │
                         └────────┬────────┘
                                  │ JSON-RPC 2.0
                         ┌────────▼────────┐
                         │  Mnemos MCP     │
                         │     Server      │
                         │   (4 tools)     │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
        ┌─────▼─────┐      ┌──────▼──────┐    ┌──────▼──────┐
        │   FTS5    │      │   sqlite-   │    │    Jina     │
        │  BM25     │      │     vec     │    │   Reranker  │
        │ (lexical) │      │  (vectors)  │    │   (Jina v2) │
        └─────┬─────┘      └──────┬──────┘    └──────┬──────┘
              │                   │                   │
              └─────► RRF Merge ◄─┘                   │
                       │                              │
                       └──────► Cross-Encoder ◄───────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │   Results    │
                                └──────────────┘
```

On the default SQLite backend, everything (content, FTS index, 1024-dim vectors, memory links, Nyx consolidation history) lives in one SQLite file and every write is a single atomic transaction. Full layered architecture, data model schema, retrieval pipeline mechanics, and storage-backend tradeoffs in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Feature reference (hybrid pipeline, model rationale, temporal decay, Nyx phases, contradiction detection, auto-widen, dedup, forgetting) in [docs/features.md](docs/features.md).

## CML: token-minimal memory format

Mnemos uses **CML (Condensed Memory Language)** as a soft convention for writing memories. CML is not a parser, schema, or compressor; it's a tiny set of prefixes and symbols (`F:`, `D:`, `L:`, `→`, `∵`, `@`, etc.) that lets the writer pack common semantic patterns into the smallest number of tokens that still preserves meaning. Every token stays recognizable English, so FTS5, the bi-encoder, and the cross-encoder all see text they can handle.

Realistic compression: 14% on already-dense fact entries, up to 60% on narrative prose. Full grammar, fidelity benchmarks, and the "compression vs efficient language" rationale in [docs/cml.md](docs/cml.md).

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: five-minute install + first memory + first search
- **[docs/usage.md](docs/usage.md)**: CLI reference, MCP client setup, Nyx cycle config, session hooks, ingest, storage backends, RAM
- **[docs/features.md](docs/features.md)**: what's in the box: pipeline, models, temporal decay, Nyx phases, contradiction detection, forgetting
- **[docs/cml.md](docs/cml.md)**: CML grammar, fidelity benchmarks, opt-out
- **[docs/benchmarks.md](docs/benchmarks.md)**: full per-mode methodology, LoCoMo, end-to-end QA, consolidation quality
- **[docs/comparison.md](docs/comparison.md)**: MemPalace head-to-head and wider-field comparison table
- **[docs/philosophy.md](docs/philosophy.md)**: why 4 tools, why no LLM in the search path, what the Nyx cycle actually learns
- **[docs/origin.md](docs/origin.md)**: why this exists, why the name, why v10 in a brand-new repo
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: deep technical reference: data model, retrieval mechanics, storage internals
- **[docs/agent-instructions.md](docs/agent-instructions.md)**: copy-paste CLAUDE.md blocks for CML and prose modes

## Supporting Mnemos

Mnemos is MIT-licensed and developed by one person on a home server, in spare time, as a labor of curiosity. There is no company behind it, no VC pressure, and no plans to add a paid tier. If Mnemos saves you time, helps your AI assistant remember you better, or just makes you smile, you can chip in:

- ⭐ **Star the repo**. Costs nothing, helps people find it.
- ☕ **Buy me a coffee**: [buymeacoffee.com/dracaglitch](https://buymeacoffee.com/dracaglitch)
- 💖 **GitHub Sponsors**: [github.com/sponsors/draca-glitch](https://github.com/sponsors/draca-glitch)
- 🐛 **Open issues** for bugs, feature requests, or weird edge cases you hit in the wild
- 🛠️ **Send PRs**: extra extractors for the ingest module, a real Postgres backend, NLI-based contradiction detection, anything

Sponsorship goes directly toward keeping the development server running and toward more time spent improving Mnemos instead of doing my day job. Nothing you sponsor unlocks paid features. Mnemos stays one piece of software, fully open source, with everything in the public repo.

## License

MIT: see [LICENSE](LICENSE).

## Credits

Built on top of:
- [FastEmbed](https://github.com/qdrant/fastembed): multilingual e5-large ONNX embeddings
- [sqlite-vec](https://github.com/asg017/sqlite-vec): vector search in SQLite
- [Jina Reranker v2](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual): cross-encoder
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval): benchmark dataset
