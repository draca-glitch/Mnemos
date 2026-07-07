<p align="center">
  <img src="assets/mnemos-logo.svg" alt="Mnemos" width="220">
</p>

# Mnemos

> **The last memory you'll ever need.**
>
> A persistent memory system for AI agents.
> Named after Mnemosyne (Greek: μνήμη, *memory*).

**Runs on any computer. Works with any AI.**

A local, CPU-only memory store for Claude Code, Cursor, ChatGPT Desktop, Gemini, or any MCP-compatible AI client. Hybrid retrieval (BM25 + vectors + cross-encoder rerank), no LLM in the search path, four hot-path MCP tools (plus two maintenance tools), one SQLite file by default.

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

Restart your AI client. The 6 MCP tools (four hot-path: `memory_store`, `memory_search`, `memory_get`, `memory_update`; two maintenance: `memory_bulk_rewrite`, `memory_list_tags`) are now available and the agent knows how to use them.

For other clients (Cursor, ChatGPT Desktop, Gemini), CLI usage, hooks, and the optional Nyx consolidation cycle, see [docs/usage.md](docs/usage.md). Full five-minute walkthrough in [QUICKSTART.md](QUICKSTART.md).

## Key properties

- **CPU-only**: no GPU required. Embeddings and reranking via ONNX models that run on a regular laptop, NAS, Raspberry Pi 4+, or budget VPS. Even the optional weekly-tier consolidation LLM can run locally on CPU in "slow mode"; the nightly consolidation tier needs no LLM at all (see below), and the Nyx cycle is a background job where quality matters more than speed.
- **MCP-native**: works with any MCP-compatible AI client out of the box.
- **CLI-friendly**: a full `mnemos` command-line tool ships alongside the MCP server, so you can store, search, ingest, and consolidate from any shell, script, or cron job, with or without an AI client attached.
- **100% local**: no API calls, no telemetry, no cloud dependencies. Your memory stays on your machine.
- **Pluggable backbones**: storage (SQLite by default, atomic single-file; Qdrant scaling layer for HNSW at 25K+ memories; Postgres backend is a stub, not implemented yet), embedder (any FastEmbed-compatible via `MNEMOS_EMBED_MODEL`), reranker (any cross-encoder via `MNEMOS_RERANKER_MODEL`), consolidation LLM (any OpenAI-compatible endpoint, or skip entirely).
- **NLI decision layer (optional, v10.15)**: dedup confirmation and contradiction detection can run on natural-language-inference models instead of the reranker (`pip install mnemos[nli]`, export models once with `scripts/export_nli_onnx.py`, then `MNEMOS_DEDUP_CONFIRM=nli` + `MNEMOS_CONTRADICT_MODE=nli` + `MNEMOS_NYX_CONTRADICT_FINDER=nli`). A reranker answers "same topic?"; NLI answers "same claim or opposite claim?", which is what the store decision actually needs. Benchmarked on real production memories: contradiction AUC 0.94 vs 0.69, dedup false blocks 1 vs 16+. Language-agnostic: English content routes to an English checkpoint, everything else to a multilingual XNLI checkpoint. Runs on onnxruntime like the rest of the pipeline (fp32; int8 was rejected by a score-parity gate, see CHANGELOG 10.16.0), torch fallback via `mnemos[nli-torch]`. Since v10.17 the same NLI layer also gates the nightly consolidation cluster and finds contradictions there, so it earns its keep on both the store path and the background cycle.
- **Two-tier consolidation (v10.17)**: the nightly Nyx cycle runs **zero-LLM**: SQL triage, mutual-top-k candidacy, an NLI cluster gate, mechanical line-union merges (facts are *selected and unioned*, never model-rewritten, τ=0.90), and an NLI contradiction finder that queues contradiction-candidate links. Deduping, clustering, and contradiction detection need no API key and no local LLM. A separate **weekly** tier adds LLM-driven weave / generative merge / synthesis and a contradiction judge that drains the nightly queue, *only* when an endpoint is configured. `mnemos consolidate --execute` runs the zero-LLM core by default and skips the enrichment phases (with a grep-able WARNING) instead of failing when no LLM is set.
- **Self-checking & tier-2 recall (v10.19–v10.24)**: `mnemos doctor` verifies integrity, schema, FTS↔vector sync, content↔vector coherence (catches content edited without a re-embed), vector-model provenance, empty-store misconfiguration, and the archived-vector index; `mnemos doctor --migrate` applies safe repairs behind an automatic backup. Archived originals stay searchable through a separate tier-2 vector index (`mnemos reindex-archived`, `memory_search(..., expand_merged=True)`) even after Nyx has merged them away.

## Benchmarks

Same pipeline runs against three public benchmarks:

- **LongMemEval** (retrieval recall, 470 non-abstention questions): four configurations published, from the lite BM25+vector mode up to the canonical BM25+vector+rerank+CML.
- **LoCoMo** (retrieval recall on long continuous conversations, 1,540 evaluable QA pairs): same pipeline, no parameter tuning, top-K capped to avoid the bypass regime.
- **LongMemEval end-to-end QA with abstention** (500 questions including 30 abstention cases): LLM-judged answer correctness paired with the retrieval pipeline.

**Every configuration is published, including the one where Mnemos looks worst** (`hybrid --cml` drops R@1 on single-session-preference to 53.33% without the reranker to rescue it). Every run is first-run, no parameter tuning, no preprocessing of test data, no LLM in the retrieval path. Result JSON files with timestamps live in [`benchmarks/`](benchmarks/) so anyone can verify the numbers. Per-mode methodology, the full number tables, comparison against MemPalace and the wider field, end-to-end QA, and consolidation-quality numbers in [docs/benchmarks.md](docs/benchmarks.md) and [docs/comparison.md](docs/comparison.md).

## The 6 MCP tools

Hot path (four, used on every session):

```python
memory_store(project, content, tags?, importance?, type?, subcategory?,
             valid_from?, valid_until?, verified?, layer?)
memory_search(query, project?, subcategory?, type?, layer?,
              valid_only?, search_mode?, limit?)
memory_get(id)
memory_update(id, [any field])
```

Maintenance (two, used rarely):

```python
memory_bulk_rewrite(pattern, replacement, use_regex?, project?, tags?,
                    dry_run?, max_affected?)
memory_list_tags(project?, limit?, min_count?)
```

That is the entire surface: CRUD-plus-search on the hot path, pattern rewrite and schema introspection on the maintenance path. Hierarchy is metadata (project / subcategory columns), not architecture. Filters, validity windows, and search modes are parameters on `memory_search`, not new tools. Why this matters in [docs/philosophy.md](docs/philosophy.md#why-4-hot-path-tools-and-not-45).

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
                         │  (4 hot-path    │
                         │  + 2 maint.)    │
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

     Store path (memory_store, optional NLI decision layer v10.15+):

        new memory ──► 3-way candidate pool ──► NLI (DeBERTa-v3, ONNX)
                       (FTS5 + CML + vec)       ├─ bidirectional entailment
                                                │   ≥ 0.85 → duplicate, block
                                                └─ max-direction P(contra)
                                                    ≥ 0.98 → warn + link
```

Search asks "what is relevant?" (lexical + semantic + topicality). The store decision asks "is this the same claim, or the opposite claim?", which is an entailment question, so it runs on NLI models rather than the reranker. English content routes to an English checkpoint, everything else to a multilingual one.

On the default SQLite backend, everything (content, FTS index, 1024-dim vectors, memory links, Nyx consolidation history) lives in one SQLite file and every write is a single atomic transaction. Full layered architecture, data model schema, retrieval pipeline mechanics, and storage-backend tradeoffs in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Feature reference (hybrid pipeline, model rationale, temporal decay, Nyx phases, contradiction detection, auto-widen, dedup, forgetting) in [docs/features.md](docs/features.md).

## Deployment profiles

Mnemos scales down to a Pi and up to a production server with an LLM budget. Three profiles cover the useful spectrum; pick the one that matches your constraints.

### 1. Edge / Pi-class (sub-1 GB RAM)

```bash
export MNEMOS_ENABLE_RERANK=0
export MNEMOS_CONTRADICT_MODE=vec
# Resource-aware loading (v10.5.0), all optional and off by default:
export MNEMOS_EAGER_WARMUP=0       # load models on first use, not at startup
export MNEMOS_MODEL_IDLE_TTL=1800  # unload models after 30 min idle, reclaim RAM
export MNEMOS_MIN_FREE_MB=500      # degrade to lighter search instead of OOM
```

- **Models loaded**: FastEmbed e5-large only (~600 MB; drops to ~250 MB with `int8` variants)
- **Search**: FTS5 + vec + RRF merge, no cross-encoder rerank
- **Contradiction detection**: Tier-1 vec gate only; any close pair flagged as `contradicts`
- **Nyx consolidation**: the full zero-LLM nightly tier runs (triage, candidacy, mechanical line-union merges, contradiction bookkeeping); the weekly LLM tier is skipped. The NLI cluster gate/finder engage only if the `mnemos[nli]` extra is installed, else the tier falls back to vec-gated candidacy
- **Resource-aware models (v10.5.0)**: with `MNEMOS_MODEL_IDLE_TTL` set, an idle embedder/reranker is dropped and its RAM returned to the OS (the next query pays a one-off reload); `MNEMOS_MIN_FREE_MB` makes the search path degrade to vec-only then FTS5 under memory pressure rather than risk an OOM. Both default off, so nothing changes unless you opt in.
- **Good for**: personal memory on constrained hardware, offline-first deployments

### 2. Standard self-hosted (default, no API costs)

```bash
# Default config - no env vars needed
```

- **Models loaded**: FastEmbed e5-large (~600 MB) + Jina cross-encoder (~500 MB)
- **Search**: full hybrid pipeline (FTS5 + vec + RRF + rerank), this is what the 98.1% R@5 benchmark runs on
- **Contradiction detection**: three-tier with `relates` silent-link zone, no API cost per store
- **Nyx consolidation**: the zero-LLM nightly tier runs in full (NLI cluster gate + mechanical line-union merges + contradiction detection); the weekly LLM tier (weave/generative-merge/synthesis) stays dormant until an endpoint is added
- **Good for**: most users, homelabs, single-server personal deployments

### 3. Full LLM (premium precision, paid API per operation)

```bash
export MNEMOS_LLM_API_KEY=sk-...
export MNEMOS_LLM_MODEL=gpt-4o-mini  # or claude-haiku, etc.
export MNEMOS_CONTRADICT_MODE=llm
export MNEMOS_RETRIEVAL_LOG=1        # optional: real-query analytics
export MNEMOS_TOOL_USAGE_LOG=1       # optional: MCP call diagnostics
```

- **Models loaded**: e5-large + Jina + LLM API (no local LLM weights)
- **Search**: full hybrid pipeline (unchanged from standard profile)
- **Contradiction detection**: five-way LLM classification - `contradicts`, `refines`, `evolves`, `relates`, `unrelated`
- **Nyx consolidation**: both tiers run on schedule - the zero-LLM nightly tier plus the weekly LLM tier (weave/generative-merge/synthesize) build a model of the user over time
- **Analytics**: retrieval and tool-usage logs feed autoimprove benchmarks and quality analysis
- **Good for**: production deployments where memory is a shared agent resource, teams, long-running research systems

The profiles are cumulative, not exclusive: flipping `MNEMOS_LLM_*` on at any point unlocks LLM-dependent features without touching the rest. No "upgrade" migration needed - just env var changes.

## CML: token-minimal memory format

Mnemos uses **CML (Condensed Memory Language)** as a soft convention for writing memories. CML is not a parser, schema, or compressor; it's a tiny set of prefixes and symbols (`F:`, `D:`, `L:`, `→`, `∵`, `@`, etc.) that lets the writer pack common semantic patterns into the smallest number of tokens that still preserves meaning. Every token stays recognizable English, so FTS5, the bi-encoder, and the cross-encoder all see text they can handle.

Realistic compression: 14% on already-dense fact entries, up to 60% on narrative prose. Full grammar, fidelity benchmarks, and the "compression vs efficient language" rationale in [docs/cml.md](docs/cml.md).

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: five-minute install + first memory + first search
- **[docs/usage.md](docs/usage.md)**: CLI reference, MCP client setup, Nyx cycle config, session hooks, ingest, storage backends, RAM
- **[docs/features.md](docs/features.md)**: what's in the box: pipeline, models, temporal decay, Nyx phases, contradiction detection, forgetting
- **[docs/cml.md](docs/cml.md)**: CML grammar, fidelity benchmarks, opt-out
- **[docs/english-primary.md](docs/english-primary.md)**: the English-primary store convention, why it feeds the NLI layer, and the migration runbook for existing non-English stores
- **[docs/benchmarks.md](docs/benchmarks.md)**: full per-mode methodology, LoCoMo, end-to-end QA, consolidation quality
- **[docs/comparison.md](docs/comparison.md)**: MemPalace head-to-head and wider-field comparison table
- **[docs/philosophy.md](docs/philosophy.md)**: why the tool surface is minimal, why no LLM in the search path, what the Nyx cycle actually learns
- **[docs/origin.md](docs/origin.md)**: why this exists, why the name, why v10 in a brand-new repo
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: deep technical reference: data model, retrieval mechanics, storage internals
- **[docs/agent-instructions.md](docs/agent-instructions.md)**: copy-paste CLAUDE.md blocks for CML and prose modes

## Supporting Mnemos

Mnemos is MIT-licensed and developed by one person on a home server, in spare time, as a labor of curiosity. There is no company behind it, no VC pressure, and no plans to add a paid tier. If Mnemos saves you time, helps your AI assistant remember you better, or just makes you smile, you can chip in:

- ⭐ **Star the repo**. Costs nothing, helps people find it.
- ☕ **Buy me a coffee**: [buymeacoffee.com/dracaglitch](https://buymeacoffee.com/dracaglitch)
- 💖 **GitHub Sponsors**: [github.com/sponsors/draca-glitch](https://github.com/sponsors/draca-glitch)
- 🐛 **Open issues** for bugs, feature requests, or weird edge cases you hit in the wild
- 🛠️ **Send PRs**: extra extractors for the ingest module, implement the Postgres backend (currently a stub), NLI-based contradiction detection, anything

Sponsorship goes directly toward keeping the development server running and toward more time spent improving Mnemos instead of doing my day job. Nothing you sponsor unlocks paid features. Mnemos stays one piece of software, fully open source, with everything in the public repo.

## License

MIT: see [LICENSE](LICENSE).

## Credits

Built on top of:
- [FastEmbed](https://github.com/qdrant/fastembed): multilingual e5-large ONNX embeddings
- [sqlite-vec](https://github.com/asg017/sqlite-vec): vector search in SQLite
- [Jina Reranker v2](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual): cross-encoder
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval): benchmark dataset
