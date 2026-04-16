# Usage

> Install, register with an MCP client, run from the CLI, configure the optional Nyx cycle.

## Installation

### Requirements
- Python 3.11+
- ~500MB disk for FastEmbed model + Jina reranker (auto-downloaded on first use)
- **No GPU required**: everything runs on CPU via ONNX

### Setup

```bash
git clone https://github.com/draca-glitch/Mnemos.git
cd Mnemos
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Register with Claude Code (or any MCP client)

The easiest path for Claude Code is the built-in CLI:

```bash
claude mcp add -s user mnemos /path/to/venv/bin/mnemos serve
```

That writes the server into your user-scoped MCP config (`~/.claude.json`) automatically.

For other MCP clients (Cursor, ChatGPT Desktop, Gemini, etc.), add the equivalent entry to whatever config file that client uses:

```json
{
  "mcpServers": {
    "mnemos": {
      "type": "stdio",
      "command": "/path/to/venv/bin/mnemos",
      "args": ["serve"],
      "env": {
        "MNEMOS_DB": "/home/you/.mnemos/memory.db",
        "MNEMOS_NAMESPACE": "default"
      }
    }
  }
}
```

Restart your AI client. The 4 hot-path memory tools (`memory_store`, `memory_search`, `memory_get`, `memory_update`) plus the 2 maintenance tools (`memory_bulk_rewrite`, `memory_list_tags`) will be available.

### Disable Claude's built-in automemory (important if you use Claude)

If you use Claude (Code, Desktop, or claude.ai) and you are adopting Mnemos, **turn off Claude's built-in automatic memory feature**. The two systems will otherwise compete: Claude's memory will start writing facts about you in parallel, the agent will receive memories from both stores in its context, and you will get duplicate or conflicting recall behavior with no clear single source of truth.

How to disable, depending on the surface:

- **Claude Code**: edit `~/.claude/settings.json` (user scope, applies globally) or `.claude/settings.json` (project scope) and set:
  ```json
  {
    "autoMemoryEnabled": false
  }
  ```
  The agent will then rely on whatever MCP-attached memory system (Mnemos) you have configured.
- **Claude Desktop / claude.ai**: in your account settings, disable "Reference past chats". This is a separate feature from Claude Code's `autoMemoryEnabled` flag and has to be turned off on its own. Mnemos lives outside Anthropic's infrastructure and your AI client should not be writing into a separate, opaque store at the same time.

Mnemos is designed to be the single, owned, queryable memory layer for your AI assistant. Two parallel memory systems is worse than either one alone, pick one and let it be the source of truth. If you want to keep using Claude's automemory and not adopt Mnemos, that is a perfectly fine choice; it just is not the configuration this project assumes.

## CLI usage

```bash
# Core CRUD
mnemos add --project dev "F:Mnemos uses sqlite-vec for vectors"
mnemos search "vector storage" --project dev
mnemos get 42
mnemos update 42 --importance 8
mnemos delete 42                    # archive
mnemos delete 42 --hard             # permanent

# Discovery / introspection
mnemos stats                         # active/archived counts per project
mnemos map                           # topic index by project + subcategory
mnemos digest --days 7               # recent memories
mnemos briefing                      # compact ~370-token session-start summary
mnemos embed-status                  # vector coverage report
mnemos doctor                        # health check (schema, FTS, embeddings, stale)

# Predictive priming (used by session hooks)
mnemos prime "current task description"

# Nyx cycle consolidation (requires LLM, see below)
mnemos consolidate                   # dry run, default phases
mnemos consolidate --execute         # apply changes
mnemos consolidate --nyx --execute   # include synthesis (Phase 5)

# MCP server (typically invoked by your AI client, not directly)
mnemos serve
```

## Optional: Nyx cycle consolidation

Mnemos can run a 6-phase weekly **Nyx cycle** that merges related memories, detects contradictions, and synthesizes cross-domain insights. The LLM-driven phases (Dedup, Contradict, Synthesize) need an API endpoint. **Phase 6 (Bookkeeping) always runs without an LLM** and handles vector cleanup, decay, and stale link pruning purely in SQL.

**Pick a smart model, not a fast one.** The Nyx cycle is intentionally a background job; it runs weekly (or whenever you trigger it) and the entire purpose is to take its time *thinking* about your memories. The quality of the consolidation, dedup, and cross-domain synthesis is bounded by how good the LLM is at reasoning, not by how fast it answers. A slow but capable model (Qwen 2.5 32B locally, Claude Sonnet/Opus, GPT-4o, DeepSeek R1) will produce much better results than a fast lightweight model. Latency does not matter here. Quality does. If your Nyx cycle takes 20 minutes to run instead of 2, that is fine, because it is running while you are asleep or doing something else.

**You do not need a GPU even for the LLM.** Modern 32B-class models like Qwen 2.5 32B, Llama 3.1 70B (quantized), or DeepSeek R1 distill variants can run entirely on CPU through Ollama or llama.cpp, as long as you have enough RAM (usually 32-64 GB depending on quantization). On a typical desktop or small server without a graphics card, expect generation speeds of around 1-5 tokens per second depending on model size and quantization, instead of the 50-100 tok/s you would get on a GPU. For Mnemos that is completely fine. The Nyx cycle does not care if Phase 2 takes 30 seconds or 5 minutes per merge cluster, as long as the merge is correct. **Mnemos remains GPU-free end to end**, including consolidation, if you choose a local model, or if you choose not to run the consolidation cycles at all (store and retrieve work without any LLM).

To enable LLM-powered phases, set environment variables for any OpenAI-compatible endpoint:

```bash
# OpenAI
export MNEMOS_LLM_API_URL=https://api.openai.com/v1/chat/completions
export MNEMOS_LLM_API_KEY=sk-...
export MNEMOS_LLM_MODEL=gpt-4o-mini

# Local Ollama (free, runs on your machine)
export MNEMOS_LLM_API_URL=http://localhost:11434/v1/chat/completions
export MNEMOS_LLM_API_KEY=ollama
export MNEMOS_LLM_MODEL=qwen2.5:14b

# Or use OpenRouter, DigitalOcean Gradient, Together.ai, Groq, Fireworks, etc.
```

Without these set, `mnemos consolidate` will skip LLM phases and only run bookkeeping. The core memory features (store/search/get/update plus bulk_rewrite/list_tags) **never require an LLM**. Mnemos itself only uses local CPU models for embeddings and reranking.

### Per-phase model routing

The default is one model everywhere, whatever `MNEMOS_LLM_MODEL` is set to. Per-phase overrides are first-class env vars:

```bash
export MNEMOS_LLM_MODEL=gpt-4o-mini                   # global default
export MNEMOS_LLM_MODEL_MERGE=gpt-4o-mini             # Phase 2 Dedup merge
export MNEMOS_LLM_MODEL_WEAVE=gpt-4o-mini             # Phase 3 cross-category links
export MNEMOS_LLM_MODEL_CONTRADICT=claude-sonnet-4.6  # Phase 4 classification
export MNEMOS_LLM_MODEL_SYNTHESIZE=claude-opus-4.6    # Phase 5 creative generation
```

Any unset phase variable falls back to `MNEMOS_LLM_MODEL`. Each phase call is **stateless inference**, the LLM receives the cluster or pair it needs to reason about, nothing else; nothing persists between calls; the memory state lives in the database, not in the model. The phase-specific tested recommendations (what actually got benchmarked for each phase and which models performed best) live in [`ARCHITECTURE.md#per-phase-model-choice-what-was-tested`](ARCHITECTURE.md#per-phase-model-choice-what-was-tested).

## Session hooks (Claude Code)

Inject relevant memories at session start automatically:

```bash
chmod +x scripts/mnemos-session-hook.sh
```

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{
      "type": "command",
      "command": "/path/to/mnemos/scripts/mnemos-session-hook.sh start"
    }]
  }
}
```

## Storage backends

Mnemos has a pluggable storage layer. Summary:

- **SQLite backend (default, atomic)**, class `SQLiteStore`. Single SQLite file, row + FTS + vector in one transaction, up to ~10K memories on SSD. What most users should run.
- **Qdrant scaling layer (split storage)**, class `QdrantStore`. SQLite remains authoritative for content, FTS, metadata, and relations; Qdrant mirrors the vector index for HNSW at scale. Use when you want HNSW performance at tens of thousands of memories or already run Qdrant for bulk external content (mail, docs, ebooks). Cross-system atomicity is lost by design; SQLite is the source of truth and Qdrant can be rebuilt from it.
- **Postgres backend (planned, atomic)**, class `PostgresStore`. Postgres + pgvector in the same database, same single-transaction guarantee as SQLite, adds ACID multi-tenancy and MVCC. Stub today; contributions welcome.

The retrieval pipeline is identical across backends; only the vector index location changes. Full details, the atomicity tradeoff, real-world scale observations, SQLite-only vs split-storage as a deployment decision, and a code example, in [`ARCHITECTURE.md#storage-backends`](ARCHITECTURE.md#storage-backends).

## Memory usage

Mnemos is CPU-only but loads real ONNX models into RAM. Summary:

- **With reranker (default):** ~1.5-1.7 GB resident. Canonical configuration, benchmark numbers reported on this.
- **Without reranker (`MNEMOS_ENABLE_RERANK=0`):** ~1-1.2 GB resident. Runs on a 2 GB+ Raspberry Pi. Trades ~0.5 pp of R@5 for ~500 MB less RAM.
- **Disk:** ~800 MB total for both ONNX models (e5-large embedder + Jina cross-encoder), downloaded once on first use and cached under `~/.cache/fastembed`.
- **Sub-1 GB hardware:** not designed for it out of the box. Swap to a smaller embedder (e.g. `BAAI/bge-small-en-v1.5`) via `MNEMOS_EMBED_MODEL` if you truly need 512 MB total; retrieval quality drops but Mnemos still runs.

Full per-component breakdown (Python + Mnemos, SQLite + sqlite-vec, embedder, reranker) and sub-1 GB configuration notes in [`ARCHITECTURE.md#ram-and-disk-footprint`](ARCHITECTURE.md#ram-and-disk-footprint).

## Multi-user / Auth

Mnemos core is single-tenant by design. All operations are scoped to a `namespace` (default `"default"`); multi-user deployments use different namespaces. **There is no auth in the storage layer**, authentication is a transport-layer concern (the MCP server you run, the HTTP API you build on top, whatever auth your gateway already handles). Same separation Postgres uses.

Most users land on single-tenant ("my AI assistant has a memory of me") and the namespace + filesystem permissions story is enough. Multi-tenant deployments wire up OAuth / JWT / API keys in front of the MCP server and map authenticated identities to distinct namespaces, the hooks are there, the policy is the deployment's call.

Full discussion (the "a system should only be as complicated as it needs to be" framing and the design consequences) in [`ARCHITECTURE.md#multi-user-auth-and-why-the-storage-engine-stays-auth-free`](ARCHITECTURE.md#multi-user-auth-and-why-the-storage-engine-stays-auth-free).

## Beyond memories: one search across all your stuff

Mnemos started life as a curated memory store, but the same retrieval pipeline (FTS5 + vector + RRF + reranker) works for **any text content you want to search semantically**. The `mnemos ingest` command turns Mnemos into a unified semantic search layer for everything you might want your AI assistant to look through, not just CML facts you typed in by hand.

Out of the box you can index:

```bash
# A folder of notes (Obsidian, Logseq, Joplin, plain markdown, anything)
mnemos ingest ~/notes --project notes --pattern "*.md" --recursive

# A code repository
mnemos ingest ~/projects/myapp --project code --pattern "*.py" --recursive --chunk 1500

# Documentation
mnemos ingest ~/docs --project docs --recursive

# A single file
mnemos ingest ~/important-doc.txt --project reference
```

For sources that are not already plain text on disk (IMAP mailboxes, SQL databases, web APIs, application stores like Joplin or Logseq), the realistic pattern is **two-step**: a small extractor script pulls the source into a local cache (a folder of `.eml` files, a SQLite table, a JSON dump, whatever fits), and then `mnemos ingest` indexes that cache. My own production setup runs cron extractors every 30 minutes that pull IMAP mailboxes into per-account SQLite databases, and Mnemos searches across all of them through a custom extractor registered in the ingest pipeline. The two-step pattern keeps the ingest layer simple (everything is text-on-disk) while letting you pull from anything that has an API.

After ingestion, the same `memory_search` MCP tool finds the new content alongside your curated memories. Your AI client can search across **memories, notes, code, and docs in one query**. Project filters keep them separate when you want, but a default search hits everything.

Built-in extractors handle plain-text formats (txt, md, py, js, json, yaml, html, sql, and a dozen more). For binary or structured formats like PDF, EPUB, or eml, you can register a custom extractor without modifying core:

```python
from mnemos.ingest import register_extractor
import pypdf

def extract_pdf(path):
    reader = pypdf.PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)

register_extractor(".pdf", extract_pdf)
```

For very large bulk content (mail archives, document collections of 100K+ items), bypass `mnemos ingest` and write directly against the storage layer for maximum throughput:

```python
from mnemos.storage import SQLiteStore
from mnemos.storage.base import Memory
from mnemos.embed import embed

store = SQLiteStore()
for doc in your_bulk_source:
    text = extract_text(doc)
    embedding = embed([text], prefix="passage")[0]
    store.store_memory(
        Memory(project="mail", content=text, subcategory=doc.folder),
        embedding=embedding,
    )
```

This is exactly how I run Mnemos in production: a single SQLite store for curated memories plus several Qdrant collections for bulk indexed mail, project documents, personal notes, ebooks, and work files (8 collections, ~500K vectors total). All searched through the same Mnemos retrieval pipeline (BM25 + vector + RRF + cross-encoder), all from the same 4 hot-path MCP tools, all CPU-only.

**Mnemos is not just a memory system. It is a unified, local, CPU-only semantic retrieval layer for everything you own that has text in it.**
