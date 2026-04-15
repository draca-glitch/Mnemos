<p align="center">
  <img src="assets/mnemos-logo.svg" alt="Mnemos" width="220">
</p>

# Mnemos

> **The last memory you'll ever need.**
>
> A persistent memory system for AI agents.
> Named after Mnemosyne (Greek: μνήμη, *memory*).

**Runs on any computer. Works with any AI.**

**[→ Five-minute Quickstart](QUICKSTART.md)** if you just want to install and try it.

- **CPU-only**: no GPU required. Embeddings and reranking via ONNX models that run on a regular laptop, NAS, Raspberry Pi 4+, or budget VPS. Even the optional consolidation LLM can run locally on CPU in "slow mode" (1-5 tok/s on a 32B model with enough RAM) since the Nyx cycle is a background job and quality matters more than speed
- **MCP-native**: works with Claude Code, Cursor, ChatGPT Desktop, Gemini, or any MCP-compatible AI client out of the box
- **CLI-friendly**: a full `mnemos` command-line tool ships alongside the MCP server, so you can store, search, ingest, and consolidate from any shell, script, or cron job, with or without an AI client attached
- **100% local**: no API calls, no telemetry, no cloud dependencies. Your memory stays on your machine
- **Pluggable everywhere, true open source, the way it was supposed to mean.** Swap any of the four backbones independently. Storage (SQLite backend by default, atomic single-file; Qdrant scaling layer for HNSW vector search at 25K-plus memories; Postgres backend planned as an atomic server-hosted option); embedder (any FastEmbed-compatible model via `MNEMOS_EMBED_MODEL`); reranker (any cross-encoder via `MNEMOS_RERANKER_MODEL`); consolidation LLM (any OpenAI-compatible endpoint via `MNEMOS_LLM_API_URL` + `MNEMOS_LLM_MODEL`, or skip entirely). Defaults are picked, not mandated. Not "open code, locked backbone", open code and a stack you can take apart and reassemble on your own terms.
- **Benchmarked**: **98.94% Recall@5** on LongMemEval with the cross-encoder reranker enabled, no preprocessing of test data, no parameter tuning, first run. The lighter BM25+vector-only mode (no rerank at all) posts **98.30%** on the same benchmark, and the canonical configuration with cemelification (the way Mnemos is intended to run) measures **99.15%**. All three are first-run results. Mnemos gets there without an LLM in the search path and without vendor lock-in. The result files in [`benchmarks/`](benchmarks/) are how you verify it yourself.

> **Why v10 in a brand-new repository?** Mnemos has been in private production for months. Each of the nine internal versions involved real experimentation: adding features, removing the ones that did not pull their weight, evaluating retrieval quality, and iterating on what actually made the system smarter rather than just bigger. v10 is the state that was running on the day I decided to publish, the existing architecture, a packaging and documentation pass, and the benchmark numbers this README cites, all generated from that same code. As for the repository history: I had never used git for coding before this project. The entire v1 to v9 history of Mnemos lived as plain files on a home server, iterated on manually, never under version control. I opened my first GitHub account and pushed this repository in April 2026. The system has months of internal history behind it. The repository does not. See [CHANGELOG.md](CHANGELOG.md) for the incomplete timeline; I was not used to professional coding workflows and the changelog only captures what I happened to write down at the time.

> **Why "Mnemos"?** The first reason is straightforward: **Mnemosyne** (μνημοσύνη) is the Greek goddess of memory and mother of the nine Muses. Her name literally means "remembrance". A memory system named after her writes itself.
>
> The second reason is a wink. **Mythos** is the rumored name of Anthropic's next Claude model. Mythos tells stories, Mnemos remembers them. Same Greek mythology bench, same family of words, complementary roles: if Mythos becomes the model that powers your AI assistant, Mnemos is the memory it draws from. The naming was already in place before any of that surfaced publicly, but I was not going to pretend the pairing was not too good to keep. To be clear: this project has no affiliation with Anthropic. It is appreciation, not partnership. Just two names from the same root, doing the two things memory and storytelling have always done together. And I am really looking forward to Mythos. If it lands the way I am hoping, the pairing of a storytelling-oriented Claude on top of a curated Mnemos store is exactly the kind of setup I would want to use myself. So if anyone at Anthropic is reading this, I'd love to evaluate it. I promise only to use it for good.

## Origin: scratching a real itch, then "wait, hold my beer"

Mnemos started because the default `memory.md` approach felt deeply inefficient and dumb. A flat markdown file that Claude (or any AI) reads on every session is fine for a handful of facts, but it does not scale, it does not rank, it does not forget anything that should be forgotten, and it does not actually let the model *remember the user* in any meaningful way. I wanted my AI assistant to know who I am, what I care about, what I have decided, what I prefer, the way a long-term colleague would. Not just to re-read a static text file at every session start.

So I started building my idea of a memory system. Over months it grew into a real retrieval system: FTS5 for lexical search, sqlite-vec for semantic search, RRF fusion, a cross-encoder for high-precision dedup, exponential temporal decay, a weekly Nyx cycle for consolidation. It has been running in private production powering personal AI agents for months before this repo existed. **I was never planning to publish it.** It was just my own infrastructure, built for my own use, by someone who got tired of `memory.md`. In fact I imported my entire ChatGPT history into it just recently, just to see what would happen, and watching the Nyx cycle quietly stitch together patterns across years of unrelated conversations was the moment I realized this thing was doing something I had not seen anywhere else.

Then [MemPalace](https://github.com/MemPalace/mempalace) surfaced, claiming state-of-the-art retrieval with a hierarchical memory metaphor. I read their README and benchmarks and had the strangest reaction: *"wait, hold my beer."* I had never benchmarked Mnemos against anything before, because until a few days ago I did not even know LongMemEval existed; I had just been building the system for my own use and trusting my own gut about whether it was getting better. So I pointed Mnemos at the same LongMemEval dataset MemPalace had used. First run, no tuning, no preprocessing of the benchmark data: **98.94% Recall@5** on verbatim LongMemEval transcripts.

That result is what tipped this from "private side project I had no plans to share" into "I should clean this up and put it on GitHub for public scrutiny". If you are reading this README, the MemPalace comparison is the reason it exists at all.

> **A note on scope.** Mnemos is a personal memory system. It is not trying to compete with production-grade multi-tenant memory services like [Mastra](https://mastra.ai), [Mem0](https://mem0.ai), [Supermemory](https://supermemory.ai), [Hindsight](https://hindsight.ai), and the other commercial systems in this space. Those are built for teams, hosted APIs, enterprise SLAs, and commercial support contracts, and Mnemos will not hold a candle to them on that axis. **For personal Claude (or any MCP-compatible AI) memory on a single machine, though, it is more than enough, and that is the audience it is built for.** If you are running memory for a support-agent fleet with thousands of customers, reach for one of the commercial options instead. The benchmark comparisons below are about the retrieval-recall axis specifically, the one axis where a personal tool and a multi-tenant service can meaningfully be measured against each other, not a claim that Mnemos replaces what they do.

I am comparing against MemPalace specifically because they are, by their own account and by the articles written about them, the current number one in the AI personal memory space. If you are going to benchmark yourself, you benchmark against whoever is on top. This is not personal; it is just how comparisons work. Here is how the two systems actually stack up:

|                              | MemPalace | **Mnemos** |
|---|---|---|
| Storage backend              | ChromaDB only | **SQLite (default), Qdrant, Postgres-ready** |
| Vector search                | ChromaDB defaults | **FastEmbed e5-large (1024-dim, ONNX)** |
| Lexical search (BM25)        | None | **FTS5 with stemming + AND/OR fallback** |
| Hybrid retrieval             | None | **RRF fusion (k=60)** |
| Cross-encoder reranking      | None | **Jina Reranker v2 multilingual** |
| Deduplication                | None | **3-way: FTS + CML + vector → reranker** |
| Temporal decay               | None | **Exponential, separate half-lives for episodic/semantic** |
| Validity windows             | None | **`valid_from` / `valid_until` per fact** |
| Contradiction detection      | None | **Auto-detect on store + memory_links** |
| Hierarchical organization    | Wing/Room metadata | **project/subcategory metadata** |
| MCP tools                    | 19 (manual navigation) | **4 (CRUD + search)** |
| Consolidation                | Mining modes | **6-phase Nyx cycle (LLM-driven)** |
| Auto-widen on thin results   | None | **Cross-project fallback** |
| **LongMemEval R@5**          | **96.6%** raw ChromaDB mode (no LLM in retrieval, their reproducible baseline)<br>**100%** with Haiku LLM reranker (~500 outbound API calls to Anthropic per benchmark run, requires API key, network round trip per query, not air-gappable, rerank pipeline not yet in their public benchmark scripts as of April 2026) | **99.15%** canonical BM25+vector+rerank+CML (the way Mnemos is intended to run)<br>**98.94%** BM25+vector+rerank (clean, no preprocessing)<br>**98.30%** BM25+vector lite (no rerank, no CML)<br>**All three configurations: 0 API calls, fully local, runs air-gapped.** The Jina cross-encoder is a local ONNX model, not a remote LLM<br>**No-LLM-in-search-path comparison** vs MemPalace raw 96.6%: roughly **50% / 69% / 75% fewer errors** across the three Mnemos tiers |

The article I read about MemPalace, the one that changed my mind on whether to release my system or not, talked about it like it was the next best thing since AI sliced bread. When I actually looked at the architecture, my reaction was different. To me MemPalace sounds like overly complex mimicry of how the human brain organizes memories spatially, when the AI on the other end neither wants nor needs a spatial metaphor to retrieve information. It just needs a good search, and the complexity may carry more structural overhead than the retrieval task requires. Their headline "+34% palace boost" is, by their own transparency note, what you get from any standard ChromaDB metadata filter.

The apples-to-apples comparison is **no generative LLM in the search path on either side**: Mnemos clean (BM25+vector+rerank) **98.94%**, Mnemos canonical (BM25+vector+rerank+CML) **99.15%**, even Mnemos lite (BM25+vector only, no rerank at all) **98.30%**, all vs MemPalace raw ChromaDB **96.6%**. The Mnemos cross-encoder (Jina v2) is a local discriminative ONNX model, not a remote LLM; it runs on CPU in ~50 ms per query, no network, no API key, no rate limits, fully air-gappable. The Mnemos numbers are first-run, no parameter tuning. **Even our lightest configuration (no rerank, no CML, the kind that runs on a Pi) clears MemPalace's strongest reproducible no-LLM number by 1.7 percentage points.** MemPalace also publishes a stronger configuration that adds Claude Haiku as a generative LLM reranker in the search loop, scoring **100%** on the same dataset; that result requires a per-query call to Anthropic's API (~500 outbound calls per benchmark run, not air-gappable, rerank pipeline not in their public benchmark scripts). It also lives in a different metric class entirely, see the methodology note below. The [benchmark section](#benchmark) has the full per-mode methodology.

> **A note on "100% R@5".** A perfect score on a real-world IR benchmark with hundreds of diverse questions is unusual. In standard information-retrieval research, top systems on hard public datasets like LongMemEval typically land in the high-90s, not at 100%, because the benchmarks contain genuinely ambiguous questions where the "correct" session is contested. **A 100% claim on LongMemEval almost always means the LLM in the rerank step is effectively *answering* each question rather than scoring candidates**, at which point the metric is no longer pure retrieval recall, it is QA accuracy with retrieval as a preprocessing step. That is a legitimate metric to chase, but it is not the same metric as Mnemos's 98.30% / 98.94% / 99.15%. Mnemos deliberately does not put a generative model in the retrieval pipeline, partly because the architecture proves you do not need one to land in the high-90s, and partly because once you do, "retrieval recall" stops measuring retrieval and starts measuring LLM inference quality.

> **Credit where it is due**: MemPalace published a [transparency note in April 2026](https://github.com/MemPalace/mempalace) acknowledging and correcting several earlier overclaims in their original release (compression ratios, palace-boost framing, contradiction-detection wiring, etc.). The numbers cited above reflect their current published state. That kind of openness is exactly how open source is supposed to work, and the technical comparison here is in the same spirit: head-to-head on architecture and benchmark numbers, not on marketing.

> **And to be equally honest about Mnemos**: Mnemos does use an LLM, but only in the optional Nyx cycle (the weekly background consolidation that merges duplicates, detects contradictions, weaves cross-category links, and synthesizes cross-domain insights). **For the core memory operations, storing a memory and retrieving it, no LLM is involved at any point.** Store goes directly through the dedup pipeline (FTS + CML subject + vector + cross-encoder), embed, and SQLite write, no generative model touched. Retrieve goes through FTS5 + sqlite-vec + RRF + cross-encoder rerank, no generative model touched. The LLM only enters when you explicitly invoke `mnemos consolidate`, and even then only for phases 2 through 5; phase 6 (bookkeeping) and phase 1 (triage) work without one. The Nyx cycle can run against a local model (Ollama, llama.cpp) if you want the whole stack air-gapped, or it can be skipped entirely. **Without the LLM, you still get the full memory system**, store, retrieve, BM25 + vector search with or without the cross-encoder, on-store dedup, on-store contradiction detection, temporal decay, validity windows, auto-widen, all storage backends, the lot. What you give up is the adaptive layer: the Nyx cycle's batch dedup of similar memories accumulated over weeks, the cross-domain synthesis that builds Mnemos's "knowledge about the user" over time, and the long-running consolidation that turns a sprawling raw store into a curated one. The static core is fully functional without it; the adaptive enrichment is what the LLM unlocks. The "no LLM in the search path" framing above refers specifically to the retrieval pipeline, which is genuinely LLM-free per query, every query, regardless of whether the Nyx cycle has ever run.

Mnemos skips the metaphors and ships the retrieval pipeline that actually scores. **Fewer tools, deeper plumbing, more honest defaults.** An AI neither needs nor wants to function like a human brain. **It does not need a palace, it needs efficiency.**

```
┌─────────────────────────────────────────────────────────────┐
│  No LLM in search path             R@5     Errors            │
├─────────────────────────────────────────────────────────────┤
│  Mnemos canonical (CML+rerank)     99.15%   0.85% ← this repo│
│  Mnemos clean (BM25+vector+rerank) 98.94%   1.06% ← this repo│
│  Mnemos lite (BM25+vector only)    98.30%   1.70% ← this repo│
│  MemPalace raw ChromaDB            96.6%    3.4%             │
│  Dense retrieval (paper baseline)  ~90.0%  ~10%              │
│  BM25 baseline (paper)             ~85.0%  ~15%              │
├─────────────────────────────────────────────────────────────┤
│  WITH generative LLM in search path (different metric class) │
│  MemPalace + Haiku rerank          100%     0%    ~500 API   │
│                                                    calls/run │
└─────────────────────────────────────────────────────────────┘
```

## What is Mnemos?

**A memory system should be a memory.** It stores data and retrieves it. That is the entire job.

The AI memory field seems to have gradually drifted toward treating memory as a reasoning surface. Based on what the major memory systems publish about their own architectures, most of the memory products people are shipping today appear to do significant reasoning work inside the memory layer itself: calling LLMs during retrieval, augmenting queries with inference steps, exposing nineteen tools where four would do, and burning tokens to redo work the agent on the other end of the conversation has already done. I have not audited every competitor's source code, nor do I intend to, and I do not claim to know exactly how each one behaves at runtime. But from their public docs, their benchmarks, and their tool surfaces, the field looks like it has slowly forgotten what a memory is, which is ironic.

Your brain does not run inference every time you try to remember where you put your keys. It runs lookup. The reasoning happens around the memory, not inside it. You think *"where did I put my keys?"*, you query memory, memory hands you back *"on the kitchen counter"*, and then you reason about what to do next. Memory itself is dumb, fast, and reliable.

Mnemos is built that way on purpose. The agent thinks. Mnemos remembers. The agent decides what is worth storing and how to phrase it; Mnemos receives those bytes and persists them. The agent decides what to ask about; Mnemos finds the right items and hands them back. There is no LLM call inside the retrieval or storage pipeline. The search path does use two small models (`e5-large` for embeddings and Jina v2 for cross-encoder reranking) but neither is a generative language model. They are discriminative scorers that take inputs and emit numbers, locally, on CPU, in milliseconds, without generating any tokens or making any API calls. When a query comes in, the answer just *happens*. No model is "working" to think it through. There is no reasoning step inside Mnemos that duplicates what the agent on the other side already did. There are four MCP tools (`store`, `search`, `get`, `update`) because those are the verbs a memory has. Everything else is the agent's job, not the memory's.

The result is a system that does the one thing it is supposed to do, extremely well:

- **Accurate.** **99.15% R@5** on LongMemEval in the canonical CML configuration the system was designed to run in. **98.94% R@5** with the benchmark data fed in verbatim: no preprocessing, no tuning, no LLM in the search path. Both are first-run numbers. I was not chasing a benchmark; the architecture I picked happened to also be the one that wins it. Full per-mode breakdown in the [benchmark section](#benchmark) below.
- **Efficient.** Four MCP tools fit in under 1.5 K tokens of system-prompt overhead per conversation, from start to end, compared to memory systems that ship nineteen or more tools and consume 5 to 8 K tokens of overhead before retrieving anything. Runs on a Raspberry Pi. CPU-only. Per-query cost is bounded by your local CPU, not by your LLM API bill. The 99.15% canonical number is achieved without an LLM "working" to search anything; the search just happens. The retrieval pipeline reads the bytes and emits the answer. See [CML](#cml-token-minimal-memory-format) below for the storage-side efficiency story.

Not a reasoning palace. Not an agent. Not a thinking creature. A memory.

Beyond the two-job principle, Mnemos also:

- **Curates memories** rather than storing verbatim chats: beliefs, decisions, preferences, learnings. *The curation decision is made by the agent calling `memory_store()`. Mnemos does not run an LLM to reinterpret or filter what the agent chose to remember.*
- **Speaks its own language, CML**: a token-minimal memory format that strips English connective tissue while keeping every fact. The compression you actually get depends on how dense the input was: narrative prose with lots of connective tissue condenses up to 60%, fact-dense production-style memories more like 15% to 25%. Think of it as a denser, no-filler language. *CML itself is a format. Mnemos stores whatever the agent writes with no LLM in between. Free-form prose input converges toward CML over time via the Nyx cycle, which is LLM-driven and runs weekly, never on the search path.* [More below](#cml-token-minimal-memory-format).
- **Combines lexical + semantic retrieval**: FTS5 BM25 + e5-large vectors + Reciprocal Rank Fusion + Jina cross-encoder reranking. *No LLM in the search path. FastEmbed e5-large and Jina v2 are local ONNX discriminative scorers, not generative models, they compute similarity and relevance, they do not generate text.*
- **Models human forgetting**: exponential temporal decay with separate half-lives for episodic vs semantic memories. No memory is ever actually deleted; old ones just rank lower in search until you bring them back. *Pure SQL and math, no LLM involvement.* See [Forgetting](#forgetting-nothing-gets-deleted-automatically) below.
- **Consolidates during "sleep"**: weekly LLM-driven Nyx cycle merges related memories into condensed representations, in the background, asynchronously, never on the search path. *LLM-driven. Configurable via `MNEMOS_LLM_MODEL`; skipped entirely if no LLM is configured (the rest of Mnemos still works).*
- **Detects contradictions automatically**: flags when new information conflicts with existing facts. *On-store real-time check uses vector similarity + same-project filter + Jina cross-encoder rerank, no LLM. A deeper Phase 4 contradiction scan runs during the weekly Nyx cycle and is LLM-driven; it only fires if an LLM is configured.*
- **Tracks temporal validity**: facts can have expiry dates and historical "valid_from" windows. *Pure schema and query-time filtering, no LLM involvement.*

## Architecture

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

On the default SQLite backend, everything (content, FTS index, 1024-dim vectors, memory links, Nyx consolidation history) lives in one SQLite file and every write is a single atomic transaction, a row, its FTS entry, and its vector either all land or all roll back together. The Qdrant scaling layer keeps SQLite authoritative for all text, FTS, and relations and mirrors the vector index to Qdrant for HNSW performance once you are past the tens-of-thousands-of-memories range. A Postgres backend is planned as a second atomic option (Postgres + pgvector in the same database, same single-transaction guarantee as SQLite but with ACID multi-tenancy and MVCC).

Full layered architecture, data model schema, retrieval pipeline mechanics, and the complete storage-backend tradeoff discussion live in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). This section is a visual summary.

## The 4 MCP tools

```python
memory_store(project, content, tags?, importance?, type?, subcategory?,
             valid_from?, valid_until?, verified?, layer?)
memory_search(query, project?, subcategory?, type?, layer?,
              valid_only?, search_mode?, limit?)
memory_get(id)
memory_update(id, [any field])
```

That's the entire surface. No `navigate_to_wing`, no `open_room`, no `list_halls`. Just CRUD plus search. Hierarchy is metadata, not architecture.

### Why 4 tools and not 45

I have always lived by the teaching: *teach me one way to do ten things, not ten ways to do one thing.* It applies to the kitchen knife you keep sharp instead of buying ten gadgets, the language you speak fluently instead of dabbling in five, the tool in your toolbox you actually know how to use, the keyboard shortcuts you can hit without looking, the APIs you design, and the memory system you trust your AI assistant to. PUBG showed us a frying pan can be used as a weapon, not just for frying things. Same energy.

Some memory systems expose upwards of **19 MCP tools** for navigating their internal metaphors: tools to enter wings, open rooms, list halls, traverse closets, follow tunnels, and so on. Reading a tool list that long honestly felt like sitting down to play *King's Quest* in 1984: *> look at door > open door > look in room > pick up key > use key with lock > open closet > look in closet*. Mnemos exposes **4**.

This is not about minimalism for its own sake. It is about how AI clients actually use tools:

1. **Every tool definition burns context tokens.** The full schema for 19 tools costs hundreds to thousands of tokens on every single request, on every single session, forever. Four tools is roughly a fifth of that overhead.
2. **More tools means more choice paralysis.** When the model has 19 ways to look something up, it has to reason about which tool fits, often picks suboptimally, and sometimes chains multiple navigation calls when one search would have answered the question. Four orthogonal tools (store, search, get, update) leave no room for ambiguity.
3. **Surface area is bug area.** Each tool is a contract you have to maintain, document, and not break. A pluggable storage backend is hard enough without 19 tool signatures pinned to a particular metaphor.
4. **The metaphor is not the system.** "Memory palace" is a mnemonic device for human memorization, not a database design pattern. Hierarchies are perfectly representable as `project` plus `subcategory` columns, filtered at query time. You do not need a tool called `enter_wing()` for that. It is metadata.

The four Mnemos tools cover the entire CRUD-plus-search surface that any memory system needs. Hierarchical filtering, type filtering, validity windows, namespaces, layers, and rerank modes are all parameters on the existing search tool, not new tools. Adding capability means adding optional parameters, never new tools.

If you ever feel constrained by four tools, the right reaction is "what parameter is missing from `memory_search`", not "I need a `memory_traverse_subcategory_tunnel` tool". So far the answer has always been a parameter.

## Features

### Hybrid retrieval pipeline
1. **FTS5 BM25** with AND-default + OR-fallback for multi-term queries
2. **Vector similarity** via FastEmbed e5-large (1024-dim, ONNX, CPU-native, ~7ms/embed)
3. **RRF fusion** (Reciprocal Rank Fusion, k=60) merges both rankings
4. **Cross-encoder reranking** (Jina v2 multilingual) for final precision. **The cross-encoder is a local ONNX discriminative scorer, not a remote LLM**, it runs on CPU in ~50 ms per query, no API call, no network round trip, no API key required. **Enabled by default** in the public package, but the lite mode without it (`--mode hybrid`) already lands at **98.30% R@5** on LongMemEval, in the same tier as any other verified no-LLM number in the field. The cross-encoder pushes the canonical configuration up to **99.15% R@5**, it is the cherry on top, not a hard requirement to be competitive. Disable with `MNEMOS_ENABLE_RERANK=0` or `Mnemos(enable_rerank=False)` if you want the smallest possible RAM footprint; you trade about 0.85 percentage points of R@5 for ~500 MB less resident memory.

### Why these specific models (and how to swap them)

The embedding model and the reranker are both **swappable**. Mnemos talks to FastEmbed-compatible models for embeddings and to any cross-encoder loadable through FastEmbed for reranking, so you can plug in whatever you prefer. Every benchmark number in this README is reported on the default e5-large embedder + Jina v2 reranker pair, and changing either side has not been measured.

#### Embedder

**`intfloat/multilingual-e5-large`** was picked for very specific reasons:

- **Truly multilingual**, not English-with-token-mapping. The memory store itself defaults to English CML by convention (the agent writes in English; the Nyx cycle keeps it that way), but the same retrieval pipeline also indexes external data sources via `mnemos ingest` (ebooks, notes, documents, PDFs) which contain Swedish, English, and other languages depending on the source. The semantic match needs to work *across* languages too: an English query like "breakfast" should match a Swedish ebook chapter about "frukost". e5-large handles 100+ languages in the same vector space, which is rare for high-quality models.
- **1024 dimensions**, a sweet spot I tested my way into. High enough to capture nuance, low enough that brute-force search in SQLite stays fast and storage stays reasonable.
- **Available as ONNX**, runs on CPU at ~7 ms per embedding without a GPU runtime or PyTorch dependency.
- **Apache 2.0 license**, no surprises.
- **Battle-tested** on MTEB and many production retrieval systems.

Swap with `MNEMOS_EMBED_MODEL`. Smaller models (BGE-small, all-MiniLM) are faster but less precise. Larger models (BGE-large, GTE-large) are slower but may score higher on specific benchmarks. The rest of the pipeline does not care.

If you want to run a 4096-dim embedder (e5-mistral-7b, NV-Embed-v2, SFR-Embedding-2_R) or even an 8192-dim Matryoshka model (Stella-en-1.5B-v5 at full resolution), set `MNEMOS_EMBED_MODEL` to whatever ONNX-exposed model you want and `MNEMOS_EMBED_DIMS` to match. Storage scales up (a 4096-dim float32 vector is 16 KB per memory vs 4 KB for the 1024-dim default), per-query embed latency moves from ~7 ms to seconds (those 7B-parameter generative-style embedders are slow on CPU), and you need more RAM. The retrieval recall lift over e5-large is real but small (typically 1-2 points on hard benchmarks). For most users the trade is not worth it. For the ones for whom it is, the swap is one env var.

#### Reranker

**`jinaai/jina-reranker-v2-base-multilingual`** was picked for matching reasons:

- **Multilingual** (same language coverage as the embedder, important because the reranker and embedder see the same text).
- **ONNX-friendly**, CPU-only, no GPU runtime required.
- **Small**: sub-300 MB on disk, ~400-500 MB resident.
- **Permissive license** (Apache 2.0).
- **Tuned for query-document relevance scoring**, not generic similarity, which is the signal reranking actually needs.

Swap with `MNEMOS_RERANKER_MODEL`, or disable entirely with `MNEMOS_ENABLE_RERANK=0` if you want the smallest possible RAM footprint (trades ~0.5 pp of R@5 for ~500 MB less resident).

Could a stronger reranker like Jina Reranker v3 push the numbers higher? Possibly, but it is currently not available as an ONNX export, would need a different runtime than FastEmbed, is a 0.6B-parameter model (about 2× the size of v2), and its realistic deployment story involves GPU acceleration in most cases. That would break the "runs on a Raspberry Pi" claim and the entire footprint argument this README is built on. Mnemos is intentionally **CPU-only and non-GPU** as a hard design constraint. If a community-maintained Jina v3 ONNX export ever lands and benchmarks well on CPU at acceptable latency, swapping it in would be the natural next step.

### Temporal modeling
- **Exponential decay** with separate half-lives:
  - Episodic memories: ~46 days (events, conversations)
  - Semantic memories: ~180 days (distilled knowledge)
- **Validity windows**: facts can have `valid_from` / `valid_until` for time-bound truths
- **`valid_only` filter**: exclude expired facts at query time

### Hierarchical organization
- Top-level projects are free-form strings. A starter set is provided (`dev`, `finance`, `food`, `health`, `home`, `personal`, `relationships`, `server`, `travel`, `work`, `writing`) but you can add or remove categories to match how you think about your memory. The storage layer does not enforce the list.
- Optional sub-categories per project (e.g., `dev/myapp`, `finance/crypto`)
- Free-form, no schema migration needed

### Nyx cycle consolidation (the optional LLM-driven part of Mnemos)

Modeled on brain sleep stages, runs weekly. **This is the only place in Mnemos where an LLM is involved at all**, store and retrieve never touch one. The Nyx cycle is opt-in (you invoke `mnemos consolidate`) and can be skipped entirely; without it, Mnemos still works as a complete memory system, you just don't get the adaptive enrichment layer.

| # | Phase | What it does | Uses LLM? |
|---|---|---|---|
| 1 | **Triage** | detect new memories since last run | ✗ no, pure SQL |
| 2 | **Dedup** | merge near-duplicates (cosine ≥0.88 tight, ≥0.75 topic); clusters larger than 2 memories are merged via hierarchical pairwise steps to preserve specifics at depth | ✓ yes |
| 3 | **Weave** | find cross-category relationships, create `memory_links` | ✓ yes |
| 4 | **Contradict** | classify same-topic memory pairs across time: supersedes, evolves, contradicts, or compatible; update `memory_links` and `valid_until` markers accordingly | ✓ yes |
| 5 | **Synthesize** | generate cross-domain insights | ✓ yes |
| 6 | **Bookkeeping** | decay old memories, cleanup orphans, prune stale links | ✗ no, pure SQL |

Phases 2 through 5 require an OpenAI-compatible endpoint (`MNEMOS_LLM_API_URL` + `MNEMOS_LLM_MODEL`). Any model works: OpenAI, Anthropic, local Ollama / llama.cpp, OpenRouter, DigitalOcean Gradient, Together.ai, Groq, Fireworks, anything that speaks the chat-completions protocol. **If no LLM is configured, phases 2-5 are skipped automatically with a warning, and phases 1 and 6 still run.** Phase 1 (Triage) and Phase 6 (Bookkeeping) are pure SQL housekeeping that work on every deployment regardless of LLM availability.

**Per-phase model routing.** The default is one model everywhere, whatever `MNEMOS_LLM_MODEL` is set to. Per-phase overrides are first-class env vars:

```bash
export MNEMOS_LLM_MODEL=gpt-4o-mini           # global default
export MNEMOS_LLM_MODEL_MERGE=gpt-4o-mini      # Phase 2 Dedup merge
export MNEMOS_LLM_MODEL_WEAVE=gpt-4o-mini       # Phase 3 cross-category links
export MNEMOS_LLM_MODEL_CONTRADICT=claude-sonnet-4.6  # Phase 4 classification
export MNEMOS_LLM_MODEL_SYNTHESIZE=claude-opus-4.6    # Phase 5 creative generation
```

Any unset phase variable falls back to `MNEMOS_LLM_MODEL`. Each phase call is **stateless inference**, the LLM receives the cluster or pair it needs to reason about, nothing else; nothing persists between calls; the memory state lives in the database, not in the model. The phase-specific tested recommendations (what actually got benchmarked for each phase and which models performed best) live in [`docs/ARCHITECTURE.md#per-phase-model-choice-what-was-tested`](docs/ARCHITECTURE.md#per-phase-model-choice-what-was-tested).

### Adaptive learning: Nyx, how Mnemos gets to know you

> **A note on the name.** **Nyx** (Νύξ, *"Night"*) is the Greek primordial goddess of Night, older even than Zeus. In the Theogony she is the mother of Hypnos (sleep) and the Oneiroi (dreams), among others. Where Mnemosyne is memory itself, Nyx is the quiet time when memory gets worked on. Naming the consolidation cycle after her is not ornamental: it literally is the process that happens in the dark while nothing else is using the system, consolidating, weaving, and letting the day's input settle into something that will be retrievable tomorrow.

The Nyx cycle is not just maintenance, it is where Mnemos actually learns about the user behind the memories. Phases 3, 4, and 5 quietly build a model of how you think, what you change your mind about, and how your interests connect to each other:

- **Weave** (Phase 3) finds semantically-close memories across different projects and stores the link. Over weeks this forms an implicit graph of "things this user mentally connects" that enriches future search results.
- **Contradict** (Phase 4) catches slow-burn temporal evolution, beliefs or preferences that shifted over time. Both versions stay queryable; the older gets a `valid_until` marker and a link to its successor. Nothing is silently overwritten.
- **Synthesize** (Phase 5) feeds clusters of related memories into the LLM and asks for novel cross-domain observations: themes, recurring concerns, patterns you might not have noticed yourself. The insights are stored as new `semantic` memories and participate in future searches like any other fact.

Run Mnemos for a few months and the surprising thing happens: it starts knowing things about you that you do not know about yourself. Not in an oracle way, just the mundane way any long-running pattern detector eventually outpaces a human looking at the same data. It sees the link between your sleep notes and your bad-mood entries before you do. It surfaces the recurring concern you have been brushing off for six months. It notices the friend you only ever mention when you are tired. The synthesizer is writing notes about you for later use, and some of those notes are uncomfortably accurate.

Opt-in and entirely local. The Nyx cycle only runs when you invoke `mnemos consolidate --execute`, and only the LLM-powered phases need an API endpoint. Phase 1 (Triage) and Phase 6 (Bookkeeping) are pure SQL and always run. If you never invoke consolidation at all, Mnemos behaves like a static memory store with no adaptive layer and loses nothing else.

Per-phase mechanism details (what each phase actually does at the SQL + LLM level), the protection filters (`importance >= 9`, `evergreen` tag, `consolidation_lock`), the hierarchical pairwise merge mechanics, and the tier-2 recall architecture live in [`docs/ARCHITECTURE.md#the-nyx-cycle`](docs/ARCHITECTURE.md#the-nyx-cycle).

> Personal note: I implemented the Nyx cycle in Mnemos v8 in February 2026, several weeks before Anthropic shipped any equivalent background-consolidation behavior into Claude. I genuinely laughed when their announcement landed and I realized I had quietly been running the same idea on my own server for weeks. Except mine works better. Hear that, Anthropic? I do not say any of this to claim invention. The "memory consolidates during sleep" concept is borrowed straight from neuroscience and a dozen prior research papers. I just want to be clear that this part of Mnemos was built independently and predates the closest commercial parallel I know about. Because a good idea is a good idea, and some things are so objectively right that everyone working on the problem ends up in the same place.

### Contradiction detection (real-time, on store)
Every `memory_store()` call checks the new memory against existing facts on the same topic via vector similarity + same-project filter + cross-encoder rerank (no LLM in the loop). Conflicts surface in the response immediately and persist as `memory_links` with `relation_type='contradicts'` for the next Nyx cycle's Phase 4 batch pass to consider. Full mechanism in [`docs/ARCHITECTURE.md#real-time-contradiction-detection-on-store`](docs/ARCHITECTURE.md#real-time-contradiction-detection-on-store).

### Auto-widen
If a project-filtered search returns fewer than 3 results, Mnemos re-runs the same pipeline without the project filter. Cheap safety net for the common "I stored it under a different project than I'm searching from" case. Details in [`docs/ARCHITECTURE.md#auto-widen`](docs/ARCHITECTURE.md#auto-widen).

### 3-way deduplication
Before storing, Mnemos pools candidates from three independent signals (FTS5 keyword overlap, CML-subject match, vector similarity) and runs them through the Jina cross-encoder for a final confidence score. Three signals catch different failure modes; any one of them missing a duplicate is fine. Details in [`docs/ARCHITECTURE.md#3-way-deduplication-on-store`](docs/ARCHITECTURE.md#3-way-deduplication-on-store).

## CML: token-minimal memory format

Every memory in Mnemos is text that an AI client will eventually read into its context window. Tokens are the actual currency of any LLM-backed system, and the memory store is the place where those tokens accumulate forever. A bloated memory format costs you context budget on every single retrieval, on every single session, for the entire lifetime of the project. Multiplied by thousands of memories, the difference between a verbose format and a condensed one is enormous.

So Mnemos uses **CML (Condensed Memory Language)** as a soft convention for writing memories. CML is not a parser, not a schema, not an encoder, not a compressor. It is just a tiny set of prefixes and symbols that the writer (you, or preferably the AI assistant on your behalf) uses to pack common semantic patterns into the smallest number of tokens that still preserves meaning. The dedup pipeline understands the conventions well enough to flag conflicts on the same subject, but nothing in Mnemos compiles, validates, or transforms CML. It is just text that happens to be denser.

> **A disclaimer.** I am not claiming CML is the best possible notation for this problem. What I *am* claiming is that **compression is not the answer; efficient language is.**
>
> The mechanism matters. AAAK and similar compression approaches work **on text**: they apply regex-based abbreviation rules and entity codes (`KAI` for `Kai`, etc.) to strings that already exist. The retriever then has to score against text that no longer matches the original surface form. BM25 loses its tokens, the bi-encoder embeds something it never saw at training time, and the cross-encoder has no idea what `KAI` means. That mechanism is why AAAK regresses retrieval on LongMemEval (84.2% R@5 vs raw 96.6%, by MemPalace's own published numbers).
>
> CML works **on the writer**, not on the text. The agent or LLM is instructed to write in CML notation in the first place. No transformation step, no encoder, no decoder, no entity table, no regex pass. Every token in a CML memory is recognizable English (`L: webhook sig needs raw body @FastAPI ∵ middleware ate stream → fix: disable body parse` reads as plain English to a human and as a structurally-tagged learning to an LLM, with no decoding step in either case), so FTS5 indexes the original tokens, the bi-encoder embeds text it can handle, and the cross-encoder scores against text that means what it says. **Compression removes signal the retriever needs. Efficient language preserves it while saying less.**
>
> There are almost certainly better notations to discover, or ways to extend CML further that would sharpen it beyond what it does today. This is where my own exploration has reached so far. If you find something better, open an issue; the grammar is small enough to evolve without breaking anything downstream.

> **CML is not a theoretical claim, it was tested.** The question "does the retrieval pipeline actually work as well on CML-stored memories as on plain-prose memories?" is answered by the `--cml` LongMemEval modes in [`benchmarks/longmemeval_bench.py`](benchmarks/longmemeval_bench.py). Each LongMemEval session is cemelified via a one-time Claude Haiku call (cached on disk by SHA256 hash), stored in the normal Mnemos pipeline, and then the full benchmark runs against that CML-formatted corpus. **`--mode hybrid --cml` scores 98.09% R@5**, essentially tied with the no-CML lite mode at 98.30% despite the bi-encoder being mildly out of distribution on structural CML. **`--mode hybrid+rerank --cml` scores 99.15% R@5**, higher than the no-CML `hybrid+rerank` at 98.94%, because the cross-encoder handles CML's structural prefixes and operators as explicit relation markers. Both result JSONs are committed: [`results_hybrid_session_cml.json`](benchmarks/results_hybrid_session_cml.json) and [`results_hybrid+rerank_session_cml.json`](benchmarks/results_hybrid+rerank_session_cml.json). Numbers stand up: CML is not a compressor the retriever has to decode, it is just denser English text, and the pipeline handles it cleanly. The orthogonal question, "does the MERGE prompt produce valid CML consistently across LLM tiers?", is answered by the separate [Consolidation quality bench](#consolidation-quality-fact-preservation-in-the-nyx-cycle) further down.

> **Brief aside: I had to invent a new English word to write this README.** The verb is **`cemelify`**, pronounced as `See-M-El` + `ify`, which is the three letter names of `CML` spoken aloud and condensed into a single syllable. It means "rewrite verbose text in CML notation". To be clear, this is not a translation in the language-to-language sense; it is more like a compaction, or a "remove redundant words" function. The output is still recognizable English, just denser, with the connective tissue stripped out and replaced with a small set of prefixes and operators that an LLM (or a human) can read directly. I wanted a single verb for this action because writing out "rewrite in CML form" everywhere felt clumsy, and existing verbs like *condense*, *canonicalize*, or *densify* are all close but none of them include the structural prefix-tags-and-operators part that CML adds on top of just shortening. So I made one up. The README uses `cemelify` / `cemelifies` / `cemelified` / `cemelification` everywhere it talks about converting prose into CML form, which is mostly something the Nyx cycle does in the background and that the AI agent does on its own when it writes a memory through the MCP layer (the agent has been instructed in CML conventions and writes in CML directly, so the bytes that arrive at `memory_store()` are already cemelified before Mnemos sees them). That is the kind of thing that happens when you spend months building infrastructure for yourself, mostly for fun, and then sit down to explain it to other people for the first time. WUUUUT.

### Type prefixes (one or two characters)
| Prefix | Meaning |
|---|---|
| `F:` | Fact / config (technical configurations, system state, attributes) |
| `D:` | Decision (with reason) |
| `C:` | Contact (people, organizations, relationships) |
| `L:` | Learning (insight, lesson, pattern observed) |
| `P:` | Preference / pattern (what the user likes, prefers, repeatedly does) |
| `W:` | Warning (safety, gotcha, risk) |

### Relation symbols
| Symbol | Meaning |
|---|---|
| `→` | Leads to, results in, points to (causal) |
| `↔` | Mutual, bidirectional, relates to |
| `←` | Back-reference, originated from |
| `∵` | Because, due to (cause) |
| `∴` | Therefore, so, conclude (logical) |
| `△` | Changed, superseded, evolved |
| `⚠` | Warning, risk, gotcha |
| `@` | At (location, time, host, file) |
| `✓` | Confirmed, verified, working |
| `✗` | Failed, broken, negated |
| `~` | Approximate, uncertain, tentative |
| `∅` | None, empty, missing |
| `…` | Continuation, more, non-exhaustive |
| `;` `>` | Separators inside a line |
| `#42` | Reference to memory ID 42 |

### Quantitative shorthand
| Symbol | Meaning |
|---|---|
| `≥` `≤` | At least, at most |
| `≈` | Approximately |
| `≠` | Not equal, differs from |
| `↑` `↓` | Increased / decreased |
| `×` | Times, by, repeated |

### Less tokens, same detail (for any reader)

CML is **lossless for any reader of the text**. It drops the connective junk tissue that English requires for grammatical sentences to sound correct but that carries no informational value ("currently", "the", "was", "approximately", "based on") but keeps every single piece of information that actually carries meaning. Both a human and an LLM can read the condensed version and reconstruct exactly the same understanding the prose carried.

This is not just a framing claim. Rewriting 20 hand-curated prose memories (209 atomic facts across two styles: 15 fact-dense production-style notes and 5 longer narrative-style entries) into CML and then checking which facts survived, **Claude Opus 4.6 preserves 100% overall**, Sonnet 4.6 preserves 98.1%, Haiku 4.5 preserves 98.1%, and gpt-4o-mini preserves 95.2%. The compression is input-dependent: fact-dense production prose condenses 15–26% (Claude tier and gpt-4o-mini all in that band), and longer narrative prose condenses **41–61%** (gpt-4o at 0.39×, gpt-4o-mini at 0.48×, Sonnet at 0.52×, Haiku at 0.55×, Opus at 0.59×). Full per-subset table across 8 models and per-memory breakdowns in the [CML fidelity benchmark](benchmarks/README.md#4-cml-fidelity-format-level-content-parity--cml_fidelity_benchpy). In practice the agent writes directly in CML, so there is no transformation step at all, the bench is just the cleanest way to prove that the format is expressive enough to hold everything a prose memory would have held.

Compare a real engineering learning, the kind of thing you actually want your AI to remember:

```
Verbose prose (73 tokens, cl100k_base):

"After three failed attempts to get the Stripe webhook integration working
 with my existing FastAPI middleware, I learned that Stripe requires the
 raw request body for signature verification, but my middleware was reading
 the body stream before it reached the webhook handler. The solution was to
 disable body parsing for the /webhooks/stripe route specifically using a
 custom dependency override."
```

```
CML (39 tokens, cl100k_base):

"L: Stripe webhook sig verification needs raw body @FastAPI (3 tries)
 ∵ middleware consumed body stream → fix: disable body parsing
 for /webhooks/stripe via custom dependency override"
```

**About 47% fewer tokens. Zero *actual* information lost.** Every entity, every cause, every fix, every constraint is still there, including the gotcha context (`(3 tries)`) and the FastAPI mechanism name (`override`). What got dropped was English filler ("currently", "the", "approximately", "based on") that carries no semantic weight. The `L:` prefix tells the AI this is a learning. The `∵` tells it the next clause is the cause. The `→` tells it the next clause is the resolution. An LLM reading this gets exactly the same actionable knowledge it would get from the prose version, and you paid roughly half the context budget for it.

How much CML actually shrinks the input depends on how much filler the input had. A technical sentence like the Stripe one above still has narrative glue ("After three failed attempts...", "I learned that...", "The solution was to...") and condenses by about 47%. Longer narrative prose with more connective tissue ("I was sitting at my desk thinking about...", "as I was saying...", "in any case...") condenses harder, up to about **61%** smaller in the bench (gpt-4o on narrative inputs lands at 0.39× at 90.6% preservation; the four top Claude/OpenAI-budget tiers cluster in the 0.48–0.59× range at 92–100% preservation). Fact-dense prose that already reads like a production memory entry (short, already minimal on filler) has less room to shrink: the CML fidelity bench in `benchmarks/` measures 14% to 26% savings against that kind of input, depending on the model doing the cemelification. Across realistic mixed memory inputs, expect savings anywhere in the **14% to ~60%** range depending on how narrative-vs-dense the original text was. The denser your prose, the smaller the win; the more it sounds like English narrative, the bigger the win.

Now multiply that ratio across 1,000 active memories returned in briefings, search results, session priming, and Nyx cycle synthesis runs over months. The compounding token savings are not a rounding error; they are the difference between an AI that has plenty of context budget for actual reasoning and one that spends most of its window re-reading bookkeeping.

### CML and the reranker: complementary, not co-required

CML is lossless for any *reader*, but it is mildly out of distribution for the *bi-encoder* that powers the first stage of semantic retrieval. e5-large was trained on natural-language pairs (Wikipedia, MS MARCO, mC4); it never saw structural prefixes like `F:`, `D:`, `L:` or operators like `→` `∵` `@` during pretraining. The condensed form embeds slightly less cleanly than equivalent prose, and on LongMemEval that translates to a small recall difference if you run pure BM25+vector mode (no reranker) on CML-stored memories. See the [`--mode hybrid --cml`](#--mode-hybrid---cml-hybrid-with-cemelification-no-reranker) benchmark above for the exact numbers, though as it turns out, even the CML-without-reranker mode lands at **98.09% R@5**, essentially tied with the no-CML lite mode (98.30%) and in the same high-90s tier as any other verified no-LLM number in the field.

**The cross-encoder reranker is the cherry on top, not a hard requirement.** CML's structural prefixes act as explicit relation markers the cross-encoder can attend to, which is exactly what reranker architectures are designed to exploit. With the reranker enabled, CML's small bi-encoder penalty disappears and the canonical CML+rerank configuration scores **99.15% R@5**, beating the non-CML rerank configuration's 98.94% on every metric except R@10 where they tie. CML and the reranker are complementary, using both gets you the best number, but you can ship either one alone, or neither, and still land in the same tier as every other public memory system's strongest verified number on this benchmark.

The combined picture:

- **CML alone, no reranker** (98.09% R@5): essentially the same recall as the no-CML lite mode (98.30%), with **15% to 60% fewer tokens** stored and reinjected per memory (the spread depends on how narrative the source prose is, see the CML section below). Net positive whenever context budget matters at all, which is most of the time.
- **CML + reranker (canonical)**: recall is *higher* than the non-CML reranker baseline at every published k except R@10 (tie), AND you still get the 15% to 60% token savings. The honest cost of running the reranker, compared to skipping it, is about **+50 ms per query** and roughly **+500 MB resident RAM** for the cross-encoder model. The RAM cost is negligible on any modern laptop or server and only matters on Pi-class hardware. The latency cost is barely perceptible to a human and is paid once per search, not per memory. For the recall improvement and the token savings combined, it is the right trade for almost every deployment.

For any active memory system the bottleneck eventually becomes context budget, not raw retrieval recall. Combining CML with the reranker gives you the best of both: upper-90s retrieval recall and a fraction of the tokens. The full per-mode benchmark numbers live in [`benchmarks/`](benchmarks/).

### Soft convention, hard rewards

CML is the default convention everything in Mnemos assumes, and in practice the system enforces it through machinery rather than through schema validation. The CLI hints in CML. The dedup pipeline matches CML subject prefixes for conflict detection. The Nyx cycle cemelifies any prose that slipped through into CML during weekly consolidation. Briefings, digests, and the topic map all render in CML. The MCP tool description teaches the agent CML conventions so it writes in CML directly.

So if you store a memory in plain prose, Mnemos will accept it and search it just fine, but over time the system pulls it toward CML form unless you actively prevent that. **You can opt out**: don't run the Nyx cycle's LLM phases, customize the agent's system prompt to write in plain prose, and you have a prose-only deployment. But by default the corpus is or is becoming CML, and the rest of the system is built around that assumption.

> **Single-flag opt-out (v10.0.1+)**: the `MNEMOS_CML_MODE` environment variable flips every CML-related surface that Mnemos controls, in one place. Set `MNEMOS_CML_MODE=off` (default is `on`) and: the MCP `memory_store` tool description drops its CML guidance and tells the agent to write clear natural prose instead, the `consolidation_lock` field descriptions on `memory_store` and `memory_update` swap from "prevent cemelification" to "prevent merging", the Nyx cycle merge and synthesis phases switch to prose-output prompts that still preserve every atomic fact, the dedup CML-subject branch falls back to FTS + vector signals only, and the Phase 3 Weave bridge insights drop the `L:` prefix. One coordinated flag, prose-only deployment is a first-class configuration, not a degraded mode. The benchmark numbers justify it: the BM25+vector lite mode at **98.30% R@5** is in the same tier as any verified no-LLM number in the field, with no CML at all in the loop.
>
> One surface Mnemos cannot toggle for you: *your AI client's own system prompt* (Claude Code's `CLAUDE.md`, Cursor rules, custom Claude Desktop system prompts, etc.). Those are user-owned and outside the MCP tool interface. See [`docs/agent-instructions.md`](docs/agent-instructions.md) for copy-paste CLAUDE.md blocks for both CML and prose modes; use the matching block for whichever mode you run, and swap if you flip the flag. Mnemos itself only instructs the AI through the tool descriptions it exposes, which `MNEMOS_CML_MODE` covers.

There are real incentives to follow the convention rather than fight it:

- CML is the format Mnemos expects you to write in. Examples, hints, briefings, and the Nyx cycle's merged super-memories all use it. The CLI still stores non-CML input rather than rejecting it, but prints a one-line nudge so you notice the convention exists and start using it
- **The Nyx cycle cemelifies memories automatically.** When Phase 2 merges duplicates or Phase 3/5 generate relationships and insights, the LLM is explicitly instructed to output only CML. So even if you store a memory in free-form prose, the next consolidation pass rewrites it (and any duplicates) into a single compact CML super-memory. Over time, the entire active store converges toward CML without you doing anything
- The Nyx cycle uses the same conventions when it merges memories, so consolidated super-memories stay compact
- Reranker results are tighter on CML inputs because the cross-encoder sees consistent structural cues

If you (or, more realistically, the AI agent on your behalf) write memories in CML, the system rewards you with smaller context bills, better dedup precision, and more reliable consolidation. If you do not, everything still works; you just leave the savings on the table until the Nyx cycle catches up.

## Forgetting: nothing gets deleted automatically

A common misconception about decay-based memory systems is that old memories disappear. In Mnemos they don't. **Decay is a ranking modifier, not a deletion mechanism.**

When a memory ages, its temporal-decay boost shrinks, so it ranks lower in search results. The memory itself stays in the database. There is a floor at 10% of the maximum boost, so even a five-year-old fact still gets a small ranking nudge if it matches a query exactly. This mirrors how human memory actually works: you have not forgotten your old school address, you just no longer associate it with "where I live".

The decay curve uses two half-lives:

| Layer | Half-life | day 30 | day 90 | day 365 |
|---|---|---|---|---|
| Episodic (events, conversations) | 46 days | 0.96 | 0.39 | 0.15 (floor) |
| Semantic (distilled knowledge) | 180 days | 1.34 | 1.06 | 0.38 |

The only ways memories ever leave the active set:

- `mnemos delete <id>` archives a memory (still in DB, hidden from default search)
- `mnemos delete <id> --hard` permanently removes the row and its vector
- The optional Nyx cycle Phase 2 merges near-duplicates into a consolidated super-memory; the originals get archived but remain queryable via `nyx_insights`
- `mnemos doctor` flags memories untouched for 90+ days as "stale" but takes no action

What protects a memory from sliding down in rank:

- **Access**: every search hit and `memory_get` resets the decay clock
- **`evergreen` tag**: skips decay entirely, for permanent facts (birthdays, blood type, addresses)
- **`semantic` layer**: 4x slower decay than `episodic` events
- **`consolidation_lock=1`**: protects from Nyx cycle merging
- **High importance (≥9)**: never demoted by Phase 6 bookkeeping

In short: Mnemos does not forget. It just stops bringing up things you have not used in a while, and lets you decide what to throw away.

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

This is exactly how I run Mnemos in production: a single SQLite store for curated memories plus several Qdrant collections for bulk indexed mail, project documents, personal notes, ebooks, and work files (8 collections, ~500K vectors total). All searched through the same Mnemos retrieval pipeline (BM25 + vector + RRF + cross-encoder), all from the same 4 MCP tools, all CPU-only.

**Mnemos is not just a memory system. It is a unified, local, CPU-only semantic retrieval layer for everything you own that has text in it.**

## Storage backends

Mnemos has a pluggable storage layer. Summary:

- **SQLite backend (default, atomic)**, class `SQLiteStore`. Single SQLite file, row + FTS + vector in one transaction, up to ~10K memories on SSD. What most users should run.
- **Qdrant scaling layer (split storage)**, class `QdrantStore`. SQLite remains authoritative for content, FTS, metadata, and relations; Qdrant mirrors the vector index for HNSW at scale. Use when you want HNSW performance at tens of thousands of memories or already run Qdrant for bulk external content (mail, docs, ebooks). Cross-system atomicity is lost by design; SQLite is the source of truth and Qdrant can be rebuilt from it.
- **Postgres backend (planned, atomic)**, class `PostgresStore`. Postgres + pgvector in the same database, same single-transaction guarantee as SQLite, adds ACID multi-tenancy and MVCC. Stub today; contributions welcome.

The retrieval pipeline is identical across backends; only the vector index location changes.

Full details, the atomicity tradeoff, real-world scale observations, SQLite-only vs split-storage as a deployment decision, and a code example, in [`docs/ARCHITECTURE.md#storage-backends`](docs/ARCHITECTURE.md#storage-backends).

## Memory usage

Mnemos is CPU-only but loads real ONNX models into RAM. Summary:

- **With reranker (default):** ~1.5-1.7 GB resident. Canonical configuration, benchmark numbers reported on this.
- **Without reranker (`MNEMOS_ENABLE_RERANK=0`):** ~1-1.2 GB resident. Runs on a 2 GB+ Raspberry Pi. Trades ~0.5 pp of R@5 for ~500 MB less RAM.
- **Disk:** ~800 MB total for both ONNX models (e5-large embedder + Jina cross-encoder), downloaded once on first use and cached under `~/.cache/fastembed`.
- **Sub-1 GB hardware:** not designed for it out of the box. Swap to a smaller embedder (e.g. `BAAI/bge-small-en-v1.5`) via `MNEMOS_EMBED_MODEL` if you truly need 512 MB total; retrieval quality drops but Mnemos still runs.

Full per-component breakdown (Python + Mnemos, SQLite + sqlite-vec, embedder, reranker) and sub-1 GB configuration notes in [`docs/ARCHITECTURE.md#ram-and-disk-footprint`](docs/ARCHITECTURE.md#ram-and-disk-footprint).

## Multi-user / Auth

Mnemos core is single-tenant by design. All operations are scoped to a `namespace` (default `"default"`); multi-user deployments use different namespaces. **There is no auth in the storage layer**, authentication is a transport-layer concern (the MCP server you run, the HTTP API you build on top, whatever auth your gateway already handles). Same separation Postgres uses.

Most users land on single-tenant ("my AI assistant has a memory of me") and the namespace + filesystem permissions story is enough. Multi-tenant deployments wire up OAuth / JWT / API keys in front of the MCP server and map authenticated identities to distinct namespaces, the hooks are there, the policy is the deployment's call.

Full discussion (the "a system should only be as complicated as it needs to be" framing and the design consequences) in [`docs/ARCHITECTURE.md#multi-user-auth-and-why-the-storage-engine-stays-auth-free`](docs/ARCHITECTURE.md#multi-user-auth-and-why-the-storage-engine-stays-auth-free).

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

Restart your AI client. The 4 tools (`memory_store`, `memory_search`, `memory_get`, `memory_update`) will be available.

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

### CLI usage

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
mnemos consolidate --nyx --execute # include synthesis (Phase 5)

# MCP server (typically invoked by your AI client, not directly)
mnemos serve
```

### Optional: Nyx cycle consolidation

Mnemos can run a 6-phase weekly **Nyx cycle** that merges related memories,
detects contradictions, and synthesizes cross-domain insights. The LLM-driven
phases (Dedup, Contradict, Synthesize) need an API endpoint. **Phase 6
(Bookkeeping) always runs without an LLM** and handles vector cleanup, decay,
and stale link pruning purely in SQL.

**Pick a smart model, not a fast one.** The Nyx cycle is intentionally a
background job; it runs weekly (or whenever you trigger it) and the entire
purpose is to take its time *thinking* about your memories. The quality of the
consolidation, dedup, and cross-domain synthesis is bounded by how good the LLM
is at reasoning, not by how fast it answers. A slow but capable model
(Qwen 2.5 32B locally, Claude Sonnet/Opus, GPT-4o, DeepSeek R1) will produce
much better results than a fast lightweight model. Latency does not matter
here. Quality does. If your Nyx cycle takes 20 minutes to run instead of 2,
that is fine, because it is running while you are asleep or doing something
else.

**You do not need a GPU even for the LLM.** Modern 32B-class models like
Qwen 2.5 32B, Llama 3.1 70B (quantized), or DeepSeek R1 distill variants can
run entirely on CPU through Ollama or llama.cpp, as long as you have enough
RAM (usually 32-64 GB depending on quantization). On a typical desktop or
small server without a graphics card, expect generation speeds of around
1-5 tokens per second depending on model size and quantization, instead
of the 50-100 tok/s you would get on a GPU.
For Mnemos that is completely fine. The Nyx cycle does not care if Phase 2
takes 30 seconds or 5 minutes per merge cluster, as long as the merge is
correct. **Mnemos remains GPU-free end to end**, including consolidation,
if you choose a local model, or if you choose not to run the consolidation
cycles at all (store and retrieve work without any LLM, as covered above).

To enable LLM-powered phases, set environment variables for any OpenAI-compatible
endpoint:

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

Without these set, `mnemos consolidate` will skip LLM phases and only run
bookkeeping. The core memory features (store/search/get/update) **never require
an LLM**. Mnemos itself only uses local CPU models for embeddings and reranking.

### Session hooks (Claude Code)

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

## Benchmark

> **Multiple metric classes, clearly separated.** Memory systems get benchmarked on fundamentally different things, and conflating them produces misleading comparisons. Mnemos publishes across all of them.
>
> - **Retrieval recall (R@K, NDCG@K) on LongMemEval.** Does the system find the right memory? Deterministic, no LLM in the measurement path. 470 non-abstention questions. Mnemos's headline numbers live here.
> - **Retrieval recall (R@K) on LoCoMo.** Same metric class, different dataset. 1,540 evaluable QA pairs across 10 long conversations (19-32 sessions each), with 446 adversarial questions excluded by methodology. Top-K capped at 10 to avoid the bypass where K ≥ session count makes retrieval trivial. Numbers in [`benchmarks/README.md`](benchmarks/README.md#3-locomo-retrieval-recall-locomo_benchpy).
> - **End-to-end QA accuracy (LLM-judge).** Given the retrieved context, does the answerer produce the correct answer? Bundled metric, measures retrieval + answerer reading comprehension + judge fairness. LongMemEval supports this on all 500 questions including abstention. This is what Mastra, Emergence, Mem0, Supermemory, and most of the field publish. Mnemos also publishes these numbers below.
> - **Consolidation quality (fact preservation).** How well does the offline Nyx cycle merge related memories without losing specifics? Our own internal metric, reported against 30 historical merge events from the Mnemos production database.
> - **CML fidelity.** Does CML preserve the same atomic-fact content as equivalent prose? 20-memory hand-curated corpus, 209 facts, 8 models tested. See [`benchmarks/README.md`](benchmarks/README.md#5-cml-fidelity-format-level-content-parity-cml_fidelity_benchpy).
>
> **Do not compare R@K numbers against QA accuracy numbers.** They measure different things. 98.94% R@5 is not directly comparable to 86% QA accuracy, one measures retrieval, the other measures a full pipeline including an answering LLM. Any table or chart that puts these numbers in the same column is metric-mixing.

Run the LongMemEval benchmark yourself:

```bash
cd benchmarks
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -O longmemeval_s.json

# Lightweight: BM25 + vector only, no reranker, no CML preprocessing
python longmemeval_bench.py --mode hybrid

# Cleanest unmanipulated benchmark: BM25 + vector + cross-encoder reranker, no preprocessing
python longmemeval_bench.py --mode hybrid+rerank

# Hybrid with cemelification at storage time, no reranker
python longmemeval_bench.py --mode hybrid --cml

# Canonical Mnemos: BM25 + vector + reranker + CML, the configuration the system was designed to run in
python longmemeval_bench.py --mode hybrid+rerank --cml
```

Mnemos was benchmarked across **four configurations** against the LongMemEval 470-question session-granularity set. Every run uses the exact same code, the same models, the same RRF weights, the same reranker thresholds, and **zero parameter tuning against the benchmark**. No held-out validation set was peeked at. No thresholds were swept. No models were swapped to chase a number. **All four are first-run results. The benchmark just ran.**

> **A note on Mnemos's natural habitat.** In production, Mnemos's curated memory store is **always CML**. The agent writes memories in CML directly via `memory_store()`, the Nyx cycle keeps it that way in the background, and by the time anything queries the store, the bytes are already CML. **The two non-CML benchmark modes (`--mode hybrid` and `--mode hybrid+rerank`) are therefore Mnemos being benchmarked outside its natural habitat**, running on raw verbatim conversation prose that the production memory store would never actually hold. Those modes correspond instead to how `mnemos ingest` indexes external content (mail, documents, ebooks, notes, prose that was never CML to begin with). They are published for three reasons: (1) the cleanest possible measurement, with no preprocessing of the test data and no LLM call anywhere in the pipeline, which is the most skeptic-proof number we can offer; (2) the fact that **Mnemos matches or exceeds every verified retrieval-recall number in the field even when handicapped by running on prose it was not designed to store as memories**; (3) and the strong non-CML numbers (98.30% / 98.94% R@5) confirm that **Mnemos handles verbose natural-language prose just fine**, which is exactly what the `mnemos ingest` pipeline relies on for indexing external content like mail, documents, ebooks, and notes, content that is and stays in prose form. The CML modes (`--mode hybrid --cml` and `--mode hybrid+rerank --cml`) are the production configuration for the curated memory store, where Mnemos is in its native form.

**One methodology disclosure for the CML modes**: they cemelify the LongMemEval transcripts at storage time before running the benchmark. The benchmark data itself is converted from raw verbatim conversation prose into CML notation via a one-time Claude Haiku 4.5 call per session, cached on disk, then handed to the standard storage pipeline. This is how Mnemos works in production (the Nyx cycle does the same thing in the background to anything that slipped through as prose). It does mean the two `--cml` runs are technically operating on a *transformed* version of the test data, but that transformation matches the production state of every memory in the store. See [How the CML benchmark actually works](#how-the-cml-benchmark-actually-works-the-part-with-the-asterisk) below for the full methodology.

The four result JSON files (with `completed_at` timestamps from the actual runs) live in [`benchmarks/`](benchmarks/) so anyone can verify the numbers below match the source data.

### `--mode hybrid` (BM25 + vector only, no reranker, no CML)

The lightest configuration Mnemos ships. FTS5 BM25 + sqlite-vec + RRF fusion, no cross-encoder, no CML preprocessing of stored data. ~1 GB RAM, ~5 to 30 ms per query, runs on a Raspberry Pi.

> **Important: this is not how Mnemos runs the curated memory store.** The memory store is always CML + rerank (canonical, see below). This mode is relevant for `mnemos ingest`-indexed external content (mail, documents, ebooks, notes), prose that was never CML to begin with, and that you want indexed on a tight RAM budget.

| Question Type | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| knowledge-update | 98.61% | 100.0% | 100.0% | 100.0% | 72 |
| multi-session | 94.21% | 98.35% | 100.0% | 100.0% | 121 |
| single-session-assistant | 94.64% | 96.43% | 96.43% | 96.43% | 56 |
| single-session-preference | 60.00% | 86.67% | 86.67% | 96.67% | 30 |
| single-session-user | 98.44% | 100.0% | 100.0% | 100.0% | 64 |
| temporal-reasoning | 89.76% | 95.28% | 98.43% | 98.43% | 127 |
| **Overall** | **92.13%** | **97.02%** | **98.30%** | **98.94%** | **470** |

NDCG@5 = 0.9612, NDCG@10 = 0.9590. Zero pipeline failures. Source: [`benchmarks/results_hybrid_session.json`](benchmarks/results_hybrid_session.json)

### `--mode hybrid+rerank` (cleanest unmanipulated measurement)

Hybrid retrieval plus the Jina v2 cross-encoder reranker. **No preprocessing of LongMemEval test data, no LLM calls anywhere in the pipeline, no parameter tuning.** The benchmark just runs and the numbers come out. The cross-encoder is a discriminative scorer, not a generative language model: it takes a query and a candidate document and emits a relevance score, locally, on CPU, in ~50 ms, without generating any tokens. **There is no LLM in the search path.** The 98.94% number is achieved without any model "working" to think the answer through. The search just happens. Completed 2026-04-10 16:04.

> **Important: this is not how Mnemos runs the curated memory store either.** The memory store is always CML + rerank (canonical, see below). Like `--mode hybrid` above, this configuration is relevant for `mnemos ingest`-indexed bulk content where the source is prose and the cross-encoder is worth the extra RAM. It is published here primarily as the skeptic-proof number: the cleanest possible measurement of the pipeline on raw LongMemEval data with nothing processed, tuned, or preprocessed.

| Question Type | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| knowledge-update | 98.6% | 100.0% | 100.0% | 100.0% | 72 |
| multi-session | 98.3% | 100.0% | 100.0% | 100.0% | 121 |
| single-session-assistant | 92.9% | 96.4% | 96.4% | 98.2% | 56 |
| single-session-preference | 73.3% | 86.7% | 100.0% | 100.0% | 30 |
| single-session-user | 98.4% | 100.0% | 100.0% | 100.0% | 64 |
| temporal-reasoning | 91.3% | 97.6% | 97.6% | 97.6% | 127 |
| **Overall** | **94.3%** | **98.1%** | **98.94%** | **99.15%** | **470** |

Source: [`benchmarks/results_hybrid+rerank_session.json`](benchmarks/results_hybrid+rerank_session.json)

**98.94% R@5 with no asterisk of any kind.** It is also at or above every other verified clean retrieval-recall number on LongMemEval I was able to confirm from public sources. The closest comparable is MemPalace raw ChromaDB at 96.6% (which, per the MemPalace maintainers, measures ChromaDB's default sentence-transformer embeddings with no MemPalace-specific code path involved); most other systems in the field either publish only LLM-judge QA accuracy (a different metric class that bundles retrieval and generation) or rely on a generative LLM in their highest-scoring configurations.

### `--mode hybrid --cml` (BM25 + vector with cemelification, no reranker)

Same BM25 + vector pipeline as `--mode hybrid` above, but with stored memories converted to CML notation at storage time using the same Nyx cycle conversion path that runs in production.

| Question Type | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| knowledge-update | 98.61% | 100.0% | 100.0% | 100.0% | 72 |
| multi-session | 96.69% | 99.17% | 100.0% | 100.0% | 121 |
| single-session-assistant | 96.43% | 100.0% | 100.0% | 100.0% | 56 |
| single-session-preference | 53.33% | 76.67% | 80.00% | 83.33% | 30 |
| single-session-user | 100.0% | 100.0% | 100.0% | 100.0% | 64 |
| temporal-reasoning | 90.55% | 96.85% | 97.64% | 97.64% | 127 |
| **Overall** | **92.98%** | **97.45%** | **98.09%** | **98.30%** | **470** |

NDCG@5 = 0.9615, NDCG@10 = 0.9573. Zero pipeline failures. Source: [`benchmarks/results_hybrid_session_cml.json`](benchmarks/results_hybrid_session_cml.json)

R@5 sits essentially tied with the no-CML BM25+vector mode (98.09% vs 98.30%), but **R@1 on `single-session-preference` collapses to 53.33%**, the lowest single-cell number in any of the four configurations Mnemos publishes. The reason: the bi-encoder (e5-large) was trained on natural-language pairs and is mildly out of distribution on CML's condensed structural form. Without a reranker to recover the gap, the bi-encoder ranks the right session at slot 1 only about half the time on the precision-heavy preference questions. **The cross-encoder reranker more than rescues this in the canonical configuration below**, preference R@1 jumps from 53.33% to 80.00% the moment the reranker is added back, because CML's structural prefixes that confused the bi-encoder become explicit relation markers the cross-encoder can attend to. This is the worst way to run Mnemos: cemelifying memories at storage time but then running retrieval without the cross-encoder, which is the exact partner that turns CML's structural prefixes from a bi-encoder confusion into a cross-encoder advantage. This row is published anyway as an honest ablation, because it is the only configuration of the four I would not personally use, and pretending otherwise would be the kind of selective reporting this README is built to avoid.

### `--mode hybrid+rerank --cml` (canonical Mnemos, the configuration the system was designed to run in)

This is what real Mnemos looks like in actual production use: stored memories already in CML form (the agent writes them that way at storage time, no Nyx cycle required), retrieval through the standard BM25 + vector + cross-encoder rerank pipeline, no LLM in the search path. **The configuration is what I actually run in production.** For the benchmark, the LongMemEval transcripts are cemelified once at storage time to match this production state; the [methodology section below](#how-the-cml-benchmark-actually-works-the-part-with-the-asterisk) explains exactly how and why.

| Question Type | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| knowledge-update | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 72 |
| multi-session | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 121 |
| single-session-user | 96.88% | **100.0%** | **100.0%** | **100.0%** | 64 |
| single-session-assistant | 92.86% | 98.21% | **100.0%** | **100.0%** | 56 |
| single-session-preference | 80.00% | 93.33% | 96.67% | 96.67% | 30 |
| temporal-reasoning | 93.70% | 97.64% | 97.64% | 97.64% | 127 |
| **Overall** | **95.74%** | **98.72%** | **99.15%** | **99.15%** | **470** |

NDCG@5 = 0.9781, NDCG@10 = 0.9774. Zero pipeline failures. Source: [`benchmarks/results_hybrid+rerank_session_cml.json`](benchmarks/results_hybrid+rerank_session_cml.json)

> **The number to take away from this section is 99.15%.** That is what Mnemos does in actual production use, in the canonical configuration the system was designed to run in. The 98.94% above is what it does in a deliberately raw measurement mode that Mnemos was not intended to use as its memory store; it is published as the most skeptic-proof number on offer, for readers who want zero asterisks. If you actually run Mnemos the way I run it, **99.15% R@5** is the number you will see.

> **Four full categories at 100% R@5. 313 questions in a row where Mnemos made zero retrieval mistakes.** Across `knowledge-update` (72/72), `multi-session` (121/121), `single-session-user` (64/64), and `single-session-assistant` (56/56), Mnemos's canonical configuration retrieves the right session every single time. The four total misses across all 470 non-abstention questions (one in single-session-preference, three in temporal-reasoning) give an **0.85% miss rate**. To my knowledge, no other memory system in any published comparison reports a result like this on LongMemEval, and the data is not being post-processed at evaluation time or scored through an LLM API. The score comes out of pure retrieval recall on the canonical configuration, every query handled locally on CPU, no generative model in the loop.

#### How the CML benchmark actually works (the part with the asterisk)

Storage requires an LLM call no matter what memory system you use, because something has to decide what is worth remembering and how to phrase it. That reasoning step happens in the agent (Claude Code, Cursor, or whichever AI is talking to the user) and is the same fixed cost for any AI memory system in the field. Mnemos does not double that cost. When the agent calls `memory_store()` with already-formed CML content, Mnemos receives the bytes, runs FTS+vector dedup, embeds via FastEmbed, and stores. **No second LLM round-trip inside Mnemos to process, format, or extract anything from what the agent already wrote.**

In the benchmark, LongMemEval provides raw verbatim conversation transcripts rather than pre-formatted CML notes. **Mnemos's own benchmark harness** (in `benchmarks/cml_convert.py`) calls Claude Haiku 4.5 once per session message via DO Gradient API to convert those transcripts into CML, caches the result on disk by SHA256 hash, then hands the CML output to the standard Mnemos storage pipeline. The LongMemEval benchmark itself does not call any LLM; Mnemos is the one doing it, as a one-time data-shape conversion that happens outside the retrieval pipeline being measured. This conversion is necessary to be able to measure how Mnemos actually works in CML mode, the benchmark dataset ships in raw conversation prose, so without converting it to CML at storage time there is no way to evaluate the canonical production configuration of the system. **This LLM call is a benchmark artifact**, present because LongMemEval data needs conversion in order to benchmark it the way Mnemos actually stores memories in real use. Real users never pay this cost: in production, the agent writing the memory does that formatting at no additional cost as part of its existing reasoning step, and any prose that slips through gets cemelified gradually by the Nyx cycle in the background. **This is also exactly why the other three benchmark configurations exist**: the non-CML runs measure Mnemos against the raw LongMemEval data with no preprocessing at all, the lite mode measures it without the reranker, and the CML-without-reranker mode measures the worst-case combination. Together the four runs let you see exactly how each design choice contributes to the final number, decide for yourself which configuration best matches what you care about, or check whether I am full of shit. In my own production setup the Nyx cycle uses **Qwen3-32B served via DigitalOcean Gradient** (their OpenAI-compatible inference endpoint) to cemelify anything that slipped through as prose. The public package has **no default model** and you set whichever one fits via `MNEMOS_LLM_MODEL` (and any OpenAI-compatible endpoint via `MNEMOS_LLM_API_URL`). If you do not configure an LLM at all, the Nyx cycle simply skips the LLM-driven phases and the rest of Mnemos still works. Most modern LLMs handle CML conversion equivalently because the task is constrained and mechanical: the grammar does the heavy lifting and the model is just a translator.

If you disagree with whether LLM-based CML preprocessing of the benchmark data counts as fair, **use the `--mode hybrid+rerank` table above instead**. That number (98.94% R@5) is the cleanest possible measurement and is at or above every verified retrieval-recall number I could compare it against. The CML number is a more accurate picture of how Mnemos actually performs in steady-state production use; the verbose number is a more conservative measurement of pure retrieval quality. Both are honest, both come from first-run no-tuning runs, both are in the same tier as every verified competitor retrieval-recall number at every comparable k, and both result files are in `benchmarks/` with timestamps you can verify against the file modification times.

### Configuration summary

| Configuration | R@5 | RAM | Latency | When to use |
|---|---|---|---|---|
| `hybrid` (BM25 + vector, lite) | 98.30% | ~1-1.2 GB | ~5-30 ms | Pi-class / 2 GB+ Pi 4 / embedded; or for `mnemos ingest`-indexed external prose content |
| `hybrid+rerank` (clean benchmark) | **98.94%** | ~1.5-1.7 GB | ~50-80 ms | The skeptic-proof published number; not what you would deploy by default, but what you cite when someone wants zero asterisks |
| `hybrid --cml` (CML, no rerank) | 98.09% | ~1-1.2 GB | ~5-30 ms | Honest ablation; bi-encoder is mildly OOD on CML, rerank rescues it below. R@5 looks fine but R@1 on `single-session-preference` collapses to **53.33%**, the worst cell in any configuration. **Don't use this unless you hate the idea of a local CPU reranker but love CML.** |
| **`hybrid+rerank --cml` (canonical, recommended default)** | **99.15%** | ~1.5-1.7 GB | ~50-80 ms | **The default for any deployment with RAM to spare.** What Mnemos was designed to run in. Steady-state of an actively-used Mnemos. |

All four are first-run, no-tuning measurements with their result JSON files committed to `benchmarks/`. **Mnemos lets you pick the hit rate you want against the resources you can spare**, on the same retrieval pipeline, by toggling two configuration flags. You are not locked in any direction. But if you ask me, you should run it in CML with the reranker on; that is the configuration the system was designed around and the one I use every day.

### LoCoMo retrieval recall (second public benchmark)

LongMemEval is the primary benchmark Mnemos was designed against, but publishing a single benchmark is brittle: any system can over-fit to one dataset. [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024) is a separate long-conversation memory benchmark: 10 conversations of 19-32 sessions each, 1,986 QA pairs across 5 categories. Median session length 2,652 chars (much longer than LongMemEval's typical sessions, which changes what configurations are practical).

The same Mnemos retrieval pipeline runs against LoCoMo with no parameter tuning. Methodology guardrails: top-K capped at 10 (the smallest LoCoMo conversation has 19 sessions; K ≥ 19 trivially returns every session and stops measuring retrieval), 446 adversarial-by-design questions in category 5 excluded from R@K (mathematically undefined; same convention LongMemEval applies to abstention).

| Mode | R@1 | R@3 | R@5 | R@10 | NDCG@5 |
|---|---|---|---|---|---|
| `hybrid` (BM25 + vector + RRF) | 57.9% | 77.0% | **84.7%** | **94.0%** | 77.1% |
| `hybrid --cml` | 49.0% | 69.9% | 79.4% | 91.0% | 70.1% |
| `hybrid+rerank --cml` | 60.9% | 79.7% | **86.1%** | 91.9% | 79.7% |

`hybrid+rerank` without CML is not in the table because it underperforms even plain `hybrid` on this benchmark: LoCoMo sessions are too long for the cross-encoder's attention window when scored at full length, and aggressive truncation loses the evidence the cross-encoder was supposed to read. CML preprocessing (sessions compressed to ~500 chars of dense facts) is the prerequisite for effective reranking on long-session data. Full methodology, per-category breakdown, and the conv-26 truncation-experiment data point are in [`benchmarks/README.md`](benchmarks/README.md#3-locomo-retrieval-recall-locomo_benchpy).

The headline takeaway: **the same Mnemos pipeline that scores 99.15% on LongMemEval scores 86.1% on LoCoMo** without any retraining or per-dataset tuning. The gap is the dataset, not the system: LoCoMo conversations are denser and longer than LongMemEval's, with a larger unanswerable subset (446 adversarial questions in LoCoMo vs 30 abstention questions in LongMemEval, both excluded from R@K by the same convention since R@K is mathematically undefined when there is no gold session to retrieve). Both benchmarks evaluate retrieval recall the same way; both numbers are first-run, no-LLM-in-retrieval, fully air-gappable.

### End-to-end QA accuracy (retrieval + answerer + judge)

LongMemEval's primary metric in the original paper is end-to-end QA accuracy: retrieval returns top-K sessions, an answerer LLM produces an answer from those sessions, and a judge LLM scores the answer against the reference. The 500 questions include 30 abstention cases where the correct behavior is refusal. Published numbers in this metric class include [Mastra at 94.87%](https://mastra.ai/research/observational-memory) (binary QA accuracy on LongMemEval-S with gpt-5-mini as answerer, per their research page), Emergence at ~95%, and several other systems. Note: [Mem0](https://mem0.ai/research) has not published a LongMemEval number at the time of writing. Their 66.9% figure is overall accuracy on LoCoMo, a different benchmark entirely.

Mnemos's R@5 = 98.94% means retrieval is near-perfect. End-to-end QA accuracy is necessarily ≤ R@5, you can only answer correctly what retrieval found. The gap between R@5 and QA accuracy tells you how much the answerer LLM fumbles correct retrievals.

Two configurations tested, 500 questions each, all results JSON files in [`benchmarks/`](benchmarks/):

**Mnemos hybrid retrieval + Claude Sonnet answerer + Claude Opus judge (raw session context, K=5):**

| Question Type | Accuracy | N |
|---|---|---|
| abstention (correct refusal) | 96.7% | 30 |
| single-session-assistant | 94.6% | 56 |
| single-session-user | 93.8% | 64 |
| knowledge-update | 81.9% | 72 |
| temporal-reasoning | 72.4% | 127 |
| multi-session | 66.1% | 121 |
| single-session-preference | 46.7% | 30 |
| **Overall** | **77.4%** | **500** |

**Mnemos hybrid retrieval + gpt-4o answerer + gpt-4o judge + per-session CML compression of retrieved context:**

| | Value |
|---|---|
| **Overall QA accuracy** | **58.6%** (293/500) |

The CML-compressed-context configuration performs worse on multi-session and preference question types because per-session CML compression *independently* destroys cross-session aggregation signals. When the same fact is mentioned across three sessions, raw text preserves three independent mentions (answerable by counting), while independent per-session CML compresses each session into a terse blob with no cross-session cardinality reference. This is a known architectural limitation of the query-time compression approach tested here, the production path (Nyx cycle consolidation before retrieval, documented below) addresses this by merging related memories into a single consolidated fact before retrieval runs, preserving aggregate signals. That configuration was not re-run against LongMemEval for this release because LongMemEval's haystack is constructed from topically-disjoint conversations and does not exercise the accumulation patterns consolidation is designed for. See the Consolidation Quality section below for the orthogonal measurement.

**Methodology disclosure for the QA numbers:**

- **Answerer:** Claude Sonnet 4.6 (raw-context run) / gpt-4o (CML-context run), temperature 0, K=5 retrieved sessions.
- **Judge:** Claude Opus 4.6 (raw run) / gpt-4o (CML run), temperature 0, per-question-type guidance prompts.
- **Same-family judge bias:** Sonnet-answerer-to-Opus-judge and gpt-4o-to-gpt-4o are both same-family evaluator pairs. Cross-family judge (e.g. gpt-4o judging Sonnet output) rerun is planned as a follow-up ablation; the numbers above may shift by up to ±3 points under a cross-family judge.
- **What the Sonnet+raw 77.4% number means:** Mnemos's retrieval paired with Claude Sonnet as answerer, using raw retrieved session text as answer context. This is what a Claude Code user pairing Mnemos with Sonnet would actually see. Not directly comparable to Mastra's 94.87% because their run uses gpt-5-mini as answerer on LongMemEval-S while ours uses Claude Sonnet on the same dataset. Answerer choice is a large factor in QA accuracy (gpt-5-mini scored 100% on our consolidation-quality bench with its reasoning tokens; different model families exhibit very different reading-comprehension behavior on this kind of retrieved-context QA). An apples-to-apples re-run against a standardized answerer is planned.
- **No tuning against the benchmark.** Same retrieval code, same hybrid mode, same K, no prompts tuned to LongMemEval's phrasing, no threshold sweeps. First runs.
- **Abstention is included.** The 30 abstention questions test whether the answerer correctly refuses when retrieved context lacks the answer. R@K evaluation necessarily skips them (no gold session to retrieve); QA evaluation must include them. Our 96.7% abstention accuracy demonstrates the answerer correctly refuses under retrieved-distractor pressure.

**Honest take on the QA numbers:**

Mnemos's 77.4% QA accuracy on the Sonnet+raw configuration sits in the upper-middle of the public field, above LongMemEval's paper baselines (60-65%), below Mastra's reported 94.87% (with gpt-5-mini as answerer), and below Emergence's reported ~95%. Under a cross-family judge rerun the number may move ±3 points. The retrieval layer (98.94% R@5) is the stronger story; the end-to-end number is bottlenecked by the answerer's reading comprehension on multi-session aggregation (66% on that type) and by judge strictness on fuzzy preference matches (47% on preferences). Neither is a retrieval problem. Mnemos's memory layer already found the right sessions in both cases. These are the limitations of treating LongMemEval's adversarial haystack of topically-disjoint conversations as a proxy for real accumulation-based memory use. See the Consolidation Quality section for where the accumulation-based improvements actually manifest.

### Consolidation quality (fact preservation in the Nyx cycle)

LongMemEval is constructed from one-shot conversation sessions and does not exercise the accumulate-merge-revisit cycle that consolidation is designed for. To measure consolidation quality on real data, Mnemos runs a separate benchmark against its own production memory database: 30 historical merge events (clusters of 2-8 memories that the Nyx cycle previously consolidated), exact ground truth via the `merged-into-<id>` tag lineage preserved on archived originals, unique-fact preservation measured by embedding-based semantic deduplication of extracted source facts followed by LLM-judged preservation check.

| Configuration | Unique-fact preservation | Cost/run | Notes |
|---|---|---|---|
| **Older MERGE_SYSTEM (Opus)** | **75.3%** | ~$6 | Historical merges already in the database from before the prompt rewrite. Fact loss concentrated on 6-8 memory clusters. |
| **Current MERGE_SYSTEM + Opus** | **89.0%** | ~$6 | Rewritten prompt ("co-location not compression" with self-audit rule) + hierarchical pairwise merge for clusters >2. |
| Current + Claude Sonnet | 90.4–94.5% | ~$1 | Two runs, ±4-point variance. 5× cheaper than Opus. |
| Current + Claude Haiku | 87.7% | ~$0.10 | Anthropic-family cheap tier. |
| Current + gpt-4o-mini | 91.8–97.3% | ~$0.05 | Two runs, ±5-point variance. 20× cheaper than Sonnet, comparable preservation. **Recommended default.** |
| Current + gpt-5-mini | **100%** (73/73 unique facts) | ~$0.50–1 | Slowest run (reasoning tokens), highest preservation of anything tested. |
| Current + Llama 3.3-70B | 84.9% | ~$0.20 | Open-weights, **self-hostable** via Ollama / vLLM. Reasonable for users without API access. |
| Current + Qwen3-32B | 61.6% | $0.07 | Over-compresses; drops too many specifics. Not recommended. |

**What this shows:** the preservation improvement under the current prompt is a genuine architectural fix, the older prompt's "output ≈ size of ONE input memory" instruction was dropping ~25% of unique facts on historical merges. The rewrite (in [`mnemos/consolidation/prompts.py`](mnemos/consolidation/prompts.py)) removes that size cap and adds a self-audit rule. Hierarchical binary merging for clusters >2 (in [`mnemos/consolidation/phases.py`](mnemos/consolidation/phases.py)) is the accompanying algorithmic fix.

**Metric caveat:** the preservation metric uses an LLM judge for the match step (Claude Sonnet), which introduces some noise, run-to-run variance is ~±4-5 points at 30 clusters. Absolute numbers should be read with that buffer; cross-configuration deltas remain meaningful because the same judge is applied to all runs.

**Nothing is ever lost even at 75%.** Mnemos's two-tier recall architecture means archived originals are always available; when retrieval asks for a specific fact that did not survive into the consolidated memory, `auto_widen` falls back to the archived originals. Higher merge preservation reduces tier-2 widening frequency (faster queries, cleaner contexts) but does not change what is recoverable. The 80-90% target is about fast-path availability, not information loss.

## How Mnemos compares to other systems

| System | R@1 | R@3 | R@5 | R@10 | Approach | Local? |
|---|---|---|---|---|---|---|
| **Mnemos canonical (BM25+vector+rerank+CML)** | **95.74%** | **98.72%** | **99.15%** | **99.15%** | FTS5 + sqlite-vec + RRF + cross-encoder, cemelification | ✅ Yes |
| **Mnemos clean (BM25+vector+rerank)** | **94.3%** | **98.1%** | **98.94%** | **99.15%** | Same pipeline, no CML preprocessing of test data | ✅ Yes |
| **Mnemos lite (BM25+vector only)** | 92.13% | 97.02% | 98.30% | 98.94% | FTS5 + sqlite-vec + RRF, no reranker | ✅ Yes |
| MemPalace raw ChromaDB (no LLM) | n/a | n/a | 96.6% | n/a | Vector-only over verbatim text | ✅ Yes |
| MemPalace + Haiku rerank (LLM in loop) | n/a | n/a | 100% | n/a | Vector + Claude Haiku LLM rerank, **per-query Anthropic API call**, ~500 outbound calls per benchmark run | ⚠️ Local + remote LLM API |
| LongMemEval paper Stella+fact baseline | n/a | n/a | ~73% | n/a | Vector + value/fact expansion | ✅ Yes |
| Dense retrieval (paper baseline) | n/a | n/a | ~90% | n/a | Vector-only | varies |
| BM25 baseline (paper) | n/a | n/a | ~85% | n/a | Lexical-only | ✅ Yes |
| Mastra, Supermemory, Mem0, OMEGA, Emergence, Zep, Letta, Cognee, EverMemOS, TiMem, etc. | not published | not published | not published | not published | LLM-judge QA accuracy only, no retrieval recall | varies |

A few things worth noting about this table:

1. **Mnemos appears to be the only memory system in the field publishing a complete R@1 / R@3 / R@5 / R@10 sweep on LongMemEval.** MemPalace publishes only R@5 plus per-type R@10, and almost everyone else publishes only LLM-judge QA accuracy (a different metric that bundles retrieval and generation). At every k where competitors publish a comparable retrieval-recall number, Mnemos is on top.

2. The competitor architectural notes are based on each system's own published documentation and benchmark methodology, not on source code audits. I have not read every competitor's source. The Mnemos source is in this repository and you can verify everything in the Mnemos rows directly.

3. **The 98.94% number (clean BM25+vector+rerank) requires no asterisk of any kind.** No CML preprocessing of test data, no LLM in the search path, no tuning. It is the most defensible single number in the table and it is still higher than every published competitor.

4. **The 99.15% number (canonical BM25+vector+rerank+CML) is the configuration the system was designed to run in**, and the one I use in production. For the benchmark to work in this mode it does involve a one-time LLM call per session to convert raw LongMemEval transcripts into CML at storage time (real users do not pay this cost; their agent writes CML directly). See [How the CML benchmark actually works](#how-the-cml-benchmark-actually-works-the-part-with-the-asterisk) above for the full disclosure.

5. **Mnemos performs zero LLM calls of any kind in the production search path.** The cross-encoder reranker is a discriminative scorer (Jina v2, ~570 MB ONNX, CPU-only), not a generative language model. Among systems in the upper-90s **retrieval recall** tier, Mnemos appears to be the only one I could verify that never calls a generative model in its retrieval pipeline.

6. **On end-to-end QA accuracy (the metric Mastra/Emergence/Mem0/etc. publish), Mnemos's paired-with-Sonnet pipeline scores 77.4%**, competitive with the middle of the published field, below Emergence's reported 95%, above LongMemEval paper baselines. The QA accuracy gap vs retrieval reflects answerer-side reading comprehension on multi-session aggregation, not a retrieval failure. See the [End-to-end QA accuracy section](#end-to-end-qa-accuracy-retrieval--answerer--judge) for full numbers and methodology.

7. **Abstention handling is largely undisclosed in the field.** LongMemEval's 500 questions include 30 abstention cases (correct behavior = refusal when the retrieved context doesn't contain the answer). Retrieval metrics (R@K) skip these since there's no gold session to retrieve. QA-accuracy numbers should include them but most systems don't disclose whether their reported accuracy is on all 500 or just the 470 non-abstention questions. Mnemos publishes both, the [QA section](#end-to-end-qa-accuracy-retrieval--answerer--judge) reports 77.4% on all 500 with Sonnet+raw, and **96.7% accuracy specifically on the 30 abstention questions** (the answerer correctly refuses when retrieved context lacks the answer, rather than fabricating). A system confidently answering questions whose correct answer is "I don't know" is a quiet quality problem the field has not been forced to benchmark.

8. **On why QA numbers do not get re-run every time something changes**: the end-to-end QA benchmark costs real money. A single full 500-question QA run against Sonnet + Opus-judge is ~$20-30 in API fees, and running a sweep of answerer/judge pairs multiplies that. I cannot afford to re-run these numbers on every code change. The retrieval benchmarks are free in API terms (they run locally on CPU with zero outbound calls) but they are not free in wall-clock time, a full run is **10 to 15 hours** on the reference hardware, so even the retrieval numbers are not re-run on every commit; they get revalidated when something material changes in the pipeline, and the QA numbers are revalidated less often still. If you reproduce either benchmark against your own account, your results will be directionally the same but may differ by a few percentage points due to model non-determinism (QA side) or to hardware / embedder-build differences (retrieval side); the [`benchmarks/`](benchmarks/) directory has the runners and the result JSONs so you can verify whichever dimension you care about.

The combined picture: **Mnemos is the only memory system in this comparison I could verify with (a) upper-90s retrieval recall on LongMemEval, (b) end-to-end QA numbers published alongside retrieval numbers rather than in place of them, (c) fully local CPU-only operation with no generative LLM in the search path, and (d) a stable four-tool MCP surface under 1.5 K tokens of system-prompt overhead.** Other systems may match it on one or two of those axes; based on what they publish, I have not found one that matches on all four.

## Supporting Mnemos

Mnemos is MIT-licensed and developed by one person on a home server, in spare time, as a labor of curiosity. There is no company behind it, no VC pressure, and no plans to add a paid tier. If Mnemos saves you time, helps your AI assistant remember you better, or just makes you smile, you can chip in:

- ⭐ **Star the repo**. Costs nothing, helps people find it.
- ☕ **Buy me a coffee**: [buymeacoffee.com/dracaglitch](https://buymeacoffee.com/dracaglitch)
- 💖 **GitHub Sponsors**: [github.com/sponsors/draca-glitch](https://github.com/sponsors/draca-glitch)
- 🐛 **Open issues** for bugs, feature requests, or weird edge cases you hit in the wild
- 🛠️ **Send PRs**: extra extractors for the ingest module, a real Postgres backend, NLI-based contradiction detection, anything

Sponsorship goes directly toward keeping the development server running and toward more time spent improving Mnemos instead of doing my day job. Nothing you sponsor unlocks paid features. Mnemos stays one piece of software, fully open source, with everything in the public repo.

I published this because I suddenly realized other people might actually want it too, after reading articles about the AI personal memory space, not just me. As I mentioned at the start of this README, I originally had no plan to release Mnemos at all; it was just my own infrastructure for my own use. But here we are, because at some point I thought *why not*. Everything in this repository is open, honest, and straightforward. If you do not want to use it, do not. If you do, I hope it serves you at least a fraction of the way it has served me.

## License

MIT: see [LICENSE](LICENSE).

## Credits

Built on top of:
- [FastEmbed](https://github.com/qdrant/fastembed): multilingual e5-large ONNX embeddings
- [sqlite-vec](https://github.com/asg017/sqlite-vec): vector search in SQLite
- [Jina Reranker v2](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual): cross-encoder
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval): benchmark dataset
