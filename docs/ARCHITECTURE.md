# Mnemos Architecture

## Design principles

1. **A memory system should be a memory.** It stores data and retrieves it. That is the entire job. No LLM in the retrieval or storage pipeline. The agent thinks; Mnemos remembers.
2. **CML is a language, not a compression algorithm.** Condensed Memory Language is a token-minimal notation the agent writes directly. It is not compressed or encoded; it is just denser English with structural prefixes and operators. Both humans and LLMs read it without decoding. The Nyx cycle cemelifies anything that slipped through as prose.
3. **Retrieval is configurable, not fixed.** Four modes, two flags: BM25 + vector only (lite), BM25 + vector + reranker (clean), BM25 + vector + CML (not recommended), BM25 + vector + reranker + CML (canonical). The reranker is enabled by default but not required for upper-90s numbers; even the lite mode lands in the same tier as any verified no-LLM number in the field.
4. **Hybrid retrieval beats single-method.** Combining lexical (BM25) and semantic (vector) signals via RRF, then reranking with a cross-encoder, consistently outscores either alone.
5. **Curated > Verbatim.** Memories are distilled facts, decisions, and learnings, not raw chat transcripts. Higher signal-to-noise.
6. **CPU-only is a feature, not a constraint.** ONNX models run on every laptop, NAS, Pi 4+, and budget VPS. No GPU monopoly.
7. **Storage is pluggable, retrieval is not.** The pipeline (FTS5 + vec + RRF + rerank) is the same across backends. Only persistence varies.
8. **No auth in the engine.** Mnemos is a memory engine. Authentication is the responsibility of the transport layer (MCP server, HTTP API, etc.).
9. **Forgetting is a feature.** Exponential temporal decay matches how human memory works. Old, unaccessed memories naturally fade.
10. **A system should only be as complicated as it needs to be.** Four tools, one database file, no LLM in the pipeline, no GPU. Add complexity when the workload demands it, not before.

## Layered architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Client (any MCP)                       │
│  Claude Code  │  Cursor  │  ChatGPT  │  Gemini  │  custom   │
└────────────────────┬────────────────────────────────────────┘
                     │ JSON-RPC 2.0 / stdio
┌────────────────────▼────────────────────────────────────────┐
│                  mnemos.mcp_server                           │
│      4 memory tools: store, search, get, update              │
│      + 1 schema tool: list_tags                              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                     mnemos.core.Mnemos                       │
│  • 3-way dedup on store                                      │
│  • BM25 + vector search (FTS + vec + RRF + rerank)           │
│  • Auto-widen on thin results                                │
│  • Contradiction detection                                   │
│  • Dynamic importance bumping                                │
└────────┬───────────────────┬──────────────────────┬─────────┘
         │                   │                       │
┌────────▼──────┐  ┌─────────▼────────┐  ┌──────────▼──────┐
│ mnemos.embed  │  │  mnemos.rerank   │  │  mnemos.query   │
│  (FastEmbed   │  │  (Jina cross-    │  │  (FTS5 query    │
│   e5-large)   │  │   encoder v2)    │  │   builder)      │
└───────────────┘  └──────────────────┘  └─────────────────┘
                                                  │
┌─────────────────────────────────────────────────▼───────────┐
│                  mnemos.storage.MnemosStore                  │
│                       (abstract base class)                  │
└───┬────────────────────┬───────────────────────┬────────────┘
    │                    │                       │
┌───▼────────┐  ┌────────▼────────┐  ┌──────────▼──────────┐
│ SQLiteStore│  │   QdrantStore   │  │   PostgresStore     │
│ (default)  │  │   (production)  │  │  (stub, PR open)   │
│            │  │                 │  │                      │
│ FTS5 + vec │  │ FTS5 + Qdrant   │  │ tsvector + pgvector │
└────────────┘  └─────────────────┘  └─────────────────────┘
```

## Data model

### Memory
The fundamental unit. Has:
- **Identity**: `id`, `namespace`, `created_at`, `updated_at`
- **Content**: `content` (the actual memory text), `tags`
- **Classification**: `project` (e.g., dev, finance), optional `subcategory` (e.g., crypto)
- **Type**: `fact`, `decision`, `learning`, `preference`, or `todo`
- **Layer**: `episodic` (events, fast decay) or `semantic` (knowledge, slow decay)
- **Importance**: 1-10, dynamically bumped on access
- **Validity**: `valid_from`, `valid_until` (optional time-bound truths)
- **State**: `status` (active/archived), `verified`, `consolidation_lock`
- **Telemetry**: `access_count`, `last_accessed`, `last_confirmed`

### Memory link
A typed relationship between two memories: `related`, `contradicts`, `supports`, etc. Stored with a strength (0-1).

### Nyx insight
A consolidation event recording which source memories were merged into a result. Stored in the `nyx_insights` table. Includes `consolidation_type` (`aggregation` vs `supersession`) so tier-2 recall can decide whether expansion is safe.

## The retrieval pipeline

```
Query "what does the user hold in their portfolio?"
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: Query understanding    │
│  • Stop word removal             │
│  • AND-default with OR-fallback  │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────┬─────────────────────┐
│   FTS5 BM25      │   Vector search     │
│   (lexical)      │   (semantic)        │
│                  │                     │
│   ranks 1-50     │   ranks 1-50        │
└────────┬─────────┴──────────┬──────────┘
         │                     │
         └─────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  RRF fusion (k=60)   │
        │  score = Σ 1/(k+rank)│
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Cross-encoder       │
        │  rerank top 20       │
        │  (Jina v2)           │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Auto-widen          │
        │  (if <3 in project,  │
        │   search all)        │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Attach links        │
        │  (related, contradict│
        └──────────┬───────────┘
                   │
                   ▼
              [results]
```

### Auto-widen

A project-filtered search that returns fewer than three results is widened: the project filter is dropped and the same pipeline runs across all projects. The mechanism exists because users store facts in whichever project felt right at the time, and when they later search from a different context, the project they filtered by may not be where the memory ended up. Rather than returning "no results" and making the caller reformulate, Mnemos broadens the search once and lets the ranking fall out of the unfiltered run. The threshold (3) is low enough that auto-widen only fires on genuinely thin results; a healthy project-scoped search with 5+ hits stays scoped.

### 3-way deduplication on store

Every `memory_store()` call runs a pre-insert dedup pass before writing:

1. **FTS5 keyword overlap** against existing memories in the same project
2. **CML-format subject matching**, if the new memory's CML prefix + subject (`F:server→db-prod-01`) exactly matches an existing memory's subject, they are almost certainly the same topic
3. **Vector cosine similarity** against the embedding index

Candidates from all three signals are pooled and passed through the Jina cross-encoder for a final relevance score. If the top score exceeds the dedup threshold, the new memory is merged into (or flagged as duplicate of) the existing one. The three signals catch different failure modes: keyword overlap finds verbatim repeats, subject matching finds same-topic different-phrasing, and vector similarity finds semantic paraphrases. A single one of them missing a duplicate is normal; all three missing it is rare.

### Real-time contradiction detection on store

Separately from dedup, every `memory_store()` call also checks whether the new memory conflicts with existing facts on the same topic. The mechanism is non-LLM:

1. Vector similarity gate (L2 distance below `CONTRADICTION_VEC_THRESHOLD`)
2. Same-project filter (cross-project topical overlap is almost always complementary, not contradictory, this is the dominant false-positive killer)
3. Jina cross-encoder rerank for topical-match confidence

A conflict surfaces as a warning in the response to the caller plus a `memory_links` row with `relation_type='contradicts'` that the next Nyx cycle's Phase 4 batch pass will consider. This is the synchronous counterpart to Phase 4 Contradict: the on-store check catches an incoming memory conflicting with what is already in the store the moment it arrives; the batch pass catches slow-burn evolution that only becomes visible once enough related memories accumulate. Both write into the same `memory_links` infrastructure.

## The Nyx cycle

Modeled on brain sleep stages. Runs weekly (or on demand) to consolidate memories. Six phases, only phases 2-5 require an LLM (`MNEMOS_LLM_API_URL` + `MNEMOS_LLM_MODEL`). If no LLM is configured, phases 2-5 are skipped automatically and Nyx still runs as SQL-only housekeeping.

### Phase 1. Triage

Pure SQL. Reads `consolidation_log` to find the last run timestamp, then counts memories with `created_at > last_run_at` among the active set. Decides surge mode when the new-memory count exceeds `SURGE_THRESHOLD`, which relaxes the per-phase LLM-call budget on the theory that a burst of new memories is worth extra consolidation effort. Returns the list of new-memory IDs for downstream phase use.

### Phase 2. Dedup

Two-tier clustering and merge. Protection filters apply first, memories with `importance >= 9`, with the `evergreen` tag, or with `consolidation_lock = 1` are skipped entirely and not considered for merging.

**Clustering.** Two similarity thresholds, both running against the same embedding matrix:

- `TIGHT_THRESHOLD = 0.88` (tier 1A), near-duplicate detection
- `TOPIC_THRESHOLD = 0.75` (tier 1B), same-topic merge

Clusters are capped at `MAX_CLUSTER_SIZE = 8` to keep merge prompts tractable.

**Merge strategy.** For clusters of size 2, one LLM call with the `MERGE_SYSTEM` prompt. For clusters of size 3 or more, **hierarchical pairwise merging**: each step merges two inputs at a time into one output, which then participates as an input at the next level. The LLM never sees more than two memories at once, so the natural per-step "this output should be roughly the size of its inputs" intuition holds even for deep clusters. A flat N-way merge of 8 memories forces the model into aggressive fact-dropping to hit a single-memory target size; the hierarchical variant removes that pressure and preserves specifics. The prompt itself is written in a co-location-not-compression style with an explicit self-audit rule ("before emitting, walk each input memory and confirm every distinct fact appears somewhere in the output").

**Storage side.** The new merged memory is stored with the `consolidated` tag and a `consolidation_type` of `aggregation`. Every source memory is flipped to `status='archived'` and its `tags` are appended with `merged-into-<new_id>`. Both the source's embedding and the new memory's embedding are updated in `embed_vec` in the same transaction. Nothing is deleted, the sources stay queryable via archive-aware retrieval paths (see tier-2 recall).

### Phase 3. Weave

Scans pairs of memories from different projects for cross-category connections. A candidate pair must have cosine similarity above a weave-specific threshold (lower than the tight/topic thresholds, because cross-domain connections are usually weaker in embedding space than intra-domain duplicates). The LLM is asked whether the pair genuinely reflects a thematic or causal connection; if yes, a `memory_links` row is created with `relation_type='related'` and a strength score. Output is pruned to the top-K per memory to prevent link explosion.

### Phase 4. Contradict

Scans same-project pairs that are topically similar (cosine above a contradict-specific threshold) and classifies their relationship. The LLM outputs one of four labels:

- `COMPATIBLE`, same topic, different aspects; no change
- `SUPERSEDED`, newer memory updates the same fact with a new value; older gets a `valid_until` timestamp set to the newer memory's `created_at`, and a `memory_links` row is written with `relation_type='supersedes'` and `consolidation_type='supersession'`
- `EVOLVED`, a belief or preference that genuinely shifted over time; both versions stay queryable, the older gets a `valid_until` but also a `relation_type='evolves'` link so the evolution chain is preserved for tier-2 recall and for user queries like "when did I stop believing X?"
- `CONTRADICTS`, two memories state incompatible things and it is not clear which is current; flagged with `relation_type='contradicts'` but neither is marked superseded. Leaves the disambiguation to either the user or the next consolidation cycle with more context.

This is the **batch** contradiction pass. The real-time on-store contradiction check (see the [Contradiction detection](../README.md#contradiction-detection-real-time-on-store) section of the README) runs a lighter non-LLM version, vector similarity plus same-project filter plus cross-encoder rerank, every time `memory_store()` is called, so obvious conflicts get flagged immediately in the response rather than having to wait a week.

### Phase 5. Synthesize

Feeds clusters of semantically-related memories into the LLM and asks for novel observations: recurring themes, cross-domain patterns, tensions, preferences evolving in parallel across domains. The output is stored as new memories with `type='learning'`, `layer='semantic'`, and a `nyx_insights` row linking back to the source memories that informed the insight. These synthesized memories participate in future searches like any other fact and are themselves eligible for further consolidation in later Nyx cycles, so the system's self-generated observations compound over time rather than being one-shot reports.

### Phase 6. Bookkeeping

Pure SQL, always runs regardless of LLM configuration.

- **Temporal decay**: adjust `importance` downward for memories with recent `access_count` below a threshold, following exponential-decay half-lives (`EPISODIC_HALFLIFE_DAYS` and `SEMANTIC_HALFLIFE_DAYS` in `constants.py`). Nothing is deleted, decayed memories just rank lower in hybrid search until access revives them.
- **Orphan cleanup**: remove rows in `embed_vec` and `embed_meta` whose `source_id` no longer exists in `memories` (archived memories keep their embeddings; only deleted ones get cleaned).
- **Stale link pruning**: drop `memory_links` rows where either endpoint has been archived for longer than the `STALE_LINK_DAYS` threshold, so the link graph reflects the current active memory surface without dragging old archived chains along.

### LLM configuration

Phases 2-5 call through the `llm_utils` abstraction. `MNEMOS_LLM_MODEL` is the global default; per-phase env var overrides are first-class:

| Env var | Phase | Falls back to |
|---|---|---|
| `MNEMOS_LLM_MODEL_MERGE` | Phase 2 Dedup merge | `MNEMOS_LLM_MODEL` |
| `MNEMOS_LLM_MODEL_WEAVE` | Phase 3 cross-category links | `MNEMOS_LLM_MODEL` |
| `MNEMOS_LLM_MODEL_CONTRADICT` | Phase 4 temporal evolution | `MNEMOS_LLM_MODEL` |
| `MNEMOS_LLM_MODEL_SYNTHESIZE` | Phase 5 creative insight generation | `MNEMOS_LLM_MODEL` |

Parallel `MNEMOS_LLM_API_URL_<PHASE>` variables exist for the rare case of mixing providers across phases (e.g. local Ollama for cheap phases and a remote API for Synthesize). Each also falls back to the global `MNEMOS_LLM_API_URL`.

Every LLM call through the Nyx cycle is **stateless inference**, the LLM receives only the cluster or pair it needs to reason about; nothing persists between calls; the memory state lives in the database, not in the model. That property lets you route different phases to different providers without any coordination between them.

### Per-phase model choice: what was tested

All numbers below were measured on the reference production memory store. Sample sizes are small (6-30), so recommendations are directional. Larger-scale validation would tighten the intervals.

| Phase | Benchmark | Sample | Result |
|---|---|---|---|
| 2 Dedup merge | Consolidation-quality bench (fact preservation), see [`../benchmarks/README.md`](../benchmarks/README.md) | 30 historical merge events | **gpt-5-mini 100%**, gpt-4o-mini 91.8-97.3%, Sonnet 90.4-94.5%, Opus 89%, Haiku 87.7%, Llama 3.3-70B 84.9%, Qwen3-32B 61.6% |
| 3 Weave | Agreement with Opus on LINK_TYPE classification | 6 cross-category pairs | Sonnet 4/6 agree, gpt-4o-mini 4/6 agree. Weave classification has inherent variance. Opus disagrees with itself at higher sample sizes too. |
| 4 Contradict | Agreement with Opus on 4-label classification | 6 same-category pairs | **Sonnet 6/6 agree** with Opus, gpt-4o-mini 4/6 agree. Phase 4 writes `valid_until` and supersession markers into memory state, so classification errors have downstream effect. |
| 1 Triage priority | Agreement with Opus on HIGH/LOW priority | 10 memories | Sonnet 9/10 agree, gpt-4o-mini 8/10 agree. Topic label differs by style across all three models but priority rank is stable. |
| 5 Synthesize | Not yet benchmarked |, | Creative-generation task. Left on Opus-class pending a benchmark that can meaningfully judge "did the insight capture something true about the user." |

**Recommended per-phase routing (the default pattern the author runs):**

| Phase | Recommended model | Why |
|---|---|---|
| Phase 2 Merge | `gpt-4o-mini` | Best cost/quality ratio. Run the consolidation bench if you want to confirm on your own corpus. |
| Phase 3 Weave | `gpt-4o-mini` | Classification variance is inherent; using a more expensive model does not reduce it. Cheap model suffices. |
| Phase 4 Contradict | Sonnet-class | Classification precision matters because Phase 4 mutates memory state (`valid_until`, supersession links). Sonnet tracks Opus 6/6 on the tested sample. |
| Phase 5 Synthesize | Opus-class | Low call volume, high quality requirement, creative generation. Cost is dominated by the other phases. |
| Phase 1 Triage + Phase 6 Bookkeeping | N/A | Pure SQL, no LLM. |

Per-phase tests were run with `/root/scripts/nyx-phase-quality-check.py` and the consolidation bench. Both are in `benchmarks/` and result JSONs are reproducible.

### Tier-2 recall

Merging is never silent deletion. The archived originals of any merged memory stay in the database with a `merged-into-<id>` tag on their own row and a `memory_links` chain pointing at the merged result. The retrieval pipeline's `auto_widen` behavior uses this chain: when a query's top-K results include a consolidated memory and the caller asks for episodic detail (or when retrieval is thin and needs to surface more context), Mnemos walks the `merged-into-*` tags backward from the merged memory to its constituents and returns them alongside. The merged memory is the fast path; the sources are the precision path.

## Storage backends

Mnemos has a pluggable storage layer. The backend choice is about scale and atomicity requirements, not about functionality, the retrieval pipeline is identical across backends.

Two categories of backend:

**Atomic backends** hold text, FTS, vectors, and relations in one system and commit them in a single transaction. Every `memory_store()` call is all-or-nothing.

**Scaling layers** keep SQLite (or another atomic store) as the source of truth and mirror the vector index into a dedicated vector DB for HNSW performance at scale. Cross-system atomicity is lost by design; the authoritative store can rebuild the vector index if it ever drifts.

| Backend | Category | When to use | Atomic? | Status |
|---|---|---|---|---|
| **SQLiteStore** (default) | Atomic | Personal use, up to ~10K memories on SSD, ~5K on HDD. Single file you can `cp` to back up. | ✅ Yes, single SQLite database, single transaction boundary | ✅ Production |
| **QdrantStore** | Scaling layer | Need HNSW vector search at 25K-to-millions of memories. Already running Qdrant for other indexing (mail, docs, notes). | ⚠️ No by design, SQLite stays authoritative and Qdrant mirrors the vector index | ✅ Reference impl |
| **PostgresStore** | Atomic | Multi-tenant production, ACID, MVCC concurrency, server-hosted rather than embedded | ✅ Yes, single PostgreSQL database (Postgres + pgvector in one transaction) | 🚧 Stub, contributions welcome |

### The atomicity tradeoff

The default SQLite backend is a single SQLite file with a single transaction boundary, so every write either lands fully or rolls back fully. The Qdrant scaling layer splits storage across two systems (SQLite for content, FTS, metadata, and links; Qdrant for vectors) and there is no two-phase commit between them, so a network blip or a crash mid-write can leave the SQLite row inserted but the corresponding Qdrant vector missing (or vice versa). Mnemos retries on failure and re-emits embeddings for any row whose vector is missing, so the system self-heals on the next embed-sync pass, but the *atomic* property of SQLite-only mode is genuinely lost when you split the stores. Workloads that require strict transactional consistency at write time should stay on the SQLite backend (or the Postgres backend once it lands). Workloads that need vector search at scale and can tolerate a short "vector not yet synced" window can go split.

### Scale in practice

Most users running canonical Mnemos only for memories (CML + reranker + Nyx cycle) will never hit the SQLite ceiling. CML keeps each individual memory small. The reranker makes a condensed corpus retrievable at full quality. The Nyx cycle continuously merges duplicates and similar memories into compact super-memories. The active memory count grows much slower than the raw number of things ever stored. The Qdrant scaling layer is most useful when a deployment also wants to index bulk external data alongside memories (mail archives, document libraries, ebooks, source code, hundreds of thousands of chunks) without switching the memory stack.

Author's own deployment as a reference point: after two months of daily use plus a bulk import of a ChatGPT history, the active memory set sits at 327 memories with 833 archived but still queryable via tier-2 recall. That is 1,160 total memories ever stored, consolidated down to roughly 28% of that as the active working set across 33 Nyx cycle runs. For a user storing a handful of facts per day, reaching 5,000 active memories takes years even before consolidation. "Needing to switch to a real database" is a problem most users will never have.

### SQLite-only vs split-storage as a deployment decision

For the pure memory workload (store facts, recall facts, let Nyx do its thing), single-file SQLite is strictly simpler: SQLite is part of Python's standard library, nothing to install, run, secure, back up, or keep alive. No daemon, no port, no auth, no DBA. The memory store is one `.db` file in the filesystem.

Split storage exists for deployments that want to throw bulk external content at the same retrieval pipeline. The author's production runs the SQLite backend for the curated memory store plus Qdrant collections for 8 external content types (mail, project documents, personal notes, ebooks, work files, ~500K vectors). Qdrant holds only the vectorized index of that bulk content, not the content itself. The retrieval pipeline is identical regardless of backend; only the vector index location changes.

```python
from mnemos import Mnemos
from mnemos.storage import SQLiteStore, get_qdrant_store

# Default: SQLite, single file
m = Mnemos()

# Production: Qdrant for vectors, SQLite for metadata + FTS
m = Mnemos(store=get_qdrant_store(
    qdrant_url="http://localhost:6333",
    collection="my_memories",
))
```

## Performance characteristics

### SQLite backend (default), `SQLiteStore`

| Memories | Vector search (NVMe) | Vector search (HDD warm) |
|---|---|---|
| 1K   |   ~5ms |  ~5ms |
| 5K   |  ~25ms |  ~25ms |
| 10K  |  ~45ms |  ~50ms |
| 25K  |  ~75ms | ~100ms |
| 50K  | ~265ms | ~400ms (cache pressure) |
| 100K | ~475ms | unusable without large page cache |

Recommended ceiling: ~10K memories on SSD, ~5K on HDD.

### Qdrant scaling layer, `QdrantStore`

HNSW indexing. Sub-50ms vector search at million-scale. Use this when you have:
- More than 10K memories
- Already running Qdrant for other indexing (mail, docs, notes)
- Multi-process write concurrency requirements

### Cold start

First query in a session loads:
- FastEmbed e5-large ONNX (~500MB, ~3-5s on CPU)
- Jina cross-encoder ONNX (~250MB, ~2-3s on CPU)

Both are cached in memory after first load. Subsequent queries are fast (<200ms total).

### RAM and disk footprint

Mnemos is CPU-only but it does load real models into RAM. Component breakdown:

| Component | Disk | RAM (loaded) | When |
|---|---|---|---|
| Python + Mnemos package | ~50 MB | ~100-150 MB | Always |
| SQLite + sqlite-vec | trivial | ~10-50 MB (page cache) | Always |
| e5-large embedder (ONNX) | ~500 MB | ~800 MB - 1 GB | Always loaded once on startup |
| Jina cross-encoder reranker (ONNX) | ~280 MB | ~400-500 MB | Only if `enable_rerank=True` |

**With reranker** (default): **~1.5-1.7 GB resident**. First call spools the cross-encoder ONNX (1-2s cold-start), then subsequent reranks are ~50 ms. The MCP server warms it up at boot so the spool cost is paid once at startup rather than on the first user query.

**Without reranker** (opt-out via `MNEMOS_ENABLE_RERANK=0`): **~1-1.2 GB resident**. Fits comfortably on a 2 GB+ Raspberry Pi 4/5 or a small VPS. Trades about half a percentage point of R@5 for ~500 MB less resident memory. All benchmark numbers remain in the same tier as any verified no-LLM recall number in the field.

**Disk**: about 800 MB total for both ONNX models, downloaded automatically from HuggingFace on first use and cached under `~/.cache/fastembed`.

**Sub-1 GB hardware**: Mnemos is not designed for it. The e5-large embedder is ~500 MB in memory by itself and the runtime + SQLite + Python interpreter add the rest, so even the lightest default configuration sits around 1 GB. For 512 MB devices (Pi Zero, micro-VPS, embedded), swap `intfloat/multilingual-e5-large` for a smaller embedder like `BAAI/bge-small-en-v1.5` (English only, 33 MB ONNX, ~150 MB RAM) via `MNEMOS_EMBED_MODEL` and disable the reranker. Retrieval quality drops noticeably (no multilingual, less semantic headroom) but the system still works as a basic memory store.

## Multi-source indexing (production pattern)

The author runs Mnemos in **split-storage mode (SQLite + Qdrant)** in production. Mnemos itself only indexes the curated memory layer, but the same FastEmbed/Jina pipeline is used to index bulk content via Qdrant collections:

```
┌────────────────────┐
│  Mnemos memories   │  ← curated, hand-stored, ~1K-10K items
│  (SQLite + vec)    │
└────────────────────┘
         +
┌────────────────────┐
│  Qdrant collections│  ← bulk indexed, ~500K vectors total
│  • mail            │
│  • documents       │
│  • notes           │
│  • ebooks          │
│  • work-files      │
│  • etc.            │
└────────────────────┘
         │
         ▼
   Both searched via the same
   retrieval pipeline (FTS+vec+RRF+rerank)
```

This is why the storage abstraction matters: you can run Mnemos for memories *and* point its retrieval pipeline at Qdrant collections that already index your other content. Same retrieval quality, no duplication.

## Multi-user, auth, and why the storage engine stays auth-free

Mnemos core is single-tenant by design. All operations are scoped to a `namespace` (default `"default"`); multi-user deployments simply use different namespaces. **There is no auth in the storage layer.** Authentication is a transport-layer concern, add it at the layer that exposes Mnemos to clients:

- **MCP via stdio**: OS-level (whoever runs the process)
- **HTTP API**: JWT, OAuth, API keys, your choice
- **Multi-tenant SaaS**: your gateway, your rules

This is the same separation Postgres uses (no auth in the query engine, all roles externally) and SQLite uses (file system permissions). It keeps the memory engine simple and lets deployments wrap it however their needs dictate.

In practice the single-user case is where most Mnemos deployments sit: "my AI assistant has a memory of me", not "a shared memory service for a team". For that case the namespace + filesystem permissions story is enough. Building a real auth layer when there is only one user is just dead code and additional attack surface. Genuinely multi-tenant workloads (hosted services with multiple paying users, team-shared assistants, internal tools with role-based access) are easily solvable by putting any standard auth layer (OAuth, JWT, API keys, IAM) in front of the MCP server and mapping authenticated identities to distinct namespaces. The hooks are there; the policy is wired up by the deployment. Keeping the storage engine auth-free means users who never need auth do not pay the cost of an unused feature.

**A system should only be as complicated as it needs to be.** That principle runs through every design decision in this project: four MCP tools instead of nineteen, one SQLite file instead of a separate vector database, no auth in the storage layer instead of a half-baked role system, no LLM in the search path instead of a generative reasoning step nobody asked for. Add complexity when the workload demands it, not before.
