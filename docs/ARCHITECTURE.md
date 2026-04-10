# Mnemos Architecture

## Design principles

1. **Curated > Verbatim.** Memories are distilled facts, decisions, and learnings, not raw chat transcripts. Higher signal-to-noise.
2. **Hybrid retrieval beats single-method.** Combining lexical (BM25) and semantic (vector) signals via RRF, then reranking with a cross-encoder, consistently outscores either alone.
3. **CPU-only is a feature, not a constraint.** ONNX models run on every laptop, NAS, Pi 4+, and budget VPS. No GPU monopoly.
4. **Storage is pluggable, retrieval is not.** The pipeline (FTS5 → vec → RRF → rerank) is the same across backends. Only persistence varies.
5. **No auth in the engine.** Mnemos is a memory engine. Authentication is the responsibility of the transport layer (MCP server, HTTP API, etc.).
6. **Forgetting is a feature.** Exponential temporal decay matches how human memory works. Old, unaccessed memories naturally fade.

## Layered architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Client (any MCP)                       │
│  Claude Code  │  Cursor  │  ChatGPT  │  Gemini  │  custom   │
└────────────────────┬────────────────────────────────────────┘
                     │ JSON-RPC 2.0 / stdio
┌────────────────────▼────────────────────────────────────────┐
│                  mnemos.mcp_server                           │
│           (4 tools: store, search, get, update)              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                     mnemos.core.Mnemos                       │
│  • 3-way dedup on store                                      │
│  • Hybrid search (FTS + vec + RRF + rerank)                  │
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
│ (default)  │  │   (production)  │  │   (stub. PR open)  │
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

### Dream insight
A consolidation event recording which source memories were merged into a result. Includes `consolidation_type` (`aggregation` vs `supersession`) so tier-2 recall can decide whether expansion is safe.

## The retrieval pipeline

```
Query "what does the user hold in their portfolio?"
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: Query understanding    │
│  • Stop word removal             │
│  • Swedish stem stripping        │
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

## The dream cycle

Modeled on brain sleep stages. Runs weekly (or on demand) to consolidate memories:

1. **Triage** - detect new memories since last run, decide surge vs normal mode
2. **Dedup** - merge near-duplicates (cosine ≥0.88 tight, ≥0.75 topic). Marks consolidation as `aggregation`.
3. **Weave** - find cross-category relationships, create `memory_links` with `relation_type='related'`
4. **Contradict** - detect temporal evolution, set `valid_until` on superseded versions, mark consolidation as `supersession`
5. **Synthesize** - generate cross-domain insights via local LLM (optional)
6. **Bookkeeping** - apply temporal decay, cleanup orphaned vectors, prune stale links

## Performance characteristics

### SQLiteStore (default)

| Memories | Vector search (NVMe) | Vector search (HDD warm) |
|---|---|---|
| 1K   |   ~5ms |  ~5ms |
| 5K   |  ~25ms |  ~25ms |
| 10K  |  ~45ms |  ~50ms |
| 25K  |  ~75ms | ~100ms |
| 50K  | ~265ms | ~400ms (cache pressure) |
| 100K | ~475ms | unusable without large page cache |

Recommended ceiling: ~10K memories on SSD, ~5K on HDD.

### QdrantStore

HNSW indexing. Sub-50ms vector search at million-scale. Use this when you have:
- More than 10K memories
- Already running Qdrant for other indexing (mail, docs, notes)
- Multi-process write concurrency requirements

### Cold start

First query in a session loads:
- FastEmbed e5-large ONNX (~500MB, ~3-5s on CPU)
- Jina cross-encoder ONNX (~250MB, ~2-3s on CPU)

Both are cached in memory after first load. Subsequent queries are fast (<200ms total).

## Multi-source indexing (production pattern)

The author runs Mnemos in **hybrid mode** in production. Mnemos itself only indexes the curated memory layer, but the same FastEmbed/Jina pipeline is used to index bulk content via Qdrant collections:

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
   hybrid pipeline (FTS+vec+RRF+rerank)
```

This is why the storage abstraction matters: you can run Mnemos for memories *and* point its retrieval pipeline at Qdrant collections that already index your other content. Same retrieval quality, no duplication.

## Why no auth in core?

Authentication is a transport-layer concern. Mnemos itself only knows about `namespace` (a string scoping all operations). Add auth at the layer that exposes Mnemos to clients:

- **MCP via stdio**: OS-level (whoever runs the process)
- **HTTP API**: JWT, OAuth, API keys, your choice
- **Multi-tenant SaaS**: your gateway, your rules

This is the same separation Postgres uses. It keeps the engine simple and lets you wrap it however your deployment needs.
