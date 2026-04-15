# Features

> Reference for what's in the box. Architecture-deep details live in [ARCHITECTURE.md](ARCHITECTURE.md); design rationale in [philosophy.md](philosophy.md).

## Hybrid retrieval pipeline

1. **FTS5 BM25** with AND-default + OR-fallback for multi-term queries
2. **Vector similarity** via FastEmbed e5-large (1024-dim, ONNX, CPU-native, ~7ms/embed)
3. **RRF fusion** (Reciprocal Rank Fusion, k=60) merges both rankings
4. **Cross-encoder reranking** (Jina v2 multilingual) for final precision. **The cross-encoder is a local ONNX discriminative scorer, not a remote LLM**, it runs on CPU in ~50 ms per query, no API call, no network round trip, no API key required. **Enabled by default** in the public package, but the lite mode without it (`--mode hybrid`) already lands at **98.30% R@5** on LongMemEval, in the same tier as any other verified no-LLM number in the field. The cross-encoder pushes the canonical configuration up to **99.15% R@5**, it is the cherry on top, not a hard requirement to be competitive. Disable with `MNEMOS_ENABLE_RERANK=0` or `Mnemos(enable_rerank=False)` if you want the smallest possible RAM footprint; you trade about 0.85 percentage points of R@5 for ~500 MB less resident memory.

## Why these specific models (and how to swap them)

The embedding model and the reranker are both **swappable**. Mnemos talks to FastEmbed-compatible models for embeddings and to any cross-encoder loadable through FastEmbed for reranking, so you can plug in whatever you prefer. Every benchmark number on this repo is reported on the default e5-large embedder + Jina v2 reranker pair, and changing either side has not been measured.

### Embedder

**`intfloat/multilingual-e5-large`** was picked for very specific reasons:

- **Truly multilingual**, not English-with-token-mapping. The memory store itself defaults to English CML by convention (the agent writes in English; the Nyx cycle keeps it that way), but the same retrieval pipeline also indexes external data sources via `mnemos ingest` (ebooks, notes, documents, PDFs) which contain Swedish, English, and other languages depending on the source. The semantic match needs to work *across* languages too: an English query like "breakfast" should match a Swedish ebook chapter about "frukost". e5-large handles 100+ languages in the same vector space, which is rare for high-quality models.
- **1024 dimensions**, a sweet spot I tested my way into. High enough to capture nuance, low enough that brute-force search in SQLite stays fast and storage stays reasonable.
- **Available as ONNX**, runs on CPU at ~7 ms per embedding without a GPU runtime or PyTorch dependency.
- **Apache 2.0 license**, no surprises.
- **Battle-tested** on MTEB and many production retrieval systems.

Swap with `MNEMOS_EMBED_MODEL`. Smaller models (BGE-small, all-MiniLM) are faster but less precise. Larger models (BGE-large, GTE-large) are slower but may score higher on specific benchmarks. The rest of the pipeline does not care.

If you want to run a 4096-dim embedder (e5-mistral-7b, NV-Embed-v2, SFR-Embedding-2_R) or even an 8192-dim Matryoshka model (Stella-en-1.5B-v5 at full resolution), set `MNEMOS_EMBED_MODEL` to whatever ONNX-exposed model you want and `MNEMOS_EMBED_DIMS` to match. Storage scales up (a 4096-dim float32 vector is 16 KB per memory vs 4 KB for the 1024-dim default), per-query embed latency moves from ~7 ms to seconds (those 7B-parameter generative-style embedders are slow on CPU), and you need more RAM. The retrieval recall lift over e5-large is real but small (typically 1-2 points on hard benchmarks). For most users the trade is not worth it. For the ones for whom it is, the swap is one env var.

### Reranker

**`jinaai/jina-reranker-v2-base-multilingual`** was picked for matching reasons:

- **Multilingual** (same language coverage as the embedder, important because the reranker and embedder see the same text).
- **ONNX-friendly**, CPU-only, no GPU runtime required.
- **Small**: sub-300 MB on disk, ~400-500 MB resident.
- **Permissive license** (Apache 2.0).
- **Tuned for query-document relevance scoring**, not generic similarity, which is the signal reranking actually needs.

Swap with `MNEMOS_RERANKER_MODEL`, or disable entirely with `MNEMOS_ENABLE_RERANK=0` if you want the smallest possible RAM footprint (trades ~0.5 pp of R@5 for ~500 MB less resident).

Could a stronger reranker like Jina Reranker v3 push the numbers higher? Possibly, but it is currently not available as an ONNX export, would need a different runtime than FastEmbed, is a 0.6B-parameter model (about 2× the size of v2), and its realistic deployment story involves GPU acceleration in most cases. That would break the "runs on a Raspberry Pi" claim and the entire footprint argument this README is built on. Mnemos is intentionally **CPU-only and non-GPU** as a hard design constraint. If a community-maintained Jina v3 ONNX export ever lands and benchmarks well on CPU at acceptable latency, swapping it in would be the natural next step.

## Temporal modeling

- **Exponential decay** with separate half-lives:
  - Episodic memories: ~46 days (events, conversations)
  - Semantic memories: ~180 days (distilled knowledge)
- **Validity windows**: facts can have `valid_from` / `valid_until` for time-bound truths
- **`valid_only` filter**: exclude expired facts at query time

## Hierarchical organization

- Top-level projects are free-form strings. A starter set is provided (`dev`, `finance`, `food`, `health`, `home`, `personal`, `relationships`, `server`, `travel`, `work`, `writing`) but you can add or remove categories to match how you think about your memory. The storage layer does not enforce the list.
- Optional sub-categories per project (e.g., `dev/myapp`, `finance/crypto`)
- Free-form, no schema migration needed

## Nyx cycle consolidation (the optional LLM-driven part of Mnemos)

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

Per-phase model routing, env vars, and recommended models in [usage.md](usage.md#per-phase-model-routing). The narrative on what the Nyx cycle actually learns about you over time lives in [philosophy.md](philosophy.md#adaptive-learning-how-mnemos-gets-to-know-you).

## Contradiction detection (real-time, on store)

Every `memory_store()` call checks the new memory against existing facts on the same topic via vector similarity + same-project filter + cross-encoder rerank (no LLM in the loop). Conflicts surface in the response immediately and persist as `memory_links` with `relation_type='contradicts'` for the next Nyx cycle's Phase 4 batch pass to consider. Full mechanism in [`ARCHITECTURE.md#real-time-contradiction-detection-on-store`](ARCHITECTURE.md#real-time-contradiction-detection-on-store).

## Auto-widen

If a project-filtered search returns fewer than 3 results, Mnemos re-runs the same pipeline without the project filter. Cheap safety net for the common "I stored it under a different project than I'm searching from" case. Details in [`ARCHITECTURE.md#auto-widen`](ARCHITECTURE.md#auto-widen).

## 3-way deduplication

Before storing, Mnemos pools candidates from three independent signals (FTS5 keyword overlap, CML-subject match, vector similarity) and runs them through the Jina cross-encoder for a final confidence score. Three signals catch different failure modes; any one of them missing a duplicate is fine. Details in [`ARCHITECTURE.md#3-way-deduplication-on-store`](ARCHITECTURE.md#3-way-deduplication-on-store).

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
