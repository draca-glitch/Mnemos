# Changelog

All notable changes to Mnemos. Dates are from the original private development
repository, where the system existed under an internal name (`agent-memory`)
before being open-sourced as Mnemos in this repo.

This changelog documents real version history. Mnemos was not built on a
weekend; it grew through nine internal iterations over months of personal use,
each one adding or removing features based on what actually improved retrieval
quality and what only added complexity.

> **A note on the git history in this repository**: I did not use git for
> this project until I created my first GitHub account on April 10, 2026.
> The system has months of evolution behind it. This repository has one
> commit, because that is when I started versioning it. The version
> progression below is real and the dates are accurate, however I wasn't
> very good at writing down what I did as I did it.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/).

---

## [10.0.1] - 2026-04-15 (single-flag CML opt-out + LoCoMo benchmark)

### Added
- **`MNEMOS_CML_MODE` environment variable**, the unified switch that the
  v10.0.0 README described as "Planned" is now shipped. `MNEMOS_CML_MODE=on`
  (default) keeps the original behavior. `MNEMOS_CML_MODE=off` flips all
  CML-related surfaces in one place:
  - The MCP `memory_store` tool description drops the CML format guidance
    and tells the agent to write clear natural prose instead
  - The Nyx cycle's merge prompt swaps in `MERGE_SYSTEM_PROSE`, which keeps
    every unique-fact-preservation rule but instructs prose output, no CML
    prefixes or relation symbols
  - The Nyx cycle's synthesis prompt swaps in `SYNTHESIS_SYSTEM_PROSE`,
    insights are emitted as blank-line-separated prose paragraphs rather
    than `L:`-prefixed CML lines
  - `_parse_insights` reads either format based on the mode
  - Phase 3 Weave's bridge-insight memory content drops the `L:` prefix
  - `Mnemos._unified_dedup`'s CML-subject branch is gated off; dedup
    falls back to FTS + vector signals (both still run)
  - `consolidation_lock` field descriptions on the `memory_store` and
    `memory_update` MCP tools swap from "prevent cemelification" to
    "prevent merging" since cemelification does not happen in prose mode
  Single coordinated flag, not a collection of half-matched overrides.
  README now also calls out the one surface Mnemos cannot toggle: the
  user's own AI-client system prompt (Claude Code `CLAUDE.md`, Cursor
  rules, etc.). If you have added "write in CML" instructions there,
  remove them manually when switching to prose mode.
- Prose variants of the merge and synthesis prompts in
  `mnemos/consolidation/prompts.py` (`MERGE_SYSTEM_PROSE`,
  `SYNTHESIS_SYSTEM_PROSE`) that preserve the same atomic-fact-preservation
  rules as the CML variants with natural-language output format.
- **LoCoMo retrieval-recall benchmark** (`benchmarks/locomo_bench.py`): runs
  the same hybrid pipeline against the [LoCoMo](https://github.com/snap-research/locomo)
  dataset (Maharana et al., ACL 2024). 10 conversations, 19-32 sessions
  each, 1,986 QA pairs across 5 categories. Methodology guardrails baked
  in: top-K capped at 10 (smallest LoCoMo conversation has 19 sessions,
  so K below that means retrieval is doing real work), adversarial-by-
  design questions in category 5 (446 items) excluded from R@K with the
  same convention LongMemEval uses for abstention, per-conversation
  session counts published alongside the result for verification.
  Three modes shipped with results:
  - `hybrid`:               R@5 = 84.7%, R@10 = 94.0%
  - `hybrid --cml`:         R@5 = 79.4%, R@10 = 91.0%
  - `hybrid+rerank --cml`:  R@5 = 86.1%, R@10 = 91.9%
  The `hybrid+rerank` mode (no CML) is documented as not-recommended on
  LoCoMo: median session length is 2,652 chars, p90 4,090 chars, the
  Jina cross-encoder cannot see a whole session at once when scoring
  relevance, and aggressive truncation cuts off the very evidence the
  cross-encoder was supposed to read. CML preprocessing (sessions
  compressed to ~500 chars of dense facts) is the prerequisite for
  effective reranking on long-session benchmarks; the `hybrid+rerank
  --cml` row is the configuration that pairs the cross-encoder with
  text it can see in full. Conv-26 control data point: full-text rerank
  scored 78.0% R@5 vs `hybrid` 86.0% on the same conversation.
- LoCoMo dataset (`benchmarks/locomo10.json`, 2.8MB) committed for
  reproducibility (the dataset is small enough to ship; LongMemEval
  remains downloadable-on-demand because it is much larger).
- LoCoMo section in `benchmarks/README.md` (full methodology + per-
  category breakdown + interpretation) and a brief headline section in
  the main README right after LongMemEval's results.

### Changed
- README "Soft convention, hard rewards" callout updated from "Planned" to
  document the now-implemented switch and its exact per-surface effects.
- Top-of-README benchmark intro updated to list five metric classes
  (LongMemEval R@K + LoCoMo R@K + LongMemEval QA + consolidation
  quality + CML fidelity) instead of three.
- Top-of-README "Benchmarked" feature bullet softened: removed the
  comparative claim ("Matches or exceeds every reproducible retrieval-
  recall number I have been able to verify from other public memory
  systems") and the inline MemPalace re-attribution. The Mnemos numbers
  stand on their own; the comparative framing in the Origin section's
  table is the only place head-to-head numbers appear.

---

## [10.0.0] - 2026-04-15 (first public release)

This version is essentially the packaging and documentation work to make
the private system releasable. All the core features (BM25 + vector retrieval,
CML, Nyx cycle, decay, dedup, contradiction detection) already existed
in v6 through v9. What v10 added was the public-facing scaffolding to
turn a personal server script into a proper open-source Python package
that other people could install and use.

### Added (packaging for public release)
- Reorganized as an installable Python package (`mnemos/`, `mnemos/storage/`,
  `mnemos/consolidation/`) with `pyproject.toml` and CLI entry points
- **Pluggable storage backends**: `MnemosStore` abstract base class with
  two categories. *Atomic* backends hold text + FTS + vectors in one
  transaction: the SQLite backend (default, production, class
  `SQLiteStore`) and the Postgres backend (stub, planned for multi-tenant
  ACID, class `PostgresStore`). *Scaling layer*: the Qdrant backend
  (class `QdrantStore`) keeps SQLite authoritative and mirrors the vector
  index to Qdrant for HNSW performance at 25K-plus memories.
- **`mnemos ingest`** CLI command for indexing external content (notes,
  code, docs) with a pluggable extractor API for custom formats
- **`mnemos doctor`**: health check for schema, FTS sync, embedding
  coverage, and stale memories
- **LongMemEval benchmark runner** under `benchmarks/`, with the
  reproducible results documented in the README
- **OpenAI-compatible LLM client** for the consolidation phases,
  replacing the hardcoded model references in the private version.
  Works with any provider (OpenAI, Ollama, OpenRouter, DigitalOcean
  Gradient, Together.ai, Groq, Fireworks, etc.) with graceful fallback
  when no LLM is configured
- Public-facing README, ARCHITECTURE.md, and CHANGELOG (this file)
- **End-to-end QA accuracy benchmark** against LongMemEval (500 questions including abstention), published alongside existing R@K retrieval numbers. Mnemos is the only memory system in the public landscape publishing both metric classes with clear methodology disclosure.
- **Consolidation-quality benchmark**: fact preservation rate against historical merge events from a production memory store. Measures how well the Nyx cycle merge step preserves specifics across clusters of 2-8 memories.
- **Rewritten `MERGE_SYSTEM` prompt** in `mnemos/consolidation/prompts.py`: co-location-not-compression philosophy with an explicit self-audit rule. Unique-fact preservation improves from 75.3% (older prompt) to 89.0% (new prompt) on the 30-cluster historical benchmark.
- **Hierarchical binary merge** in `mnemos/consolidation/phases.py`: Phase 2 dedup merges clusters >2 via pairwise hierarchical steps (size-aware target per step) instead of one-shot N-way. The LLM never sees more than two memories at once, so the "output roughly the size of one input" intuition holds even for deep clusters; compounding compression at each level is mitigated by the new prompt's size-scaling language.
- **`consolidation_lock` parameter** on `memory_store` MCP tool: agents can flag prose-format memories (runbooks, long docs, code blocks) as don't-cemelify at store time instead of needing a follow-up `memory_update` call.
- **CML-vs-prose format guidance** in the `memory_store` tool description: explicit direction to the agent on when to use CML (facts, decisions, configs, preferences, warnings) vs when to use prose (runbooks, long docs, code, creative writing).
- **CML fidelity benchmark** (`benchmarks/cml_fidelity_bench.py`): format-level content parity test on a 20-memory / 209-fact hand-curated corpus split into 15 fact-dense production-style entries and 5 longer narrative-style entries so both ends of the compression range are directly measured rather than asserted. Uses the prose → CML transformation as the measurement lens to show the CML format can hold every atomic fact that equivalent prose would have held. Separate from the LongMemEval `--cml` retrieval-parity runs (ranking parity) and from the consolidation-quality bench (cluster-merge compression).
  - **Overall preservation**: Opus 100%, Sonnet 98.1%, Haiku 98.1%, gpt-4o 96.2%, gpt-4o-mini 95.2%, Llama 3.3-70B 88.5%, Qwen3-32B 80.6% partial, Minimax m2.5 54.0% partial.
  - **Narrative compression** (validates the "up to 60%" claim): gpt-4o 0.39×, gpt-4o-mini 0.48×, Sonnet 0.52×, Haiku 0.55×, Opus 0.59×, all at 90–100% preservation.
  - **Dense compression** (the more conservative regime): Claude tier + gpt-4o-mini in the 0.74–0.86× range (14–26% smaller) at 97–100% preservation.
  - Per-subset split table and per-memory breakdowns in [`benchmarks/README.md`](benchmarks/README.md#4-cml-fidelity-format-level-content-parity--cml_fidelity_benchpy).
- **Honest CML compression framing**: the "35–60% fewer tokens" claim in earlier drafts was calibrated against a single narrative example in the README. The fidelity bench now measures both regimes directly: 14–26% on fact-dense production prose and 41–61% on narrative prose. The main README now quotes 14–60% with the input-density dependence called out explicitly and cites the bench numbers for each end of the range.
- **Three-way dedup on store actually wired up**: the CML-subject tier of `_unified_dedup` was documented but previously a no-op. Now a store attempt whose first line uses the same `<prefix>:<subject>` as an existing memory in the same project contributes a candidate into the cross-encoder rerank pool, alongside FTS and vector signals.
- **Claude automemory disable note** in the README installation section: explicit guidance to turn off Claude Code's built-in `autoMemoryEnabled` (and Claude Desktop / claude.ai "Reference past chats") so a parallel memory system does not compete with Mnemos.

### Formalized (existing features that got proper names)
- **Subcategory column**: the second level of the project hierarchy
  that was always there informally, now a proper indexed column
- **`valid_from` / `valid_until` columns**: the temporal model that was
  already driving decay and supersession detection, now queryable fields
- **Real-time contradiction detection on store**: extends the Nyx cycle
  Phase 4 contradiction logic into immediate detection at write time
- **Nyx cycle naming**: the background consolidation cycle, internally
  called the "dream cycle" through v9, formally renamed to the **Nyx
  cycle** in v10. Νύξ is the Greek primordial goddess of Night, mother
  of Hypnos (sleep) and the Oneiroi (dreams); the naming keeps the
  Mnemosyne family thread without pop-culture contamination from
  alternatives like Hypnos (hypnotism) or Morpheus (The Matrix).
  All code, schema, and tag identifiers updated to match (`run_nyx_cycle`,
  `nyx_insights`, `nyx_state`, `--nyx` CLI flag, `nyx-cycle` tag string,
  etc.)

### Changed
- Default storage path moved to `~/.mnemos/memory.db`
- LLM consolidation prompts generalized to remove personal user profile
- Namespace-aware multi-user support added to the storage layer (no auth
  in core; auth is intentionally a transport-layer concern)

---

## [9.3] - 2026-03-11

### Added
- Weekly memory health check job
- Stripped metadata from memory embeddings to improve dedup precision

### Changed
- General memory system hardening pass and code cleanup

---

## [9.1] - 2026-03-08

### Added
- **Auto-widen on thin results**: when a project-filtered search returns
  fewer than three hits, automatically broadens to a cross-project search
  to surface relevant context from other categories

---

## [8.2] - 2026-03-02

### Added
- Migration to FastEmbed (multilingual e5-large, ONNX) as the embedding
  backbone, replacing earlier Ollama-based embeddings
- Local LLM utilities for the Nyx cycle consolidation phases
- Orphan vector cleanup added to nightly consolidation

### Fixed
- `sqlite3.Row.get()` compatibility bug in the memory embedding helper

---

## [8.0] - 2026-02-19

### Added
- **Continuous exponential temporal decay**: replaced earlier stepped
  decay buckets with `exp(-λ * days_since_access)`. Episodic and semantic
  layers get separate half-lives (~46 and ~180 days respectively).
- **Decay floor** at 10% so old memories never disappear entirely from
  ranking
- **Evergreen tag** that opts a memory out of decay completely
- **`last_confirmed` field**: tracks when a memory was last verified by
  the user, used as a ranking boost
- **Nyx cycle Phase 4 (Contradict)**: detects temporal evolution and
  supersession between memories on the same topic during consolidation,
  flagging conflicts in `memory_links` with `relation_type='contradicts'`
- **Knowledge-dense session briefing** that replaced the earlier
  topic-only memory map

### Changed
- Hourly embed-sync timer to catch fire-and-forget embed failures
- Hybrid search threshold extracted as a tuneable constant

---

## [7.1] - 2026-02-16

### Added
- **Single unified database**: merged the separate vec DB into the main
  memory database so memories, FTS, and embeddings share one SQLite file
  with atomic transactions
- **AND-default FTS queries** with OR fallback for high precision
- **Importance access decay**: memories accessed less often slowly drift
  toward lower importance over time

---

## [7.0] - 2026-02-15

### Added
- Complete `memory-mcp.py` rewrite from a thin Node wrapper to a full
  in-process Python MCP server
- **Synchronous embedding on store** instead of fire-and-forget
- **Three-way deduplication on store**: FTS keyword overlap, CML subject
  matching, and vector cosine similarity, all reranked by a cross-encoder
- **Dynamic importance**: access count thresholds auto-bump memory
  importance (5 accesses → at least 6, 10 → at least 7, 20 → at least 8)
- **Expanded CML notation**: added `∴` (therefore), `~` (uncertain /
  approximate), `…` (continuation), `↔` (mutual), `←` (back-reference),
  `#N` (memory ID reference) plus a quantitative shorthand table
  (`≥` `≤` `≈` `≠` `↑` `↓` `×`)
- Switched memory embeddings from Ollama (Qwen) to FastEmbed (nomic ONNX)
  for CPU-native inference

---

## [6.0] - 2026-02-13

### Added
- **`project` as the canonical hierarchy root**: after dropping the
  legacy `category` and `source` columns, `project` became the single
  organizational axis
- **CML (Condensed Memory Language)**: token-minimal memory format with
  type prefixes (`D:` `C:` `F:` `L:` `P:` `W:`) and relation symbols
  (`→` `∵` `△` `⚠` `@` `✓` `✗` `∅`)
- CML migration tool and consolidation engine
- Conflict detection on stores against existing CML subjects
- Compact session map for compressed briefings
- FTS fallback warning when full-text search misses
- `embed-status` command for embedding coverage reports

### Changed
- Dropped legacy `category` and `source` columns from the schema
- Replaced the Node-based MCP wrapper with a Python implementation
- General memory system cleanup pass: removed dead code paths and the
  unused Node server

---

## [5.0] - 2026-02-10

### Added
- FTS-based deduplication on store
- Status filters (`active` / `archived` / `all`) at query time
- Trimmed session digest

### Changed
- Simplified ranking formula
- Consolidated query logic into a shared module
- Removed several unused features

---

## [3.0] - 2026-02-08

### Added
- Hybrid FTS5 + vector search
- Initial deduplication
- Cross-memory links table
- Memory versioning concept

---

## [2.0] - 2026-02-05

### Added
- **FTS5 full-text search** replacing basic SQLite LIKE queries
- Basic importance field on memories
- Project categories for organizing memories by topic

### Changed
- Improved search relevance by weighting recent memories higher

---

## [1.0] - 2026-01-28

### Added
- **Initial memory system**: the thing that replaced `memory.md`. A Python
  script that stores and retrieves text memories in a single SQLite table.
  Basic store, search, get, update. Nothing fancy. Just "I got tired of a
  flat markdown file that Claude re-reads on every session start, so I
  wrote a database for it."
- This is the version where the itch got scratched. Everything that follows
  is months of iterating on "how do I make this actually good"

---

[10.0.0]: https://github.com/draca-glitch/Mnemos/releases/tag/v10.0.0
