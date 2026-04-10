# Changelog

All notable changes to Mnemos. Dates are from the original private development
repository, where the system existed under an internal name (`agent-memory`)
before being open-sourced as Mnemos in this repo.

This changelog documents real version history. Mnemos was not built on a
weekend; it grew through nine internal iterations over months of personal use,
each one adding or removing features based on what actually improved retrieval
quality and what only added complexity.

> **A note on the git history in this repository**: I did not use git for
> this project until I created my first GitHub account on April 10, 2026,
> the same day I pushed Mnemos here. The system has months of evolution
> behind it. This repository has one commit, because that is when I started
> versioning it. The version progression below is real and the dates are
> accurate.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/).

---

## [10.0.0] - 2026-04-09 (first public release)

The initial open-source release. Equivalent to the private `v10` running in
production at the time of the LongMemEval benchmark run that motivated this
repo.

### Added
- **Pluggable storage backends**: `MnemosStore` abstract base class with
  `SQLiteStore` (default), `QdrantStore` (production hybrid), and
  `PostgresStore` (stub for community contribution)
- **Namespace-aware multi-user support** in the storage layer (no auth in
  core; auth is intentionally a transport-layer concern)
- **`mnemos ingest` CLI command** and `mnemos.ingest` module: walk a folder
  or single file, extract text from supported plain-text formats, chunk
  if necessary, and store as memories with provenance metadata
- **Pluggable extractor API** (`register_extractor`) so community can add
  PDF, EPUB, eml, docx and other formats without touching core
- **CLI rebrand**: full `mnemos` command-line tool with `add`, `search`,
  `get`, `update`, `delete`, `stats`, `briefing`, `digest`, `map`,
  `embed-status`, `doctor`, `prime`, `consolidate`, `ingest`, `serve`
- **`mnemos doctor`**: health check for schema, FTS sync, embedding
  coverage, and stale memories
- **LongMemEval benchmark runner** under `benchmarks/`, with reproducible
  98.1% Recall@5 result on the standard 470-question hybrid mode
- **OpenAI-compatible LLM client** for the consolidation phases (works
  with OpenAI, Ollama, OpenRouter, DigitalOcean Gradient, Together.ai,
  Groq, Fireworks, etc.) with graceful fallback when no LLM is configured
- **Expanded CML notation**: added `∴` (therefore), `~` (uncertain /
  approximate), `…` (continuation), `↔` (mutual), `←` (back-reference),
  `#N` (memory ID reference) plus a quantitative shorthand table
  (`≥` `≤` `≈` `≠` `↑` `↓` `×`)
- Public-facing documentation, ARCHITECTURE.md, and the README that
  contextualizes Mnemos against MemPalace

### Refined (existing concepts formalized)
- **Subcategory column** on memories: an explicit refinement of the
  project hierarchy that has been the canonical organization model
  since v6. Hierarchy was always there, the second level just got a
  proper name and an index in v10.
- **Explicit `valid_from` / `valid_until` columns**: a formalization of
  the temporal model that was already driving exponential decay,
  `last_confirmed` recency tracking, and dream cycle Phase 3 supersession
  detection in earlier versions. v10 lifts these into queryable fields
  and adds a `valid_only` filter at query time.
- **Real-time contradiction detection on `memory_store`**: extends the
  dream cycle Phase 3 (which has been detecting contradictions on
  consolidation runs since v8) into immediate detection at the moment a
  new memory is written. Same logic, same `memory_links` table, same
  `relation_type='contradicts'`. The only difference is when it runs.

### Changed
- Internal package layout reorganized as a proper installable Python
  package (`mnemos/`, `mnemos/storage/`, `mnemos/consolidation/`)
- Default storage path moved to `~/.mnemos/memory.db`
- LLM consolidation prompts generalized to remove the personal user
  profile baked into the private version

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
- Local LLM utilities for the dream cycle consolidation phases
- Orphan vector cleanup added to nightly consolidation

### Fixed
- `sqlite3.Row.get()` compatibility bug in the memory embedding helper

---

## [8.0] - 2026-02-17 (late)

### Added
- **Continuous exponential temporal decay**: replaced earlier stepped
  decay buckets with `exp(-λ * days_since_access)`. Episodic and semantic
  layers get separate half-lives (~46 and ~180 days respectively). This
  established the temporal model that v10 later formalized into explicit
  `valid_from` / `valid_until` fields.
- **Decay floor** at 10% so old memories never disappear entirely from
  ranking
- **Evergreen tag** that opts a memory out of decay completely
- **`last_confirmed` field**: tracks when a memory was last verified by
  the user, used as a ranking boost and as the seed concept for the
  validity windows that v10 made explicit
- **Dream cycle Phase 3 (Contradict)**: detects temporal evolution and
  supersession between memories on the same topic during consolidation,
  flagging conflicts in `memory_links` with `relation_type='contradicts'`.
  This is the foundation that v10 extended into real-time contradiction
  detection on store.
- **Knowledge-dense session briefing** that replaced the earlier
  topic-only memory map

### Changed
- Hourly embed-sync timer to catch fire-and-forget embed failures
- Hybrid search threshold extracted as a tuneable constant

---

## [7.1] - 2026-02-17 (mid)

### Added
- **Single unified database**: merged the separate vec DB into the main
  memory database so memories, FTS, and embeddings share one SQLite file
  with atomic transactions
- **AND-default FTS queries** with OR fallback for high precision
- **Importance access decay**: memories accessed less often slowly drift
  toward lower importance over time

---

## [7.0] - 2026-02-17

### Added
- Complete `memory-mcp.py` rewrite from a thin Node wrapper to a full
  in-process Python MCP server
- **Synchronous embedding on store** instead of fire-and-forget
- **Three-way deduplication on store**: FTS keyword overlap, CML subject
  matching, and vector cosine similarity, all reranked by a cross-encoder
- **Dynamic importance**: access count thresholds auto-bump memory
  importance (5 accesses → at least 6, 10 → at least 7, 20 → at least 8)
- Switched memory embeddings from Ollama (Qwen) to FastEmbed (nomic ONNX)
  for CPU-native inference

---

## [6.0] - 2026-02-17 (early)

### Added
- **`project` as the canonical hierarchy root**: after dropping the
  legacy `category` and `source` columns, `project` became the single
  organizational axis used by every later version. This is the
  foundation that v10 refined with the explicit `subcategory` field.
- **CML (Compressed Memory Language)**: token-minimal memory format with
  type prefixes (`D:` `C:` `F:` `L:` `P:` `W:`) and relation symbols
  (`→` `∵` `△` `⚠` `@` `✓` `✗` `∅`)
- CML migration tool and consolidation engine
- Conflict detection on stores against existing CML subjects (the
  earliest ancestor of v10 real-time contradiction detection)
- Compact session map for compressed briefings
- FTS fallback warning when full-text search misses
- `embed-status` command for embedding coverage reports

### Changed
- Dropped legacy `category` and `source` columns from the schema
- Replaced the Node-based MCP wrapper with a Python implementation
- General memory system cleanup pass: removed dead code paths and the
  unused Node server

---

## [5.0] - 2026-02-16 (late)

### Added
- FTS-based deduplication on store (the precursor to the v7 three-way
  dedup)
- Status filters (`active` / `archived` / `all`) at query time
- Trimmed session digest

### Changed
- Simplified ranking formula
- Consolidated query logic into a shared module
- Removed several unused features

---

## [3.0] - 2026-02-16

### Added
- Hybrid FTS5 + vector search (the foundation of every later version)
- Initial deduplication
- Cross-memory links table
- Memory versioning concept

---

## [Pre-3.0] - 2026-02-16

### Added
- Initial commit: memory system as part of the original server scripts bundle
- Earliest working version with basic store / search / get / update over a
  single SQLite table

---

[10.0.0]: https://github.com/draca-glitch/mnemos/releases/tag/v10.0.0
