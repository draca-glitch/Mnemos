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

## [10.3.3] - 2026-04-16 (revert graceful degrade; state rerank as required)

### Changed

- **Removed the defensive rerank-off graceful degrade added in v10.3.2.**
  The cross-encoder is canonical — Mnemos's benchmark numbers and the
  `relates` silent-link refinement both require it. Pretending otherwise
  with graceful degradation paths added code without adding honesty.
  The honest API is:
  - `mode=vec` → explicit Tier-1-only, works without rerank
  - `mode=rerank` → requires `MNEMOS_ENABLE_RERANK=1`; if disabled,
    rerank() throws/returns empty and `_detect_contradictions` returns
    `[]` (user's choice to cripple the pipeline)
  - `mode=llm` → requires both rerank AND `MNEMOS_LLM_*` env vars

  If you've explicitly opted out of rerank, pick `mode=vec`. Don't ask
  Mnemos to fake a behavior it's not designed to deliver.

- **Docs in `constants.py` now state rerank-requirement explicitly** for
  the `rerank` and `llm` modes rather than implying they gracefully
  degrade. Terse and accurate: you get what you configure.

### Rationale

v10.3.2 tried to make "user disabled rerank but asked for rerank mode"
still produce warnings by falling back to vec-mode behavior. That
conflated opt-outs with misconfiguration. The right contract is: the
feature requires the component, the user either configures it or picks
a different mode. Defensive reinterpretation hides bugs; clear errors
surface them.

---

## [10.3.2] - 2026-04-16 (fix: contradiction detection honors MNEMOS_ENABLE_RERANK=0)

### Fixed

- **Contradiction detection now honors the reranker opt-out.** In v10.3.0,
  the three-tier pipeline would call `rerank()` directly in Tier 2
  regardless of `Mnemos.enable_rerank` / `MNEMOS_ENABLE_RERANK=0`.
  Consequences depending on environment:
  - On a machine where the Jina model still loaded successfully, the
    opt-out was silently negated — the ~500 MB the user was trying to
    save got loaded anyway during the first contradiction check.
  - On truly constrained machines where the model import failed, the
    rerank call raised, was caught, and `_detect_contradictions`
    returned `[]` — dropping ALL contradictions silently.

  Fix: when `mode=rerank` and `self.enable_rerank` is False, degrade
  gracefully to the `mode=vec` path (all vec-gated candidates → flagged
  as contradicts, same as pre-v10.3 behavior, with warnings). Users who
  opted out of rerank keep getting contradiction detection; they just
  lose the `relates` silent-link refinement that required the rerank
  scorer. That tradeoff is honest: you can't distinguish "same topic
  complementary" from "same topic conflicting" without either a
  cross-encoder or an LLM.

- **LLM mode without rerank** now also works. Tier 2 is skipped, every
  vec-gated candidate goes straight to LLM classification at Tier 3.
  More expensive per-pair (no topical prefilter to drop unrelated
  candidates) but functional. Users running `MNEMOS_CONTRADICT_MODE=llm`
  with rerank disabled accept that cost explicitly.

### Matrix (post-fix)

| mode   | rerank enabled | rerank disabled |
|--------|----------------|-----------------|
| off    | no detection   | no detection    |
| vec    | vec→contradicts| vec→contradicts |
| rerank | three-tier     | degrade to vec  |
| llm    | three-tier+LLM | skip Tier 2, LLM on all vec-gated |

---

## [10.3.1] - 2026-04-16 (stop-hook session summary + decay audit)

### Added

- **Reference stop-hook session summary** (opt-in via `MNEMOS_STOP_SUMMARY=1`).
  When enabled, the reference `scripts/mnemos-session-hook.sh` writes one
  episodic memory at session end containing session metadata: session id,
  timestamp, working directory, assistant turn count, tool-call breakdown
  by name, first/last user prompt snippets. Purely structural extraction
  via `jq` on `CLAUDE_TRANSCRIPT` — no LLM call. Next session's briefing
  picks up the summary by recency, giving cross-session continuity
  without violating the "in-session LLM is authoritative" principle.

- **Optional stop-hook nag** via `MNEMOS_STOP_NAG=1` for callers who want
  a "session was long but stored nothing" reminder. Also opt-in, also
  non-LLM.

Both defaults remain off. If a session consistently ends without useful
stores, the correct fix is in-session prompting (CLAUDE.md rules, tool
descriptions), not a bolted-on pipeline. These hooks are escape hatches.

### Verified (no code change, but worth documenting)

- **Decay math** matches the `~46d/~180d` half-life claim advertised in
  docs. `DECAY_RATE=0.015` (ln(2)/46 ≈ 0.015) gives episodic half-life
  of 46.2 days; `DECAY_RATE_SEMANTIC=0.00385` (ln(2)/180 ≈ 0.00385) gives
  semantic half-life of 180 days. Applied at query time in the ranking
  expression, not as a destructive rewrite.
- **Two complementary decay mechanisms** exist and both work as expected:
  1. Query-time exponential boost decay (in the ranking SQL), governs
     search result ordering
  2. Write-time `access_count` decay (-1 per week of inactivity) in the
     Nyx cycle bookkeeping phase, prevents stale access counts from
     inflating importance forever
- **Last Nyx run** confirmed via `consolidation_log` audit trail
  (v10.2.1 feature now useful for exactly this kind of audit).

---

## [10.3.0] - 2026-04-16 (three-tier contradiction detection with `relates` link)

### Why this matters

The v10.2.x contradiction detector scored topical similarity (via
cross-encoder rerank) and flagged anything above 0.35 as "contradicts."
The rerank answers "are these about the same topic?" not "do these say
opposite things?", so complementary same-topic pairs got flagged as
contradictions noisily. Over a long-running deployment this trains the
user to ignore warnings, which defeats the purpose.

v10.3.0 adds a `relates` link type for the middle zone. Moderate-score
pairs get silently linked (no warning), while only high-score pairs or
LLM-classified conflicts emit warnings. False-positive noise goes away;
real contradictions still surface.

### Added

- **`relates` link type** (alongside existing `contradicts`, `reflects`,
  `evolves`, `supersedes`, `enables`). No schema migration needed —
  `memory_links.relation_type` is free-form text, this is a new sentinel
  value.

- **`MNEMOS_CONTRADICT_MODE` env var** with four values:
  - `off` — disable contradiction detection entirely
  - `vec` — Tier 1 only (vec gate, no rerank); all vec-gated candidates
    → `contradicts`. Matches pre-v10.3 behavior for users who explicitly
    want it.
  - `rerank` (default) — Tier 1 + Tier 2. Vec gate + cross-encoder rerank
    with two thresholds:
    - `CONTRADICTION_RERANK_MIN` (0.35): below → skip, not even topical
    - `CONTRADICTION_RERANK_HIGH` (0.60): above → `contradicts` + warn
    - Between MIN and HIGH → `relates`, silent link, no warning
  - `llm` — Tier 1 + Tier 2 + Tier 3 LLM classification. Each rerank
    survivor is classified by LLM into one of {contradicts, refines,
    evolves, relates, unrelated}. Requires MNEMOS_LLM_* env vars.

- **Enriched warning shape**. Each warning now includes `classification`
  (which of the 5 classes) and `suggested_action` (`link:contradicts`,
  `link:refines`, `link:evolves`, `link:relates`, `no_action`). Silent
  `relates` links are persisted but not surfaced to the caller.

### Changed

- **`CONTRADICTION_RERANK_THRESHOLD` constant renamed to
  `CONTRADICTION_RERANK_MIN`**, with a backward-compat alias preserved
  so v10.2.x imports keep working. New companion `CONTRADICTION_RERANK_HIGH`
  introduces the silent-link zone.

- **`contradiction_warning` string** now summarizes by classification:
  `⚠ relationship flag(s) detected: 2 contradicts, 1 refines`. Previous
  version just said `⚠ 3 potential contradiction(s) detected`, which
  lumped noisy `relates`-type matches in with real conflicts.

### Classification semantics (LLM mode)

- **contradicts**: explicit conflict (A says X, B says not-X on same subject)
- **refines**: B refines/expands/corrects A's fact (same subject, added detail, no conflict)
- **evolves**: B is a temporally-later update of A (A was true then, B is true now)
- **relates**: same topic but complementary, no conflict, no temporal order
- **unrelated**: different topics despite surface similarity (no link stored)

### Backward compatibility

Existing `contradicts` links remain untouched. The default mode (`rerank`)
produces a superset of the v10.2.x link graph plus new `relates` links
for pairs that previously were either warned-about-noisily or silently
dropped. Callers that inspect `contradictions` results should handle the
new `classification` field (falls back to "contradicts" if absent for
mode=vec pre-v10.3 compat).

### Calibration

The canonical false-positive case from v10.2.x (memories #2043 and #2045,
both dominance self-insights, complementary not contradictory, sim=0.58)
now classifies as `relates` under the default `rerank` mode: silent link
persisted, no warning emitted.

---

## [10.2.3] - 2026-04-16 (memory_bulk_rewrite: find-and-replace across memories)

### Added

- **`memory_bulk_rewrite` MCP tool** (6th tool). Find-and-replace across
  memories with a preview-commit flow. Default `dry_run=true` returns per-
  memory before/after snippets without touching the DB; caller commits
  with `dry_run=false`. `max_affected` cap aborts before any write if
  the pattern would modify more memories than allowed — prevents runaway
  rewrites. Re-embeds every modified memory (content changed means
  embedding must change too). Supports both plain substring (default)
  and Python regex (via `use_regex=true`). Namespace-scoped, active-only.

  Real-world motivation: last night's cleanup required rewriting "Monica"
  to "Madoka" across 6 consolidated memories. That took ~30 round-trips
  of get+update. This tool collapses the same operation into one call
  with a dry-run preview before commit.

  Why 6 tools now: still 4 CRUD (`memory_store`, `memory_search`,
  `memory_get`, `memory_update`) + 1 schema-discovery (`memory_list_tags`)
  + 1 batch-operation (`memory_bulk_rewrite`). Bulk rewrite is a
  distinct operation category — not CRUD (doesn't operate on a single
  memory or return search results), not schema discovery (mutates
  content). The "4 tools and not 45" principle holds: each tool here
  represents an operation category that cannot be collapsed into an
  existing tool's parameter.

### Safety invariants (contract)

- `dry_run=True` is the default. A caller who omits the flag gets a
  preview, not a write.
- `max_affected` defaults to 50. Exceeded → error returned, zero writes.
- Namespace isolation enforced at the SQL level (WHERE namespace = ?).
- Archived memories excluded (WHERE status = 'active').
- Re-embedding runs via `Mnemos.update()`, which also updates the FTS
  index. No silent drift between stored content and its search
  representation.

### API

```python
mnemos.bulk_rewrite(
    pattern,            # substring or regex
    replacement,
    project=None,       # optional scope
    tags=None,          # optional scope
    dry_run=True,
    max_affected=50,
    use_regex=False,
    preview_chars=120,
)
# Returns: {matched, affected, changes[{id,before,after,diff_chars}],
#           dry_run, error?}
```

---

## [10.2.2] - 2026-04-16 (opt-in tool_usage logging)

### Added

- **`tool_usage` table + opt-in write path**. When `MNEMOS_TOOL_USAGE_LOG=1`
  every MCP tool call records `(tool_name, called_at)` — no arguments, no
  content, no IDs. Useful for health-check tooling that wants to answer
  "has the MCP server been responsive?" without parsing stdin/stdout logs.
  Default off for consistency with retrieval_log, though the privacy
  footprint is essentially zero since no user content is captured.

  Schema:
  ```sql
  tool_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    called_at TEXT DEFAULT (datetime('now', 'localtime'))
  )
  ```

  Backend API: `MnemosStore.log_tool_usage(tool_name)`. Base class no-op;
  SQLiteStore does the INSERT. MCP server calls it in `tools/call`
  dispatch when the flag is set. Failures swallowed — diagnostics only.

### MCP deployment completeness

With v10.2.2, a Mnemos MCP server provides every analytics table an
operator health-check script expects: `retrieval_log` (search history),
`consolidation_log` (Nyx audit), `tool_usage` (tool call diagnostics).
All three are opt-in; enable via env vars per deployment.

---

## [10.2.1] - 2026-04-16 (consolidation_log always available, clean audit API)

### Changed

- **`consolidation_log` and `nyx_state` tables are now created at first DB
  connection** (in `SQLiteStore.init_schema`) rather than only on the first
  Nyx cycle run (`_migrate_nyx_schema`). Previously, deployments that used
  Mnemos purely as an MCP server without ever running `mnemos consolidate
  --execute` would lack these tables entirely, breaking health-check tooling
  and "last run" queries that read `consolidation_log` defensively.
  `_migrate_nyx_schema` is retained as a safety net for older DBs that
  predate this change — CREATE IF NOT EXISTS makes it idempotent.

### Added

- **`MnemosStore.log_consolidation_run()`** method for clean orchestrator
  API. Backends that want Nyx-run audit trails override (SQLite does so
  with an INSERT into `consolidation_log`); the base is a no-op. The Nyx
  orchestrator now calls `store.log_consolidation_run(...)` at run end
  instead of issuing raw SQL, symmetric with the `log_retrieval()` pattern
  introduced in v10.2.0.

### Why this matters for MCP deployments

With v10.2.1, a Mnemos MCP server pointed at a fresh DB has every table
that production health-check tooling (Epsilon-style `memory-health-check.py`
or equivalents) expects: `memories`, `embed_meta`, `embed_vec`,
`memory_links`, `nyx_insights`, `retrieval_log`, `consolidation_log`,
`nyx_state`. That closes the last "Mnemos doesn't have the schema the
operator's scripts expect" gap.

---

## [10.2.0] - 2026-04-16 (opt-in retrieval logging for real-query analytics)

### Added

- **`retrieval_log` table + opt-in write path**. When `MNEMOS_RETRIEVAL_LOG=1`
  (or passing `enable_retrieval_log=True` to `Mnemos(...)`) every successful
  `memory_search` call persists one row per returned memory: the query
  text, the memory_id, a timestamp, and an optional session_id. Default
  off for privacy (queries often contain sensitive content, users must
  opt in consciously).

  Why: real retrieval traces are the right input for benchmark generation,
  retrieval quality analysis, and autoimprove cycles that tune search
  parameters against actual query distribution rather than synthetic
  golden sets. Without this, teams running Mnemos in production have no
  way to measure "are we serving the right memories?" after the fact.

  Schema (compatible with existing Epsilon-style deployments):
  ```sql
  retrieval_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    query TEXT NOT NULL,
    retrieved_at TEXT DEFAULT (datetime('now', 'localtime')),
    useful INTEGER DEFAULT NULL,
    session_id TEXT DEFAULT NULL
  )
  ```

  Backend API: `MnemosStore.log_retrieval(query, memory_ids, session_id=None)`.
  Default implementation is a no-op; SQLiteStore overrides with a batched
  INSERT. Qdrant and Postgres backends can override if retrieval analytics
  are desired there too.

  Failure semantics: logging is best-effort side channel. Any exception
  during log write is swallowed; search results are returned regardless.
  Callers can rely on search correctness independent of log success.

### Not yet implemented (planned for follow-up)

- `consolidation_log` table for Nyx run audit trail (when/what ran, phase
  outcomes, errors). Queued as the natural companion to retrieval_log.
- `useful` flag write path: schema allows a later UPDATE to mark a logged
  retrieval as helpful/unhelpful, enabling supervised quality signals.
  No MCP tool yet to emit that flag — the column is reserved.

---

## [10.1.2] - 2026-04-16 (session-hook ergonomics: briefing truncation + CWD priming)

Two small but daily-visible improvements to the session-hook pipeline that
callers inject at session start (and optionally on first user prompt).

### Fixed

- **Briefing truncation now sentence-aware with ellipsis marker**. Previous
  `content[:180]` raw slice left fragments like `"we don't` that read as
  mid-sentence even when they technically landed on a word boundary. New
  `_briefing_line()` helper prefers sentence-ending punctuation (`. ! ?`)
  over clause boundaries (`; ,`) over word boundaries, and always appends
  ` …` when truncation occurred. A line cut at `"...(2026-04-11). …"` now
  reads as cleanly truncated instead of dangling.

### Added

- **CWD → project/subcategory heuristic on `Mnemos.prime()`**. New optional
  `cwd` and `cwd_map` parameters. When a working directory matches a
  configured path prefix (e.g. `/root/work/mnemos → project=dev,
  subcategory=mnemos`), the inferred project filters results and the
  project/subcat tokens are prepended to the vec query. Without this,
  bare `/root` CWD signals produced vec queries that matched random
  memories across all projects. Applications with known repo layouts
  override `Mnemos.CWD_PROJECT_MAP` (class attribute) or pass `cwd_map`
  at call time. Defaults to empty list, so existing callers see no
  behavior change.

### Session hook pattern (for reference)

These two fixes only deliver full value when paired with a hook that
fires on **first user prompt** (not just session start). At SessionStart
the only context signal is CWD; at first-user-prompt the actual
question is available as a vec query. The repo now ships a reference
hook script at `scripts/memory-session-hook.sh` demonstrating the
three-hook pattern (SessionStart / UserPromptSubmit / Stop) that wraps
`Mnemos.briefing()` and `Mnemos.prime()` for Claude Code, Cursor, and
other MCP clients. See `docs/session-hooks.md` for wiring instructions.

### Design principle: no LLM at stop-time

The reference stop-hook deliberately does NOT call an LLM to summarize
what the session covered. The in-session LLM is authoritative for
memory decisions and already has full context; a post-hoc LLM pass
would duplicate effort, create split responsibility, and cost twice
per session. If a session ends without any stores, the correct fix is
in-session prompting (CLAUDE.md rules, tool descriptions) rather than
a second pipeline.

---

## [10.1.1] - 2026-04-16 (embed_vec schema compatibility)

### Fixed

- **`SQLiteStore` now supports both embed_vec schema variants** found in the
  wild. sqlite-vec vec0 virtual tables can be declared either as
  `vec0(embedding float[N])` (Mnemos default, rowid-based PK) or
  `vec0(id INTEGER PRIMARY KEY, embedding float[N])` (explicit-id PK). The
  latter does NOT expose `rowid` as a queryable column, which broke all
  `search_vec` / `_store_embedding` / hard-delete paths when Mnemos was
  pointed at a database created by other tooling using the explicit-id
  pattern.
  Fix: schema detection at first use, cached per connection
  (`_get_vec_join_col`). `search_vec` picks `ev.id` or `ev.rowid`
  automatically. `_store_embedding` uses the corresponding insert flow
  (pre-assign meta_id then insert vec with explicit id, vs insert vec
  first then capture lastrowid). Zero migration required for either
  schema; fresh Mnemos installs continue to use the rowid-based default.

---

## [10.1.0] - 2026-04-16 (bandwidth controls + tag discovery)

Three ergonomics features added after three hours of real at-scale use of the
existing API surfaced specific friction points. All three are backward-compatible
additions: existing callers see no behavior change.

### Added

- **`memory_list_tags` MCP tool** (5th tool). Returns every unique tag in the
  namespace with usage count and an example memory ID. Prevents tag drift —
  agents creating synonymous tags (`authoritative` / `canonical` / `verified`)
  because they cannot see what already exists. This is a different category
  of operation than the 4 memory CRUD tools: it introspects the tag schema,
  it does not query or mutate memory content. Think of it as `\dt` alongside
  `SELECT` rather than a new way to query.
  - Params: `project` (optional filter), `min_count` (default 1),
    `order_by` (`count` | `alpha`), `limit` (default 500).
  - CLI: `mnemos tags [--project X] [--min-count N] [--order-by count|alpha]`.
- **`snippet_chars` parameter on `memory_search`**. If set, replaces each
  result's `content` field with a query-matched window of approximately that
  many characters, using SQLite FTS5's built-in `snippet()` function with
  `⟪` `⟫` match markers. Vec-only hits (no FTS match) fall back to a head
  slice of the content. Major token-budget saver: a search hit inside a
  6000-character consolidated memory returns ~250 bytes instead of ~6 KB.
  Default (`None`) keeps full content for backward compatibility. Callers
  needing the full content of a snippeted hit use `memory_get`.
- **`include_linked` parameter on `memory_search`**. If true, folds first-hop
  linked memories into each result as `linked_memories: [{id, project,
  relation, strength, content}]` summaries. Saves round-trips when tracing
  relationship graphs — one search call returns the hit plus everything it
  links to instead of one call per link. Depth=1 only for now.

### Tool count

Core CRUD stays at 4 (`memory_store`, `memory_search`, `memory_get`,
`memory_update`). `memory_list_tags` is the 5th tool but in a distinct
category (schema discovery, not memory ops). README updated to reflect this
framing.

### Backward compatibility

All three are additive. `memory_search` gains optional params that default
to existing behavior. `memory_list_tags` is a new tool, no removal. No schema
migration needed.

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
