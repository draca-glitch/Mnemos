# Changelog

All notable changes to Mnemos. Dates are from the original private development
repository, where the system existed under an internal name (`agent-memory`)
before being open-sourced as Mnemos in this repo.

## [Unreleased]

## [10.42.0] - 2026-10-09 (an agent's memories about itself)

### Added
- **Self memory, behind `MNEMOS_SELF=1`.** An agent can keep memories about
  itself (traits, observed habits, commitments about its own behaviour) in a
  namespace of its own, `self:<agent>`, beside the user's store in the same
  database. `memory_store`, `memory_search`, `memory_get`, `memory_update`
  and `memory_list_tags` take `self: true`; `memory_bulk_rewrite` does not
  (no unattended mass rewrite of a self-model). Results from a self call
  carry the `namespace` they landed in. Off by default: the flag is not
  advertised in `tools/list`, a call that passes it anyway gets an error
  naming the switch, and turning the feature off later hides the namespace
  without touching a row.
- **Agent identity comes from the client.** The name an MCP client declares in
  `initialize` (`clientInfo.name`) is kept per `Mcp-Session-Id` on the shared
  HTTP server (bounded, oldest out) and once per process on stdio, then
  sanitised into the namespace (`Claude Code` -> `self:claude-code`). Several
  harnesses on one server each get their own self. `MNEMOS_AGENT` is the
  fallback for a client that declares no name, and the identity of the CLI.
- **`mnemos --self <command>`** runs any CLI command against the self
  namespace (`mnemos --self briefing` for a session-start block, `--self add`,
  `--self search`, `--self consolidate` for the Nyx cycle). Refuses with a
  clear message when `MNEMOS_SELF` is off or `MNEMOS_AGENT` is unset.
- `Mnemos.self_view(agent)` and `MnemosStore.for_namespace(ns)` (SQLite
  implements it; other backends raise NotImplementedError, so the feature is
  SQLite-only for now).
- Tests: `tests/test_v1042_self.py` (9).

### Changed
- `handle_message(mnemos, msg, session_id=None)`: transports pass their
  client handle through; the HTTP transport now reads `Mcp-Session-Id` on
  every request for this purpose (it was issued but ignored before).

## [10.41.1] - 2026-10-06 (the mover runs as the store's owner)

### Security
- **`mnemos move` no longer runs with more privilege than the store's owner.**
  10.41.0 let root move a store owned by another user and then chown the copy
  to that user. Every path such a move touches sits in a directory the user
  controls, so the user could swap a path for a symlink between two steps and
  have root write, chown or chmod a file of their choosing. The caller must
  now own the store, root included: run `sudo -u OWNER mnemos move ...`. The
  refusal exits 1 and names the owning uid. Run as the owner, the kernel
  enforces the owner's permissions on every step.
- Root moving a root-owned store refuses a source or destination directory
  another user can write: one not owned by root, world-writable,
  group-writable for a group other than root's, or carrying a POSIX ACL.
- **The copy in progress is private.** It is created with mode 0600,
  exclusively and without following a symlink, before SQLite opens it; the
  source's permissions and group are applied through the file descriptor once
  the copy is verified. In 10.41.0 it was created with the default umask and
  was readable by others until the final chmod.

### Changed
- Directories the move creates are no longer chowned; the owner creates them.
- Tests: `tests/test_v1041_1_move_hardening.py` (4). 515 pass.

## [10.41.0] - 2026-10-06 (move a store safely)

This update is for when you want to move your database somewhere else, since
the main work directory is not the most optimal place for it.

### Added
- **`mnemos move DEST` relocates the store without losing writes.** Moving a
  live SQLite file with `mv` fails three ways: a process that still has the
  old file open keeps writing to it after the rename, a raw copy of a WAL-mode
  file omits rows still in the `-wal`, and a config that still names the old
  path makes SQLite create a fresh empty database there. `mnemos move` refuses
  with exit code 1 while it can see another process on the store (stop the MCP
  or HTTP server, then retry), copies through the backup API under SQLite's
  exclusive lock, runs `quick_check` and compares the row count of every table
  against the source, gives the copy the source's owner and permissions, and
  keeps the old file as `<name>.moved-<timestamp>` for rollback. The old path
  is then replaced by a symlink in one atomic rename, so a stale `MNEMOS_DB`
  still reaches the real store and the path is never free for SQLite to
  create an empty database on (`--no-link` skips the symlink). `DEST` is a
  file path, an existing directory, or a path ending in `/` (a directory,
  created if missing). While the symlink is in place a second run against the
  same destination is a no-op. On a failure the source is unchanged, no file
  is left at the destination and directories the run created are removed.
  `--json` prints the result.
- Two in-use checks. On Linux `/proc` is scanned for any process that has the
  file open, which catches idle connections in every journal mode; a default
  Mnemos store is a rollback-journal database, where an idle connection holds
  no file lock. Then SQLite's exclusive lock, on every platform, which sees
  every connection that has read or written a WAL database and every active
  transaction on a rollback-journal one. The refusal names the process ids
  when it has them.
- A last line of defence for the process neither check can see (another
  user's process when not root, an idle one where `/proc` is missing, an
  `open()` that lands between the scan and the lock): the kept file is left in
  rollback-journal mode. SQLite refuses to write a rollback-journal database
  whose file was renamed under an open handle, so such a process gets an
  error on its next write instead of committing into the abandoned file. WAL
  has no such guard, which is why the kept file is switched out of it; the
  destination keeps the source's journal mode. If you roll back by renaming
  the kept file into place, switch it back with `PRAGMA journal_mode=WAL`.
- `--db PATH` picks the file to move (default `MNEMOS_DB`). The mover is
  schema-agnostic: it loads no extension and never opens the file as a Mnemos
  store, so it also relocates other SQLite databases that sit next to the
  memory store. Library entry point: `mnemos.storage.move.move_database()`.
- Known limits. An unseen process can still read the kept file and serve
  stale data until it is restarted. Without the privilege to change
  ownership the copy belongs to whoever ran the move. ACLs and extended
  attributes are not copied. If the symlink cannot be created the move still
  completes and says so; the old path is then free. On Windows there is no
  symlink swap and a short unlocked window around the rename.
- Tests: `tests/test_v1041_move.py` (25, including the lost-write race found
  in review, run against a second process).

### Changed
- Public author identity is Mikael Wedlund (`CITATION.cff`, `pyproject.toml` authors, README). The GitHub account remains `draca-glitch`.

## [10.40.1] - 2026-09-07 (validity can be cleared)

### Fixed
- **No sanctioned way to unset `valid_until`.** Both `memory_update` and
  `mnemos update` dropped `None`, so once a date was on a memory, whether by
  hand or by a Phase 4 `EVOLVED` verdict later judged wrong, it could only be
  moved, never removed. With 10.40.0 defaulting `valid_only` to True that
  meant the memory silently left every default search for good. The MCP tool
  now treats `null` or `""` on `valid_from`, `valid_until` and `subcategory`
  as "clear" (the schema admits null; omitting the key still leaves the field
  alone), and the CLI gains `--clear FIELD`, repeatable, combinable with other
  updates. 10 new tests, 487 pass.

## [10.40.0] - 2026-09-07 (current by default, confirmation has producers)

Minor bump: two defaults change and clients that relied on them will see
different results.

### Changed
- **`valid_only` defaults to True everywhere.** `memory_search`, `mnemos
  search`, `Mnemos.search`, and the store-level `search_fts`, `search_vec`
  and `search_vec_archived` now return currently valid memories unless told
  otherwise. The 10.39.0 finding said historical retrieval should be the
  explicit option and then left the default alone; on the reference store
  every one of the 40 active memories carrying a `valid_until` was already
  past it and all of them surfaced in every default search. Dedup and
  contradiction candidate searches inherit the store default: a fact whose
  validity ended is neither a duplicate of nor a contradiction to a fresh
  statement of the current one. Pass `valid_only=false` for history. The CLI
  flag `--valid-only` is replaced by `--include-expired`. Reads by ID are
  unaffected. The two vector indexes had opposite defaults before; they
  agree now.
- **A content correction or `verified=true` through `update` records
  `last_confirmed`.** 10.39.0 made the column honest (reads no longer touch
  it) and left it with no writer, so the confirmation boost in ranking was
  draining to zero with nothing to refill it. A caller that rewrote the
  content, or flagged it verified, looked at it; that is the confirmation
  event. `confirmed=false` suppresses it for mechanical edits and
  `bulk_rewrite` passes it. `confirmed=true` alone still confirms without
  changing anything.

### Added
- `verified` is settable through `memory_update` and `mnemos update
  --verified`; it was store-time only before.


## [10.39.2] - 2026-09-07 (review follow-ups)

### Fixed
- `store_link` returns False instead of raising when an endpoint is missing
  or foreign. 10.39.0 introduced the `ValueError`, and none of the six call
  sites in `core` caught it, so a dedup or contradiction hit hard-deleted
  between the search and the link would abort `store()` after the memory row
  was already written: the caller saw an error, the memory persisted, a retry
  duplicated it. A link that cannot be made must never fail the write that
  wanted it.
- Namespace scoping moved into the statements. `delete_memory` (hard and
  soft) and `get_merged_sources` no longer rely on a `get_memory` pre-check;
  the DELETE, UPDATE and nyx_insights lookup carry `namespace` themselves, and
  hard delete runs its embedding, link and row cleanup in one transaction that
  rolls back when the row is not ours. `reembed_mismatched` now uses the
  namespace argument it always received. `move_embedding_to_archive` keeps its
  pre-check because the underlying helper is connection-level and shared with
  consolidation; `namespace` is not an updatable column, so the check cannot
  race a namespace change.

### Added
- Regression coverage for the implicit-rowid `embed_vec_arch` schema that
  10.39.0 declared supported without exercising. The prefiltered archived KNN
  is now tested against both vec0 layouts.

### Changed
- The KNN validity filter binds today's date once instead of calling
  `date('now','localtime')` in the subquery. Measured: no difference (the
  cost of the prefiltered KNN over 10.38.1 is the eligibility JOIN itself,
  about +1.4 ms active and +3 ms archived on a 6k-row store, and it is the
  price of not losing eligible hits). Kept because it shares one clock
  source with the Python-side validity check in linked expansion.

### Documented
- 10.39.0 stopped refreshing `last_confirmed` on reads. `CONFIRMATION_BOOST_SQL`
  in the FTS ranking gives a 30-day and 90-day lift on that column, so
  frequently read memories lose a small recency lift until explicitly
  confirmed. The boost is never a penalty; ordering among unconfirmed
  memories is unchanged.
- 10.39.0 also stopped traversing archived nodes as bridges in linked
  expansion. That is a separate decision from validity filtering and is
  now recorded as one: on stores where consolidation leaves links pointing at
  merged originals, depth 2 and 3 expansion reaches less than before.


## [10.39.1] - 2026-09-07 (compact links)

### Changed
- Search results and linked summaries no longer include Nyx audit links
  (`contradiction-cleared`) by default. Those rows record that a pair was
  judged compatible, not that the memories relate, and on a mature store they
  outnumber every semantic relation combined (5180 of roughly 6360 link rows on
  the reference deployment, p90 of 24 links per memory). With them gone the
  p90 is 3. Each link also grew by three fields in 10.39.0, which made the
  noise more expensive. Opt back in with `memory_search(include_audit_links=true)`,
  `mnemos search --audit-links`, or `store.get_links(ids, include_audit=True)`.
  Consolidation and oversized-memory remediation still see every link; the
  filter is on the output surface only. The set lives in
  `constants.AUDIT_LINK_RELATIONS`.

## [10.39.0] - 2026-09-06

### Fixed
- Directional links now carry `source_id`, `target_id`, and `direction` in
  search results and linked summaries. The stored relation always describes
  source -> target; an incoming `superseded_by` link no longer lacks the
  information needed to distinguish the replacement from the obsolete fact.
  Oversized-memory remediation preserves direction when copying these links.
- SQLite direct ID reads, updates (including embedding-only updates), deletes,
  bulk fetches, snippets, and archive moves are namespace-scoped. Link creation
  rejects missing or foreign endpoints; link and merge-source reads filter
  legacy cross-namespace references.
- Active and archived vector search restrict eligible vector IDs before KNN.
  Namespace, project, category, status, and validity filters cannot lose all
  results because unrelated nearer vectors filled a fixed global pool.
- `valid_only=True` also filters expired and future-dated linked memories.
  Invalid or archived nodes cannot serve as bridges during multi-hop expansion.
- `memory_get` updates access telemetry without changing `last_confirmed`.
  Existing confirmation dates are preserved, not retroactively reinterpreted.
- Project-only updates regenerate embeddings and their provenance hashes,
  matching the project field already included by `prep_memory_text`.

### Added
- Explicit confirmation through `memory_update(id=..., confirmed=true)`,
  Python `Mnemos.update(..., confirmed=True)`, or `mnemos update ID --confirm`.
  Use only after checking a source or receiving explicit user confirmation.
- Synthetic regression coverage for namespace boundaries, link direction,
  validity traversal, filtered vector recall, confirmation, and embedding drift.

Validation: 448 tests pass, including 23 new regression cases.

No schema migration or model configuration change. See
[`docs/memory-correctness.md`](docs/memory-correctness.md) for client semantics
and upgrade notes.

## [10.38.1] - 2026-08-21 (record what the floor admitted, not just how much)

### Added
- **Phase 4 records the cosine that admitted each pair alongside the verdict it received.** The floor is now calibrated (10.38.0) but nothing measures whether it is calibrated *correctly*: the verdict links carry constant strengths (0.8 for `contradicts`, 0.1 for `contradiction-cleared`), so the score that let a pair through is discarded the moment it is judged. That leaves evidence for what a floor **costs** and none for where it should **sit**, and any further tightening trades known cost for unknown recall. A `contradict_audit` table (`source_id`, `target_id`, `cosine`, `verdict`, `judged_at`) now pairs the two, written at the single point where every judged pair is classified, so one cycle produces a labelled set and the operator can ask the store directly: what is the lowest cosine that ever produced a real contradiction? Everything below it is demonstrably safe to drop. Behaviour is otherwise unchanged, and the row is written regardless of `--execute` so a dry run can calibrate without mutating the store.

## [10.38.0] - 2026-08-20 (the floor calibrates itself)

Both items reported by a fleet host running a 600-memory Swedish/English store
on e5-large + jina-v2, and both reproduced here before being fixed.

### Changed
- **The phase-4 contradiction floor calibrates to the store instead of being a constant.** `CONTRADICT_MIN_SIM=0.60` is a property of the embedding space it was tuned on, and it is inert on e5-large: measured on a 746-row production store, same-project cosines span [0.7405, 0.9734] with a median of 0.8598, so the floor admitted **100.0% of 65,960 pairs** and the nominator degenerated into an exhaustive scan -- the expensive NLI pass then ran on everything, which is why one host disabled its nightly cycle rather than let it grind. The floor is now taken from the store's own distribution (`MNEMOS_CONTRADICT_SIM_PERCENTILE`, default p99), which keeps the gate selective whatever the embedder. On the same store that is 9,490 candidate pairs down to 95, a 100x reduction. Setting `MNEMOS_CONTRADICT_MIN_SIM` still pins an absolute floor and turns calibration off; calibration never drops *below* that floor, and `MNEMOS_CONTRADICT_MIN_CANDIDATES` (default 20) stops a high percentile of few pairs from starving a small store.
- **The cycle log now reports what the floor admitted.** A nominator passing nearly every pair is self-evidently broken but nothing said so, so the defect was invisible except as runtime. Phase 4 logs the effective floor and the admitted share, and warns explicitly above 50%.

### Fixed
- **`scripts/clean_tag_amplifier.py` silently stripped classification markers.** `PROTECTED` was hardcoded to two tags standing in for cluster-universality, which cannot be recomputed post-hoc -- but the marker vocabulary is per-store, so any store using different ones lost them on the very run meant to repair damage, indistinguishably from a stripped inherited topic. On the reporting host that was 164 rows across five markers, and 111 of its 182 "repairs" were pure marker loss. Now: `--protected` extends the set, likely markers are auto-nominated (ALL-CAPS or classification-sounding), and the dry run separates amplifier repair from marker loss instead of reporting one conflated row count.

6 new tests, 425 pass.

## [10.37.0] - 2026-08-20 (the version gate inverts)

### Changed
- **The protocol-version gate is a range, not an allowlist.** 10.36.0 shipped a hardcoded list of accepted revisions and 10.36.1 extended it; both were wrong, because the list can only ever name the revisions its author happened to know about, and anything outside it was refused with `-32022` (a hard 400 on Streamable HTTP). `2025-11-25` was the second working revision to be rejected that way. Since every revision in the handshake era shares one `tools/list`/`tools/call` wire format, the gate now accepts any dated revision from `PROTOCOL_LEGACY` up to `PROTOCOL_MODERN` inclusive, named or not, and serves the unnamed ones through the legacy path. Revisions newer than `PROTOCOL_MODERN` are still refused, since their changes are unknowable to this build and `-32022` carries the known list so the client can downgrade. Malformed and pre-legacy values are still refused. `KNOWN_VERSIONS` remains as what `server/discover` advertises -- documentation, not the gate -- and now names `2025-11-25`.

### Fixed
- **A refused request no longer poisons the next one on the same connection.** The Streamable HTTP handler returned its 400 for an unsupported version header *before* reading the request body, so on a keep-alive connection the unread body stayed in the socket buffer and the next request line was parsed as a continuation of it: `Bad request syntax ('{"jsonrpc": ...}POST / HTTP/1.1')`, and a valid request got a spurious 400. The body is now read before anything can return early, which removes the ordering hazard rather than patching this one instance of it. Regression test drives two requests down one `http.client` connection and fails against the previous code.

5 new tests, 419 pass.

## [10.36.1] - 2026-08-20

### Fixed
- **The version gate no longer refuses revisions Mnemos can actually speak.** 10.36.0 introduced the protocol-version check with `SUPPORTED_VERSIONS = [2026-07-28, 2024-11-05]`, which made the two intermediate published revisions (2025-03-26, 2025-06-18) a hard failure: `-32022` on the per-request `_meta` gate, and a 400 on Streamable HTTP. Before the gate existed the header was ignored entirely, so those clients worked — the check turned a working integration into a broken one, which is the opposite of what a compatibility feature should do. Both revisions still open with `initialize` and share the legacy `tools/list`/`tools/call` wire format, so the legacy path already serves them correctly; they are now listed as supported, `server/discover` advertises all four, and `initialize` echoes an intermediate proposal instead of downgrading it. 4 new tests, 414 pass.

## [10.36.0] - 2026-08-20 (the dual-era handshake)

### Added
- **Dual-era MCP protocol support (2026-07-28 alongside legacy 2024-11-05).** The 2026-07-28 spec revision removed the initialize handshake in favor of per-request versioning; Mnemos now serves both eras from the one transport-agnostic dispatch. New: `server/discover` (MUST in 2026-07-28) advertising `supportedVersions`, capabilities, instructions, and cacheability; per-request `_meta` version gate returning `UnsupportedProtocolVersionError` (-32022) with the supported list as retry data; every result carries `resultType: "complete"` and server identity in `_meta` (both additive, so legacy clients are unaffected); `tools/list` is deterministic-ordered and carries `ttlMs`/`cacheScope` per the CacheableResult interface.
- **initialize negotiates instead of asserting.** The server echoes the client's proposed `protocolVersion` when supported and answers with the legacy revision otherwise, instead of hardcoding 2024-11-05 regardless of what the client asked for.
- **Streamable HTTP validates MCP-Protocol-Version.** A request carrying an unsupported version header is refused with 400 and the -32022 error body; absent means legacy and is accepted (required on this transport since spec 2025-06-18).

9 dual-era tests through the real HTTP transport. 410 pass.

## [10.35.2] - 2026-08-19 (the tag amplifier)

### Fixed
- **Nyx no longer amplifies tags through merge and split.** Merge unioned every source's tags unbounded, and the size-guard split copied that full union to every sibling: a merge of N memories split into k pieces produced k memories each claiming all N topics while holding 1/k of the content, compounding nightly as unions merged with unions (field-measured: a 275-char sibling carrying 6KB of tags, sibling groups sharing byte-identical 77-tag strings, seeded by a 479-chunk ingestion's per-chunk tags). The v10.8.0 size guard measures content only, so it never saw them — and since `prep_memory_text` embeds retrieval-relevant tags, the unions also poisoned the affected vectors.
- Each merged memory (and each split sibling) now carries only the tags its own content supports plus the cluster-universal ones — a tag every source carried describes the whole cluster, and pure content-support would strip privacy markers like `STRICTLY-PRIVATE` that legitimately ride on every source without appearing in any content. `TAG_BUDGET` (1024 chars, universal-first, deterministic) backstops the filter.

6 regression tests. 401 pass.

## [10.35.1] - 2026-08-19 (the token id is the truth)

### Fixed
- **The reranker CML probe trusts token ids, not token strings.** Unigram/XLM-R tokenizers echo an unrepresentable character back as its own surface piece: jina-reranker-v2-base-multilingual encodes the because-operator to tokens `['▁','∵']` with ids `[6,3]` where 3 IS `<unk>` — so the string test judged it native, the map never applied, and the cross-encoder scored text whose 3rd most common operator round-trips to nothing (field-reported with the tokenizer transcript). The probe now checks ids against the tokenizer's unk id first; the string and byte-fallback checks remain as fallbacks for fakes and BPE mojibake.
- **The probe is per-symbol, and the map follows it.** `_probe_cml_support` returns the frozenset of missing operators instead of a bool, and the reranker spells out only those: jina-v2-multilingual carries 10 of 13 natively (missing: because-operator + both FTS5 snippet markers) and the ablation's R@1 gain comes from the cross-encoder attending to exactly the markers a bool would have thrown away. Measured against the real tokenizers: jina-v2 loses 3, jina-v1-turbo/tiny lose 1 (✗), gte-modernbert (byte BPE) loses all 13 — the default's behavior is unchanged. `apply_cml_map` gains an optional `symbols` argument; `None` keeps the embedder's full-map contract.

2 regression tests + 2 updated to the set contract. 395 pass.

## [10.35.0] - 2026-08-19 (the store declares its reranker)

### Added
- **`store_settings` and `mnemos settings {get,set,unset,list}`.** First setting: `reranker_model`. A multilingual store had no way to declare that it needs a multilingual reranker; correctness depended on `MNEMOS_RERANKER_MODEL` reaching every launch context separately (unit files, cron, hooks, ad-hoc shells), which is the exact env-drift trap the docs warn about for `MNEMOS_NAMESPACE` and that the embedder escaped in 10.30.0 by making the store self-describing (field report). The declaration is adopted at store open, reaches every context that opens the store, and explicit env still overrides it, per field: an exported value is an instruction, a default is a seed. doctor reports an active adoption, and the language warnings now point at `mnemos settings set reranker_model ...` as the permanent fix instead of the env-var treadmill.
- The reranker deliberately does NOT auto-pin the way the embedder does: a reranker upgrade should reach stores that never declared a preference, and only stores with a declared need resist the default.

3 new tests. 393 pass.
## [10.34.2] - 2026-08-19 (tier-2 provenance check; repair respects the lock)

### Fixed
- **Doctor now compares tier-2 PROVENANCE, not just width.** A normalization flip or same-width model change leaves the archived index geometrically identical while every archived vector is stale, and the completeness check counts vectors, so archived recall degraded under a green health check (field-reported, confirmed on 10.33.1 and 10.34.0). Doctor now reports tier-2 vectors whose recorded model differs from the current identity and points at `mnemos reembed`, which rebuilds both tiers since 10.34.1.
- **The mojibake repair honors `consolidation_lock`.** A memory that QUOTES mojibake to document it must not have its evidence rewritten into a self-negating memory (field case). Locked memories are listed as a check and left untouched; the lock is the existing "do not machine-rewrite this" bit, applied consistently.

2 regression tests. 390 pass.
## [10.34.1] - 2026-08-19 (reembed means the whole store)

### Fixed
- **`mnemos reembed` now rebuilds the tier-2 archived index in the same run.** Three consecutive field migrations needed hand surgery on `embed_vec_arch` because reembed rebuilt only the active tier while every doc and doctor message called it THE migration command; 10.32.2's `reindex-archived --rebuild` existed but was a second command nobody was told to run. The archived index is part of the store, not an optional sidecar.
- **The mojibake scan covers archived memories.** Field case: a damaged archived memory is still served by tier-2 recall, and the scan only iterated active rows, so it was never detected. Archived repairs write content directly (update() re-embeds into the active tier, the wrong index for them) and refresh their tier-2 vectors via reindex-archived's stale-hash pass in the same doctor --migrate run.

2 regression tests. 388 pass.
## [10.34.0] - 2026-08-19 (store-path guard for uncalibrated rerankers; mojibake repair)

### Fixed
- **The legacy dedup/contradiction tiers no longer run on rerankers their thresholds were not calibrated for.** Those tiers sigmoid a raw cross-encoder logit and compare it to constants tuned on jina-v2. Measured on gte-modernbert (the v10.33.0 default), on real store content: identical pairs score 0.92-0.98, merely-related pairs 0.90-0.98, and UNRELATED pairs median 0.68 with outliers past the 0.85 dedup threshold. The identical and related ranges overlap, so no threshold separates them: the tier is structurally unsuited to that model, not mistuned, and on the new default it would have false-blocked stores aggressively. The store path now uses the rerank tier only on calibrated models (`RERANK_CONFIRM_CALIBRATED`), preferring NLI when available and the vec-distance tier otherwise, with a one-line stderr notice. An explicit `MNEMOS_DEDUP_CONFIRM=rerank` is honored only for calibrated models. Search ranking is unaffected; relative scores never had this problem.

### Added
- **Mojibake detection and repair** (field-reported: a single ingestion event that decoded UTF-8 as Latin-1 left memories with damaged text, including CML operators the map can no longer see, since it looks for the real characters). `mnemos doctor` reports affected memories; `mnemos doctor --migrate` applies the deterministic codec inverse, accepted only when the round-trip encodes cleanly and strictly reduces the damage, and re-embeds the repaired memories in the same step. Anything that does not round-trip cleanly is left for a human. The operator map deliberately does NOT learn mojibake variants: that would enshrine corruption as vocabulary.

7 new tests. 386 pass.
## [10.33.1] - 2026-08-19 (startup language check judges the pinned store, not the raw default)

### Fixed
- The HTTP startup language notice read `effective_model()` BEFORE anything had touched the store, and per-store pinning only runs on first store access, so it judged the shipped default rather than what the store actually uses. First observed on the first 10.33.0 production boot: a store pinned to e5 was told its embedder is English-only. The check now samples the store first (which runs adoption) and judges afterwards.
- The regression test for this initially leaked its e5 pin into module state and broke 24 later tests; it restores the defaults in a finally block. Two env/module leaks were found in this release cycle (this one and the HTTP fixture's raw `os.environ` writes), both now documented in the tests they broke.

## [10.33.0] - 2026-08-19 (the light tier becomes the default)

The change issue #4 asked for, shipped with everything the benchmarks said it needs. Two operator notes up front:

- **A bare `mnemos reembed` on an existing store rebuilds IN PLACE under the store's pinned model.** To migrate to the new default, export `MNEMOS_EMBED_MODEL=BAAI/bge-small-en-v1.5 MNEMOS_EMBED_DIMS=384` first. Existing stores are otherwise untouched: they pin to the model their vectors were built with.
- **Rerankers are NOT store-pinned** (they touch no stored data, which cuts both ways): the default reranker change reaches every install on its next restart. Multilingual deployments must set `MNEMOS_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual` on their unit, and Mnemos now warns at store time, server startup and in doctor when non-English content meets an English-only reranker.

### Changed
- **Defaults: `BAAI/bge-small-en-v1.5` (384-dim) + `Alibaba-NLP/gte-reranker-modernbert-base` (149M).** Measured basis, LongMemEval stratified subset: bge-small R@1 92.40 equals bge-base 92.40 and e5-large 92.01 (reweighted) WITH a reranker in the pipeline; without one, e5 is clearly better (88.43 vs 81.29), so the small embedder is only free because the reranker stays. gte-modernbert beats jina-v2 on the discriminating categories (preference R@1 80.00 vs 73.33) at half the size, while both sub-120M candidates land below the no-reranker floor. End-to-end resident cost measured including activations: 4.78 GB (old pair) to 1.57 GB. The old pair remains the documented multilingual tier and is the only combination that works for non-English content.
- **The CML operator map (v2) now covers every non-ASCII operator, and its version is part of embed provenance.** v1 left the arrows and null out because the EMBEDDER's WordPiece vocabulary represents them, which is true and irrelevant: the default reranker's byte-BPE shreds them into byte pieces, and the arrows are the two most common operators in a real store (752 + 182 occurrences). Arrows map to their ASCII forms, which code-trained models have seen constantly. Changing the map changes every normalized vector without changing the model, so `CML_MAP_VERSION` feeds `embed_model_id()` (`+cmlnorm2`) and doctor reports old-map vectors as mixed provenance instead of letting two geometries mix silently.
- Custom rerankers register with fastembed automatically at load (`add_custom_model`); the default requires it, `MNEMOS_RERANKER_MODEL` may point at any HF repo with an ONNX export, and `MNEMOS_RERANKER_MODEL_FILE` overrides the file path.
- MCP warmup logs name the effective models instead of a hardcoded "e5-large/jina reranker".

### Fixed
- **`reembed` dropped the index at the configured default's width but filled it with the store-pinned model** (found in pre-release review): on any pre-flip store, the exact command doctor recommended would have emptied the index and had every insert rejected. Both sides now use the effective configuration, and doctor's advice spells out the export-then-reembed sequence.
- **The CML probe missed byte-fallback BPE.** It only looked for `[UNK]`, and byte-level vocabularies never produce one: operators fragment into UTF-8 byte mojibake instead. A model now counts as representing an operator only if some produced token still contains the character. Consequence worth knowing: gte-modernbert's published screen numbers were measured while it read raw operators as mojibake, and it won anyway; with the map engaged its scores can only be cleaner. The query is now mapped alongside the documents, since a cross-encoder scores the pair.
- **Pre-v10.6 stores (NULL `embed_meta.model`) now pin their index width** even though the model name is unknowable, so the default flip cannot leave their new inserts silently rejected against an old-geometry index.
- The language warnings cover the post-flip mixed case (multilingual embedder, English reranker arriving via the default) at store time, HTTP startup and doctor, naming the model and the one-env-var fix. Previously every check required the embedder to be English too, which is exactly backwards for this release.
- "Rerank disabled" is no longer treated as "English-only reranker" by the language checks: absent is not English. Surfaced by the fix below.
- `test_http_transport`'s fixture wrote `MNEMOS_ENABLE_RERANK=0` to raw `os.environ` without cleanup, leaking into every later test in the process; now monkeypatched.
- Cache healing (10.30.1) runs on the reranker load path too; previously a long-lived server that already held the embedder never healed an interrupted reranker download. Registration failures print the actual error instead of silently skipping.

### Notes
- Dedup/contradiction confirm thresholds were tuned on jina-v2 logits. Store-path behavior on gte is untuned: quality-sensitive deployments should set `MNEMOS_DEDUP_CONFIRM=nli` (recommended anyway) or pin jina-v2. Search ranking is unaffected (relative scores).
- Docs, diagrams and comments swept repo-wide: architecture diagrams are model-agnostic, the tier table is the reference, and the published benchmark numbers remain attributed to the configuration that produced them.
- Tests now derive vector dimensionality from `FASTEMBED_DIMS`, so the next default change costs zero test edits. 381 pass.

## [10.32.3] - 2026-08-19 (session-hook pressure guard, done properly)

### Fixed
- `scripts/mnemos-session-hook.sh` gains the memory-pressure handling that a field deployment added locally, with its three defects fixed. (1) The hook now exports `MNEMOS_MIN_FREE_MB` (default 1500, existing values respected) so under pressure the CLI degrades search vec-only then FTS5 via the package's own guard instead of going silent; it also disables the ONNX arena for these short-lived CLI processes, where arena growth is never repaid. (2) The hard low-memory skip applies only to the `start` and `prompt` branches: `stop` is jq over the transcript plus an opt-in `add` that `MIN_FREE_MB` already degrades gracefully, and skipping it silently cost the session summary for zero memory benefit. (3) The MemAvailable read is crash-proof under `set -euo pipefail`: awk exiting 0 with no output left an empty string that blew up the integer comparison ("integer expression expected"); `${avail:-99999}` closes it.

## [10.32.2] - 2026-08-19 (the tier switch reaches the archived index; doctor respects pinning)

Both field-reported from the first real tier switch on another host, hours after the tier documentation shipped.

### Fixed
- **`reembed` left the tier-2 archived index stranded.** It rebuilds only the active index, and `reindex-archived` could not repair the archived one either: after a model switch every archived row still HAS a vector, just in the wrong geometry, so a backfill that only fills missing rows finds nothing to do, and the completeness check reported the index healthy because it only counted rows. `reindex-archived --rebuild` now drops and recreates the archived index at the effective width and clears `embed_meta_arch` so the backfill genuinely re-embeds. `doctor` gains a width comparison between the two indexes and points at the flag.
- **`doctor`'s dimension check compared against the raw constants instead of the effective (store-pinned) configuration.** A store correctly pinned to a non-default model reported a false "new vectors are being rejected on insert" while inserts succeeded, making `issues_detected` cosmetic. Same bug class as the provenance check fixed in 10.30.0: any doctor check that mentions the embedder must read the effective config, not the constants.

4 regression tests.
## [10.32.1] - 2026-08-19 (language warning at store time)

### Fixed
- The language-coverage detection warned at doctor time and server startup, which is after the damage: the wrong moment to learn your tier cannot read your language is after importing a thousand memories. `store_memory` now warns on the FIRST non-English memory stored under English-only models, in the tool result and on stderr, while the store is young enough that `mnemos reembed` is free. Fires once per process; stores past 200 active memories are left to doctor, where warning on every foreign quote in a mature English store would be noise.

## [10.32.0] - 2026-08-19 (language-coverage detection)

### Added
- **`mnemos doctor` now checks language coverage.** An English-only model pair on a non-English store fails silently: nothing errors, retrieval is just quietly mediocre, and nothing attributes it. Same failure shape as a dimension mismatch or a provenance split, so it gets the same treatment: doctor samples stored content and raises an issue when substantial non-English text coexists with an English-only embedder AND reranker (a check, not an issue, when only one of the two is English-only, since FTS and the multilingual stage still cover the terms).
  The heuristic is a per-memory non-ASCII letter share with CML operators and emoji excluded, thresholded at 0.5%. That number is calibrated against a real English-primary store whose non-English TERMS OF ART are its retrieval keys: those memories measure 0.5-2%, and they are exactly the case where a multilingual reranker still earns its place. Pure-English stores measure 0 and never trip it. Unknown model ids are assumed capable, because a false "your setup is fine" is worse than a dismissible warning.
- One-line startup notice on the shared HTTP server when the same condition holds, for the operator who never runs doctor.
- `docs/usage.md`: measured model-tier table (default / multilingual / mixed) and the two switching rules: changing the embedder requires `mnemos reembed` because stored vectors are only meaningful under the model that produced them, while changing the reranker touches no stored data and is freely reversible.
- `mnemos/language.py`, `SQLiteStore.sample_contents()`. 13 tests.

## [10.31.1] - 2026-08-18 (CML operators reach the reranker too)

### Fixed
- **The reranker was reading raw CML while the embedder read normalized text.** `normalize_cml()` ran inside `embed()` only, so with `MNEMOS_EMBED_NORMALIZE_CML` on the two stages saw different documents. Harmless with the default reranker, which reads the operators natively, and not harmless with any other.
- **The reranker now decides for itself.** `_probe_cml_support()` encodes each operator against the reranker's own tokenizer at load and spells them out for scoring only when that model cannot represent them. Detected rather than configured: a flag is one more thing to set wrong, and the tokenizer can answer the question directly. It logs a line when it engages.
  Measured across the rerankers fastembed offers: `jina-reranker-v2-multilingual` (Unigram, 250k) loses 0 of 8 operators, `jina-reranker-v1-turbo-en` (BPE, 60k) loses 1, `ms-marco-MiniLM-L-6-v2` (WordPiece, 30k) loses all 8. So the need is per-model rather than a property of "small", and a hardcoded list would go stale.
  This matters because the reranker is the component that RESCUES CML: the published ablation has `single-session-preference` R@1 going 53.33% to 80.00% when it is added, and the mechanism is that CML's structural markers are what a cross-encoder attends to. A reranker that maps every operator to one shared `[UNK]` cannot do that, which previously ruled out every small English reranker for a CML store.
- `embed.apply_cml_map()` split out of `normalize_cml()` so the substitution can be used ungated by callers that decide their own policy.

## [10.31.0] - 2026-08-18 (bounded tensor shapes; a reaper on every transport)

### Fixed
- **Unbounded RSS on the shared HTTP server.** Not a leak: the ONNX Runtime CPU arena grows to the high-water mark of the tensor shapes it has served and does not return it while the session lives. Under stdio, process death reset that for free once per session; the v10.27.0 shared server never exits, so it accumulated the union of every shape every attached harness had ever asked for. Measured on two hosts independently, same ceiling: 18.3 GB in the field, 18.28 GB reproducing it here.
  The free-running axis was **padding**, which is easy to get backwards. Jina v2 caps at 1024 tokens and fastembed truncates there, so sequence length looked bounded. But fastembed calls `enable_padding()` with no length, so every batch pads to its own longest member and one long memory drags the other fifteen up with it: shape wanders through [1, 1024] with batch composition. Pinning truncation AND padding to a single length collapses that to one shape.
  `MNEMOS_RERANK_MAX_TOKENS` (default 512) pins both on the tokenizer fastembed already built. This is a policy change, not a reimplementation: token ids, special tokens, the XLM-R pair template and `logits[:,0]` post-processing are untouched, so the published benchmark remains a fastembed score stream. It reaches through a private attribute and degrades to a warning plus fastembed's dynamic padding if that moves. 512 rather than 1024 because attention is quadratic in sequence length.
  `MNEMOS_RERANK_BATCH` (default 16, was fastembed's 64) and `MNEMOS_EMBED_BATCH` (default 32, was 256) bound the batch axis.
  Measured, same workload, arena deliberately ON, reranking the 150 longest memories: **+15.68 GB claimed before, +0.89 GB after**. The second and larger pool now costs +0.00 GB because the shape is already covered. Total peak 2.73 GB with the arena on, against 2.85 GB with it disabled, so the bound gets arena-on latency at arena-off memory.
- **The idle reaper never ran on the HTTP transport.** It was started by `mcp_server.main()` only, which is backwards: a stdio harness reclaims everything by exiting, while the shared server never exits. `MNEMOS_MODEL_IDLE_TTL` on a `mnemos serve --http` unit was a no-op from v10.27.0 until now. Moved to `_resource.start_idle_reaper()`, idempotent, called by both transports.
- **NLI was missing from the reaper entirely.** `nli.py` had no `maybe_unload`, so a quiet night still left both DeBERTa sessions and their arenas resident. On a deployment running `MNEMOS_DEDUP_CONFIRM=nli` and `MNEMOS_CONTRADICT_MODE=nli` that is two of the four ONNX sessions on the process. Added, and wired into the tick.

### Changed
- `MNEMOS_RERANK_MAX_CHARS` now defaults to 0. Clipping characters is a weak proxy for clipping tokens: measured on this store, 1958 chars is 536 tokens but 3000 chars is 934, so a character budget does not bound a tensor. Superseded by the token pin, kept as an escape hatch.

### Notes
- 16 tests (`tests/test_v1031_shape_bounds.py`), including a regression test asserting the HTTP transport starts the reaper.
- `MNEMOS_DISABLE_MEM_ARENA` remains the blunt instrument for hosts that want bounded RSS without any of this, at the measured 14 to 49% latency cost.

## [10.30.1] - 2026-08-18 (heal a broken fastembed cache before model load)

### Fixed
- **Orphaned download artifacts no longer wedge the embedder permanently** (contributed by balaianu). `huggingface_hub` leaves `.incomplete` temp files behind when a download is interrupted and never collects them, and a failed initial download can leave `refs/main` empty, which makes `snapshot_download(local_files_only=True)` return the `snapshots/` parent instead of the hash subdir so onnxruntime fails with NoSuchFile. Both states block the model from ever loading, and they compound: every failed retry adds another dead partial, filling the disk and making the next attempt fail sooner. Reported from the field at 13 GB of orphaned partials across 80 failed downloads of the same 2.2 GB blob.
  `_clean_broken_cache()` now runs before the encoder is constructed. It removes `.incomplete` files older than an hour, which leaves an active download alone since its mtime stays fresh, and deletes empty `refs/main` files only, never the model dir, so already-downloaded files survive and `snapshot_download` raises `LocalEntryNotFoundError` which fastembed already handles by falling through to a network re-download. Per-file errors are swallowed, so healing can never itself prevent a model load. 9 tests.
  Note for anyone hitting this: repeated mid-download deaths usually mean the process is being killed, so it is worth checking `MNEMOS_DISABLE_MEM_ARENA` and `MNEMOS_MIN_FREE_MB` as well. This change cleans up the wreckage; those settings address why the downloads died.

## [10.30.0] - 2026-08-18 (per-store embedder pinning; CML normalization on by default)

### Added
- **Per-store embedder pinning**: on first connection a populated store reads back the model, dimensions and normalization its vectors were actually built with (`embed_meta.model`, plus the declared width of `embed_vec`) and pins the embedder to them. A default is a decision about NEW stores; a populated store already answered the question, and its answer is the only configuration under which its vectors mean anything. Without this, moving a default would break every existing store on upgrade: 768-dim vectors into a 1024-wide index are rejected on insert, so the store silently stops gaining vectors while the old ones keep answering queries and only newly-stored memories go missing from search.
  An explicitly exported `MNEMOS_EMBED_MODEL` / `MNEMOS_EMBED_DIMS` / `MNEMOS_EMBED_NORMALIZE_CML` still wins, per field: an exported value is an instruction, an unset one is only a seed. `doctor` reports any pinning that occurred, so "why is this store not using the model I configured" has an answer. To migrate a store to a new default, export the value explicitly and run `mnemos reembed`.
- `embed.adopt_store_config()`, `effective_model()`, `effective_dims()`, `effective_normalize()`.

### Changed
- **`MNEMOS_EMBED_NORMALIZE_CML` now defaults on.** It is the only change in this series with no trade-off: measured cost is +0.13% tokens, it is a no-op for e5 (0 UNK before and after), and it removes 81% of the `[UNK]` tokens any English WordPiece vocabulary produces on CML. Defaulting it is safe only because of the pinning above, which keeps existing stores on the configuration they were built with.

### Fixed
- `MNEMOS_DISABLE_MEM_ARENA` stays default-off, and the code comment's "~10-15% slower inference" is corrected. Interleaved A/B over four paired rounds measured **+49% embedding latency** with the arena disabled (median 88.5s vs 59.3s). The RSS case for enabling it is strong (21.2 GB after ~25 minutes versus 2.85 GB flat over 3.5 hours, same host and workload) but it is a real trade, not a free win, and long-lived deployments should opt in deliberately.

## [10.29.0] - 2026-08-18 (CML operator normalization for the embedder)

### Added
- **`MNEMOS_EMBED_NORMALIZE_CML`** (opt-in, default off): substitutes CML's relational operators for the words they stand for before tokenization, for the embedder only. Stored content, the FTS5 index, the cross-encoder and what the agent reads back are all untouched, so CML's token savings are kept where they are paid, in context reinjection.
  The defect it addresses is a vocabulary one, not a model-quality one. SentencePiece vocabularies (e5-large and the Jina reranker, 250k) carry the operators; English WordPiece vocabularies (bge, gte, mxbai, nomic, all-MiniLM, 30522) carry none of them and map every one to a single shared `[UNK]`. Measured on a 720-memory production store: 901 UNK tokens under bge, 0 under e5. Because `[UNK]` has one embedding, `D: migration ✓ confirmed` and `W: migration ✗ rejected` contribute the same vector at those positions, so the two become indistinguishable to the bi-encoder. That is a precision failure rather than a recall one, which is the shape the published CML ablation already shows: R@5 moves 98.30% to 98.09% while R@1 on `single-session-preference` falls to 53.33%.
  Mapped: `∵`, `∴`, `△`, `⚠`, `✓`, `✗`, plus `⟪`/`⟫` snippet markers and `═` separators (dropped). Operators an English vocabulary already carries (`→ ↔ ← @ ~ ∅ >`) are deliberately left alone. Measured cost on the same store: UNK 901 to 170, tokens +0.13%; e5 is unaffected (0 to 0 UNK, +0.05% tokens). The residual UNKs are emoji and Thai from verbatim quotes, not CML.
  This is what makes the whole 30522-vocab model tier usable at all, not a bge-specific workaround.
- `embed_model_id()`: provenance string recorded in `embed_meta.model`, suffixed `+cmlnorm` when normalization is on. Normalization changes every vector without changing the model name, so it has to be part of the identity or `doctor`'s provenance check cannot see a flip and the store silently mixes two geometries. `_store_archived_embedding` resolves its `model` at call time for the same reason.
- 15 tests (`tests/test_v1029_cml_normalize.py`).

## [10.28.0] - 2026-08-18 (configurable embedder; vector-index rebuild)

### Added
- **`mnemos reembed`**: rebuilds the entire active vector index under the currently configured model and dimensions. `embed-fill` only covers rows with NO vector, so it could never repair a model or dimension change, which is exactly when every row already has a vector from the wrong encoder. Takes a `VACUUM INTO` snapshot first (skip with `--no-backup`), reads the existing vec0 DDL and substitutes only the width so a store declaring `id INTEGER PRIMARY KEY` keeps that shape, then batch re-embeds every active memory. `--dry-run` reports the target model, previous width and row count without touching the index. The tier-2 archived index is untouched; it has its own `reindex-archived`.
- **`doctor` vector-dimension check**: compares the declared width of `embed_vec` (read from `sqlite_master`, since vec0 exposes no introspection pragma) against `FASTEMBED_DIMS` and reports a mismatch with both remedies, rebuild or revert. The failure this catches is quiet rather than loud: sqlite-vec rejects every mismatched insert, so the store stops gaining vectors while the existing ones keep answering queries, and search degrades only for anything stored after the change.
- `SQLiteStore.get_vec_dims()` and `SQLiteStore.reset_vec_index(dims)`.
- 16 tests (`tests/test_v1028_embed_config.py`).

### Fixed
- **`MNEMOS_EMBED_DIMS` now exists.** `docs/features.md` has documented it as the knob for running a different-dimensional embedder for several releases, while `constants.py` hardcoded `FASTEMBED_DIMS = 1024`. Following the documentation had no effect. It is now `int(os.environ.get("MNEMOS_EMBED_DIMS", "1024"))`, unchanged by default.
- **Embedding prefixes follow the model family.** `embed()` hardcoded e5's `"passage: "` / `"query: "` scheme, so switching `MNEMOS_EMBED_MODEL` to a BGE model glued a meaningless prefix onto every document and every query, on both sides of the index, with no error. `_prefixes()` now resolves the pair per family (e5, BGE English v1.5, and no prefix for anything unrecognized, because a wrong prefix is worse than none). The e5 default path is byte-identical: stored-vs-recomputed cosine on an existing 719-memory store is 1.0000.
- **Dimension mismatches name the knob.** sqlite-vec already rejects a mismatched insert, but its message reports the numbers without saying which setting produced them. `embed()` now raises once, up front, with the value to set and the command to run.
- `doctor`'s mixed-provenance issue pointed at `mnemos embed-fill`, which cannot repair the condition it was detecting. It now points at `mnemos reembed`.
- Comment on `HYBRID_MIN_MEMORIES` said vector search activates "above this count"; the test is `>=`, so it activates at exactly 10.

## [10.27.0] - 2026-08-18 (shared HTTP transport: one process, many harnesses)

### Added
- **Streamable-HTTP transport** (`mnemos serve --http 127.0.0.1:PORT`, `--unix PATH`): a long-lived Mnemos process that any number of MCP harnesses (Claude Code, Codex, Grok, ...) attach to over localhost, so the ONNX models (e5-large, Jina reranker, NLI) load exactly ONCE. Motivation, measured on a 31G host with three stdio-spawned copies: 16.4G + 5.5G private RSS of identical weights plus ORT arena growth; the shared file-backed RSS of the .onnx files was ~12MB, so disk cache sharing does not help. Request/response subset of MCP streamable HTTP: POST body is one JSON-RPC message or a batch, response is `application/json`; notifications get 202; GET is 405 (no server push); `Mcp-Session-Id` issued on initialize and accepted but not required, because tool state is process-global by design. Dispatch is serialized through one process-wide lock so concurrent harnesses queue briefly instead of growing per-session ORT arenas without bound. Binds are refused unless localhost (or a 0600 user-only unix socket); `MNEMOS_HTTP_ALLOW_NONLOCAL=1` overrides explicitly. Model warmup is once-per-process (`initialize` from a second client is a no-op). 11 tests; `docs/shared-http-server.md` documents the systemd user unit and per-harness config rewiring.

### Changed
- `mcp_server.py`: JSON-RPC dispatch extracted into transport-agnostic `handle_message(mnemos, msg)`; the stdio loop and the HTTP handler both call it. stdio behavior is unchanged (same methods, same error shapes, same stderr logging), and `mnemos-mcp` with no flags remains the default single-client transport.

## [10.26.2] - 2026-07-25 (audit log survives a pre-v10.2.1 consolidation_log)

### Fixed
- **A pre-v10.2.1 `consolidation_log` permanently disabled the audit log and
  latched surge mode on.** `CREATE TABLE IF NOT EXISTS` is a no-op on an
  existing table, so a DB whose `consolidation_log` predates `phase_details`
  (v10.2.1) never gained the column: `_migrate_nyx_schema` backfilled only the
  v10.5.0 counters. Every `log_consolidation_run()` INSERT then failed against
  the missing column, and the failure was swallowed. `MAX(run_at)` stayed NULL,
  so each cycle read `Last run: never`, phase 1 re-triaged the whole store,
  `is_surge` latched on permanently, and phases 2 and 3 ran at the surge caps
  forever. The backfill loop now carries per-column declarations and includes
  `phase_details TEXT DEFAULT '{}'`.
- **Audit write failures are no longer silent.** `log_consolidation_run()`
  still never raises (the audit log is a side channel, not a correctness
  dependency), but it now reports the reason on stderr. A swallowed write is
  indistinguishable downstream from a cron that never fired, which is what let
  the bug above survive undetected for months.

Diagnosed on the angssatra deployment; reproduced and tested upstream.

## [10.26.1] - 2026-07-25 (phase 4 queue mode is not an LLM call budget)

### Fixed
- **Queue mode discarded flagged pairs it had already paid to find.**
  `phase_contradict` truncated its candidate list to `max_eval`
  (`NORMAL_MAX_CALLS // 2`, so 15 at the defaults) before the `judge="queue"`
  branch. That budget exists to cap LLM calls, and queue mode makes none:
  queueing a candidate is a SQL insert. On a production store running the
  two-tier split (zero-LLM nightly, LLM weekly) the nightly finder flagged
  174 pairs, queued 15, and dropped the other 159, which re-flagged from the
  scan cache the next night and were dropped again. Queue mode is now bounded
  by `NLI_FINDER_MAX_PAIRS` (default 200), the "pairs advanced per run" knob,
  which also keeps the cosine finder path capped since nothing upstream bounds
  candidates there. `judge="llm"` behaviour is unchanged; queued pairs already
  bypassed the truncation when consumed.

## [10.26.0] - 2026-07-25 (raw_connection escape hatch; single-sourced version)

Completes the 10.25.0 store abstraction repo-wide.

### Added
- **`MnemosStore.raw_connection()`**: the sanctioned escape hatch for
  SQLite-native maintenance tooling. Returns the backend's underlying
  `sqlite3.Connection` (SQLiteStore: its own; QdrantStore: its wrapped
  SQLite side) or None on backends without one (Postgres). The Nyx
  orchestrator and repo scripts now use it instead of groping
  `_get_conn()` / `_sqlite._get_conn()` via hasattr. Nyx's SQL surface is
  deliberately NOT abstracted behind interface methods; it is a
  SQLite-native engine, and a declared capability is more honest than a
  fake abstraction. Zero `_get_conn` references remain outside
  `mnemos/storage/`.

### Changed
- **Version is single-sourced from `mnemos/__init__.py`** via
  `[tool.setuptools.dynamic]`. The 10.25.0 release shipped with a stale
  `__version__` because pyproject.toml and `__init__.py` each carried a
  copy; now there is one place to bump and pip reads it at (re)install
  time.

## [10.25.0] - 2026-07-25 (store abstraction: core.py no longer touches SQLite)

First release with an external contribution. Thanks @balaianu.

### Changed
- **All direct SQLite access extracted from core.py into the store interface**
  (#1, @balaianu). 12 new methods on `MnemosStore` with no-op defaults,
  implemented in `SQLiteStore`: `find_oversized_memories`,
  `find_cml_subject_matches`, `mark_retrieval_useful`,
  `find_memories_by_content`, `get_briefing_memories`, `get_digest_memories`,
  `get_project_map`, `get_unembedded_memories`, `get_embed_coverage`,
  `health_check`, `get_coherence_mismatches` + `reembed_mismatched`,
  `get_vector_provenance`, `check_archive_lifecycle`. core.py: 0 `_get_conn`
  refs (was 19), 0 `hasattr(_get_conn)` guards (was 13). Pre-1.0 note: the
  store ABC surface grew; custom backends inherit working no-op defaults.
- `_summarize_quick_check` moved from core.py to sqlite_store.py (it
  summarizes `PRAGMA quick_check`, which is SQLite-specific); re-exported
  from `mnemos.core` for compatibility.
- `reembed_mismatched` takes no embed callables anymore; it lazy-imports the
  embed helpers itself, matching `get_embed_coverage` and
  `get_coherence_mismatches`.

### Fixed
- **Maintenance surface on non-SQLite backends errors explicitly again.**
  The extraction initially replaced the "requires SQLite" errors from
  `doctor`, `bulk_rewrite`, `embed_fill`, `embed_status`, and
  `remediate_oversized` with silent empty successes, letting a hypothetical
  non-SQLite backend produce a vacuous "healthy" doctor report (same bug
  class as the 10.23.0 empty-store false all-clear). New
  `MnemosStore.supports_maintenance` capability flag (False by default,
  True on SQLiteStore) restores the explicit errors.
- Restored incident-anchored WHY comments that the extraction dropped
  (re-split idempotency and findall-vs-search hazard, tag substring-leak,
  integrity-check-first WAL-corruption rationale, empty-store rationale).

## [10.24.1] - 2026-07-11 (phase-4 queued-candidate cosine no longer stubbed)

### Fixed
- **Consumed queued contradiction candidates reported `sim=1.000` for every
  pair.** When the LLM tier drains `contradiction-candidate` links queued by an
  earlier zero-LLM run, it rebuilt each pair with a hardcoded `1.0` cosine
  because the consumer read only `(source_id, target_id)` from `memory_links`.
  The queue insert already stores the finder cosine in `strength`
  (`round(min(cos, 0.99), 3)`), so the consumer now reads it back. Real cosines
  (measured 0.89 to 0.95 on the affected prod pairs) surface again in the phase-4
  log instead of a misleading 1.000. Cosmetic only: the flagged set and judge
  verdicts were always correct; the stubbed sim just made recall-first finder
  output look like a scoring failure. Pairs queued before this fix still show
  1.000 until re-queued (their original cosine was never persisted for those).

## [10.24.0] - 2026-07-07 (archive-side embedding lifecycle: no more tier-2 leaks)

Driven by a forensic audit of the prod embedding-row lifecycle (the open item
from the 10.19.0 coherence check's first prod run): 51 archived memories had
no tier-2 vector, clustered exactly on consolidation days; 3 orphan arch rows
survived hard-deleted memories; embed_meta tables created before the
localtime DDL still carried a UTC embedded_at default, making synchronous
embeds look hours older than their memory's creation.

### Fixed
- **Consolidation archived memories without moving their vectors.** Both raw
  status-UPDATE sites in phases (merge, supersede) now go through a new
  `archive_memory(conn, mid, tag_suffix)` helper that archives and moves the
  vector into the tier-2 index in one step. `cleanup_orphan_vectors` remains
  as the end-of-cycle safety net.
- **Soft delete now moves the vector to the tier-2 index** instead of
  leaving it in the active index (`delete_memory(hard=False)`); the move
  never blocks the archive itself.
- **Hard delete purges tier-2 rows too.** `delete_memory(hard=True)` used to
  stop at the active index, leaving embed_meta_arch/embed_vec_arch rows
  behind forever.
- **Doctor: UTC embedded_at dialect detected and migrated.** Tables whose DDL
  predates the localtime default are rebuilt with the canonical schema and
  existing values converted via `datetime(x, 'localtime')` (DST-correct,
  detected from sqlite_master so the check needs no marker and is
  idempotent). Pre-migration backup taken.
- **Doctor: orphan tier-2 rows detected and cleaned** (`--migrate`), plus
  advisory counts for archived memories without a vector and rows with
  legacy pre-canonical hashes, both pointing at `reindex-archived`.

### Added
- `reindex-archived` re-embeds legacy-hash rows: tier-2 meta rows with a
  pre-v10.6 truncated (16-char) hash were embedded under a pre-canonical
  formula; they are re-embedded and re-hashed, reported as
  `legacy_reembedded`. Backfill of missing vectors unchanged and idempotent.
- `embed_vec_arch` is created with the declared-PK schema
  (`vec0(id INTEGER PRIMARY KEY, ...)`) on new stores, matching what the
  active side already handles; existing implicit-rowid arch tables keep
  working through a new `_arch_join_col` detection used by every arch-side
  query (store, search, move, cleanup, doctor). No rebuild of existing
  tables: both schemas are supported in the wild, same policy as embed_vec.
- Conn-level lifecycle helpers `move_embedding_to_archive_conn` /
  `_store_archived_embedding_conn` shared by the store methods, the
  consolidation phases and external plumbing.
- 9 new tests (273 total).

## [10.23.1] - 2026-07-05 (hermetic test suite: scrub MNEMOS_* env before import)

### Fixed
- Test suite was not hermetic: on a box with a live Mnemos deployment,
  MNEMOS_* env vars from the login shell (e.g. MNEMOS_NLI_BACKEND=onnx,
  MNEMOS_CONTRADICT_MODE=nli) plus a real NLI export in
  ~/.cache/mnemos/nli-onnx leaked into the suite and failed 4 tests
  (torch-fallback, multi-hop depth 1/2, rerank guard). Leak paths: import-time
  constants in mnemos/constants.py and the _onnx_model_dir() cache-dir
  fallback. New tests/conftest.py deletes all MNEMOS_* vars at conftest
  import (before any test module, and therefore mnemos, is imported) and
  points MNEMOS_NLI_ONNX_DIR at an empty temp dir so the real export is
  invisible. Deleting MNEMOS_DB doubles as a safety net against a test ever
  touching a real store. Found on TMG legacy (4 failed / 260 passed at
  v10.22.0 and v10.23.0); with the conftest, 264 pass under a hostile env.
  Test-only change, no runtime impact.

## [10.23.0] - 2026-07-05 (doctor flags an empty store instead of blessing it)

### Added
- Empty-store detection in `doctor`: a store with 0 active memories for the
  resolved namespace is reported as an issue with a config hint instead of
  "healthy". A doctor run against the wrong MNEMOS_DB path or a mismatched
  MNEMOS_NAMESPACE previously passed every check vacuously (each check
  verified nothing and reported ok), which let a false all-clear survive
  during the 2026-07-05 embed-formula incident. Two hint variants: a fully
  empty database points at MNEMOS_DB (expected once for a brand-new store),
  and a database whose memories all live in other namespaces lists the
  per-namespace counts and points at MNEMOS_NAMESPACE. Populated stores gain
  a "Store populated: N active memories" check line. No auto-fix on purpose:
  an empty store has exactly two causes, fresh install (nothing to fix) or
  misconfiguration (the fix is the caller's env, and guessing where the real
  data lives is how a tool initializes an empty store at a wrong path and
  buries the actual problem). 4 new tests; 264 pass.

## [10.22.0] - 2026-07-05 (embed-text excludes Nyx bookkeeping tags; coherence stays green)

### Fixed
- prep_memory_text folded ALL tags into the canonical embed-text, but Nyx
  rewrites bookkeeping tags (consolidated, nyx-split, merged-into-*,
  split-from-*, split-part-*) on every consolidation cycle. Because the
  coherence hash is computed over the embed-text, every memory that had
  ever been consolidated reported as content/vector mismatched (observed:
  723/723 on the production store) and `doctor` recommended a full
  re-embed that would just recur on the next Nyx run. The embed-text now
  excludes those churning, retrieval-irrelevant tags via stable_tags(),
  so coherence stays green across consolidation; semantic tags are kept.
  Adopting it requires a one-time re-embed of the active set (done on
  Epsilon: 723 verified, stale=0). 260 tests pass unchanged.

## [10.21.1] - 2026-07-05 (phase-4 finder budget also caps cache-flagged pairs)

### Fixed
- NLI_FINDER_MAX_PAIRS capped only fresh NLI scoring, so cache hits above
  threshold advanced to the judge/queue regardless of the budget. A
  MAX_PAIRS=0 "judge nothing new" run still flooded the judge with the
  whole cached-flagged backlog (observed: a judge-only run processed 50
  pairs where 11 were expected). The budget now caps pairs ADVANCED per
  run (fresh-scored or cache-flagged); cached pairs below threshold stay
  free, unadvanced pairs backfill on later runs, and MAX_PAIRS=0 is a
  true no-op. Normal runs with ample budget are unchanged. Verified by
  simulation across MAX_PAIRS in 0, 5, and 200.

## [10.21.0] - 2026-07-05 (contradiction judge gains an UNRELATED verdict)

### Fixed
- The phase-4 contradiction judge presumed its two inputs were about the
  same topic (the prompt asserted "these memories are about similar
  topics") and offered only SUPERSEDED / EVOLVED / CONTRADICTS /
  COMPATIBLE. But candidacy is exhaustive same-project: the cosine floor
  (CONTRADICT_MIN_SIM=0.60) is inert on multilingual-e5-large, where a
  measured 45% of all active pairs clear 0.78, so the judge is routinely
  handed same-project pairs about entirely different subjects. With no
  way to say "these are unrelated", the model scope-conflated shared
  vocabulary into false CONTRADICTS (observed: a hardened-VPS firewall
  memory flagged as contradicting an on-prem office LAN memory). The
  prompt now decides same-subject first and can answer UNRELATED;
  unrelated pairs are tombstoned as contradiction-cleared so they never
  re-enter the finder/judge loop. Validated by A/B on real
  same-project-different-subject pairs plus unambiguous contradiction
  controls: UNRELATED cleanly separates different subjects with zero
  regression on true contradictions (RAM 64 vs 32, nginx vs Apache,
  blood type, backup time all still flagged). Also replaced the ambiguous
  "weekly vs daily backup" CONTRADICTS example (those coexist) with a
  same-job time clash.

## [10.20.0] - 2026-07-03 (phase 0.5 removed; Nyx is namespace-scoped)

### Removed
- Phase 0.5 (Cemelify) no longer exists in the Nyx cycle. Rewriting
  already-stored memories every night is generation against content
  whose fidelity is the product: it drifted exact strings on weaker
  models, was disabled on every known deployment, and ran ungated by
  the phase list, so a zero-LLM `--phases 1,2,4,6` run on a
  key-configured host still fired LLM calls (observed in the field:
  28x 401 against a stale key). `MNEMOS_NYX_CEMELIFY` is gone;
  `cemelify()` itself remains for ingest-time shaping of NEW content.
  Stored content is never rewritten in place.

### Fixed
- Nyx loaders are namespace-scoped: `load_embeddings` and
  `load_memory_meta` read only the active namespace (default: the same
  resolution as every write path; both accept an explicit `namespace=`).
  Since v10.6.0 the write side was namespace-correct but reads saw every
  namespace in the DB file, so on a multi-tenant deployment phase 2
  could cluster tenant A's memories with tenant B's, merge them into
  the current namespace and archive both originals. Found in the field
  on a second deployment; single-namespace stores were never affected.

## [10.19.1] - 2026-07-03 (second test-fixture DB leak closed)

### Fixed
- `tests/test_store.py` carried the same fixture defect fixed for the
  tier2 tests in 10.17.0 and missed in that sweep: setting MNEMOS_DB in
  the environment before importing only isolates when that module is the
  first to import mnemos.core, which under pytest collection it never
  is. Every suite run leaked fixture memories into the developer's real
  default DB (found independently on a second machine the same day: 174
  rows there, 419 on the first; always namespace 'default', real
  namespaces untouched). Both fixtures now construct
  `SQLiteStore(db_path=...)` explicitly; a repo-wide grep confirms no
  env-var DB fixtures remain.

## [10.19.0] - 2026-07-03 (doctor verifies content/vector coherence)

### Added
- Doctor check: every active memory's embed-text hash is re-derived from
  current content and compared to `embed_meta.text_hash` (recorded at
  embed time since v10.6 "so staleness is detectable later"; this is the
  later). Catches content mutated without a re-embed: direct SQL writes
  bypassing the API, or a write-path bug, both previously invisible to
  retrieval and to every other check. `doctor --migrate` re-embeds
  flagged rows. Near-total mismatch on stores of 20+ checkable rows is
  reported as an embed-text format change across versions rather than
  row corruption; rows predating hash tracking are skipped.

## [10.18.0] - 2026-07-03 (memoized NLI finder: the nightly sweep stops re-proving old negatives)

### Added
- `nli_scan_cache`: the phase-4 finder's line-level score is a pure
  function of the two contents, so it is memoized keyed on content
  hashes. Unchanged pairs cost nothing on later runs; a changed hash
  invalidates exactly that pair. `MNEMOS_NLI_FINDER_MAX_PAIRS` now
  budgets only NEW scorings, and never-scored pairs backfill across
  subsequent nights. Measured before the cache on the production store:
  ~185 static pairs re-scored for ~31 minutes, nightly; steady state
  after: seconds, proportional to the day's new memories. Dry runs read
  the cache but never write it. Phase 6 drops rows referencing archived
  memories (cleanup_scan_cache, mirroring stale-link cleanup).

## [10.17.3] - 2026-07-03 (phase 4 remembers its verdicts)

### Fixed
- Phase 4 had no memory of past scans: COMPATIBLE verdicts left no trace,
  so the same pair re-entered the finder and judge every cycle (an
  eternal loop for every finder false positive once the queue tier
  ships), and already-linked pairs were re-scored and re-judged with the
  duplicate link hidden by INSERT OR IGNORE. COMPATIBLE now writes a
  `contradiction-cleared` tombstone link, and candidate selection skips
  pairs carrying any contradicts / superseded_by / evolves /
  contradiction-cleared / contradiction-candidate link before scoring.
  Clearance is permanent for the pair as stored; a later content update
  does not re-open it (known ceiling).

## [10.17.2] - 2026-07-03 (phase 4 scans the full active set)

### Fixed
- Phase 4 received `mergeable_embeddings`, but protected memories
  (decision-type, verified, importance >= 9) are excluded from the
  mergeable set and are exactly the population most worth
  contradiction-scanning; `load_embeddings` keeps them in
  `all_embeddings` for weave/contradict by design and weave was wired
  correctly, contradict never was. On stores whose facts are
  predominantly protected the phase permanently reported "not enough
  decision/fact memories" and the NLI finder never ran (production
  Epsilon: 1 fact visible of hundreds). Phase 4 now scans
  `all_embeddings`. Merge protection is unaffected: the phase links,
  queues and (with the existing blast-radius guard) archives, it never
  merges.

## [10.17.1] - 2026-07-03 (opt-out ONNX memory arena, all sessions)

### Added
- `MNEMOS_DISABLE_MEM_ARENA=1` disables the ONNX Runtime CPU memory arena
  on every session Mnemos creates: the e5 embedder, the Jina reranker
  (both via fastembed's exposed session options) and both NLI scorers
  (direct `SessionOptions`). The arena grows during active inference and
  never shrinks while a session stays loaded, so on busy constrained
  hosts RSS climbs past what the idle reaper can ever reclaim (reported:
  700MB to 4.8GB on a 7.3GB system). With the flag, each inference
  allocates from the system and returns it: bounded RSS for ~10-15%
  slower inference. Opt-in, default off, same contract as
  `MNEMOS_MODEL_IDLE_TTL` and `MNEMOS_MIN_FREE_MB`.
  Embedder/reranker portion contributed by the balaianu/Mnemos fork;
  extended here to the 10.16+ NLI ONNX sessions the fork predates.

## [10.17.0] - 2026-07-03 (zero-LLM daily consolidation cycle)

The daily Nyx cycle now runs with zero LLM calls: cosine nominates (by
rank), NLI decides (admission gate + line-level dedup), a mechanical union
executes merges (selection, never generation). LLM-requiring work (weave,
synthesize, contradiction judging, cemelify) is the optional enrichment
tier. Evidence: `benchmarks/weave-bench` (NLI weave classification refuted
at 3% agreement; the phase-2 gate validated on production noise clusters)
and `benchmarks/merge-bench` (mechanical merge: 24/25 exact recovery on
ground truth, 100% line coverage and digit integrity by construction).

### Added
- Mechanical merge engine (`mnemos/consolidation/mechanical.py`), the
  phase-2 default (`MNEMOS_MERGE_ENGINE=mechanical|llm`). Line union with
  bidirectional-entailment dedup at `MNEMOS_MECH_MERGE_TAU` (0.90; the one
  observed semantic false-duplicate scored 0.851, true duplicates ~1.0).
  Lines under `MNEMOS_MECH_MERGE_MIN_LINE_CHARS` (25) dedup by exact match
  only (short enumerated list lines are an NLI failure class). Newer
  phrasing wins; every output line is an input line verbatim, so fact
  preservation is provable rather than auditable.
- Phase-2 NLI admission gate (`MNEMOS_CLUSTER_GATE=nli|off`, tau
  `MNEMOS_CLUSTER_GATE_TAU` 0.70): cluster members must share at least one
  line-level bidirectional-entailment fact or they are ejected; clusters
  without shared facts dissolve unmerged. Replayed on the 2026-07-03
  production run: both cosine noise clusters dissolve entirely.
- Mutual top-k candidacy (`MNEMOS_CANDIDACY=mutual-topk|threshold`, k via
  `MNEMOS_CANDIDACY_TOP_K`): phase-2 pair candidacy from mutual
  nearest-neighbor rank instead of absolute cosine thresholds, which are
  noise in the compressed e5 space (measured: 45% of all active-pair
  similarities above 0.78).
- Phase-4 judge queue mode (`MNEMOS_CONTRADICT_JUDGE=llm|queue|auto`):
  keyless runs record NLI-flagged pairs as `contradiction-candidate`
  links; the next llm-judged run consumes the queue. `auto` resolves by
  key presence.
- Weave staleness guard: memories with outgoing `superseded_by`/`evolves`
  links are excluded as weave sources (stale state was being woven into
  fresh insights).
- Weave novelty gate (`MNEMOS_WEAVE_NOVELTY_TAU`, 0.85): a bridge insight
  entailed by either source alone is a restatement; the link is kept, the
  insight memory is not stored.
- Useful-loop: `get()` on a memory that retrieval logging recorded in the
  last 24h marks those `retrieval_log` rows `useful=1`. Zero-friction
  usefulness signal for measuring consolidation value.
- `nli.line_max_duplicate` and `nli.p_entailment` public scorers.

### Changed
- Bridge insights are stored on the episodic layer: derivative content
  earns permanence through retrieval instead of squatting in the semantic
  tier.
- Phase 2B topic merge is retired under the mechanical engine
  (aggregating distinct same-topic facts is generative LLM-tier work);
  the legacy llm engine keeps both tiers.
- A missing LLM key no longer fails the cycle (replaces the v10.4.0
  loud-fail): LLM-tier phases are skipped with a grep-able WARNING and
  the zero-LLM phases run. `MNEMOS_DISABLE_LLM=1` still opts into the
  same skip silently. Phase 0.5 cemelify additionally requires a key.

### Fixed
- `tests/test_v107_tier2.py` fixture wrote to the developer's default DB
  whenever another test module imported mnemos first (DEFAULT_DB_PATH is
  frozen at import time); 419 fixture memories had accumulated and the
  tier-2 recall assert started flaking on KNN ties. The store is now
  constructed explicitly.

## [10.16.2] - 2026-07-03 (CI test isolation)

### Fixed
- `test_is_available_true_with_onnx_models_and_no_torch` depended on transformers being installed in the test environment (true locally, false in CI); the ONNX runtime import probe now has its own seam (`_onnx_runtime_available`) and the test stubs it. No behavior change.

## [10.16.1] - 2026-07-03 (documentation)

### Changed
- Documentation sweep reflecting v10.15-10.16: README store-path diagram, ARCHITECTURE NLI-layer section and updated dedup/contradiction/phase-4 mechanics, features/philosophy/usage updates including the explicit design policy (prefer local discriminative scorers over LLM calls; the currency is RAM) and the NLI configuration reference.

## [10.16.0] - 2026-07-03 (ONNX backend for the NLI layer; self-healing temperature rejection)

### Added
- ONNX backend for the NLI decision layer, preferred over torch when a local export exists. Runtime is onnxruntime (already in the dependency tree via FastEmbed) plus the transformers tokenizer, no torch: the `mnemos[nli]` extra shrinks from a multi-GB torch pull to `onnxruntime + transformers + sentencepiece`. Models are exported once with `scripts/export_nli_onnx.py` (tooling extra: `mnemos[nli-export]`) into `MNEMOS_NLI_ONNX_DIR` (default `~/.cache/mnemos/nli-onnx/{en,multi}`), or copied between machines. `MNEMOS_NLI_BACKEND` pins `auto`/`onnx`/`torch`.
- Parity gate results (114 nli-bench pairs, both models): ONNX fp32 is score-identical to torch, max probability drift 1e-05, identical AUC to 4 decimals, zero threshold flips. int8 dynamic quantization was REJECTED by the same gate: it collapses DeBERTa-v3 to chance (contradiction AUC 0.94 -> 0.51 English, 0.84 -> 0.48 multilingual) and was not even reliably faster on CPU. The layer ships fp32-only; the torch scorer remains as fallback (`mnemos[nli-torch]`).
- `chat()` self-heals on temperature-rejecting models: a 400 naming `temperature` strips the parameter, retries immediately, and remembers the (endpoint, model) pair for the process lifetime. Nyx phases with hardcoded temperatures now work against such models with no configuration; `MNEMOS_LLM_OMIT_TEMPERATURE[_<PHASE>]` remains as an explicit override that skips even the first probe.

## [10.15.2] - 2026-07-02 (chat temperature=None omits the parameter)

### Fixed
- `consolidation.llm.chat()` now treats `temperature=None` as "do not send the parameter", the portable calling convention for model families that reject `temperature` outright (e.g. Sonnet 5 on the OpenAI-compat endpoint, which 400s the whole call). Previously omission was only possible deployment-wide via `MNEMOS_LLM_OMIT_TEMPERATURE[_<PHASE>]`; that env escape hatch still works and still wins when set.
- `scripts/translate_store_english.py` passes `temperature=None`, so the translation runbook no longer requires `MNEMOS_LLM_OMIT_TEMPERATURE_TRANSLATE=1` when translating with such models. Without the fix, every translation silently fell back to the original (chat() swallows the 400 and returns None), which read as "LLM configured but nothing happens".

## [10.15.1] - 2026-07-02 (settings centralization, English-primary migration runbook)

### Changed
- All remaining module-local tunables moved to `constants.py` as the single settings surface, each with an env override: Nyx phase-2 clustering (`MNEMOS_TIGHT_THRESHOLD`, `MNEMOS_TOPIC_THRESHOLD`), phase-3 weave (`MNEMOS_WEAVE_MIN_SIMILARITY`, `MNEMOS_WEAVE_TOP_K`), phase-5 packet size (`MNEMOS_NYX_PACKET_SIZE`), per-run LLM call budgets (`MNEMOS_NORMAL_MAX_CALLS`, `MNEMOS_SURGE_MAX_CALLS`, `MNEMOS_SURGE_THRESHOLD`), and ingest limits (`MNEMOS_INGEST_CHUNK_CHARS`, `MNEMOS_INGEST_DEFAULT_PROJECT`, `MNEMOS_INGEST_MAX_READ_BYTES`). No default values changed.

### Added
- `docs/english-primary.md`: the English-primary store convention and a migration runbook for existing non-English stores.
- `scripts/translate_store_english.py`: one-time store migration to English. Uses the NLI layer's own `is_english()` to select candidates, the configured consolidation LLM (`MNEMOS_LLM_MODEL_TRANSLATE` pins a model for the job), line-structure and per-line digit-integrity guards, dry-run mode, and package-API writes (content + vector + text hash in one transaction). Skips locked rows.

## [10.15.0] - 2026-07-02 (NLI decision layer: entailment-based dedup confirm, contradiction detection, Nyx phase-4 finder)

Replaces the cross-encoder reranker for the store DECISION questions (is this a duplicate? does this contradict?) with natural-language-inference models. A reranker scores topicality ("same topic?"); NLI scores polarity ("same claim? opposite claim?"), which is the question the store layer actually asks. Backed by a 114-pair benchmark on real production memories (benchmarks/nli-bench): contradiction AUC 0.939 vs 0.69 for the reranker (which produced ~40 false positives of 96 negatives at its best threshold); dedup AUC 0.983 with 1 false positive vs 16-21 false blocks for the raw vec-distance blocker. The reranker keeps its search-ranking role, where topicality is the right signal.

### Added
- `mnemos/nli.py`: NLI scoring layer. Language-agnostic routing: content that reads as English (cheap stopword heuristic, `is_english()`) uses an English ANLI+FEVER-hardened checkpoint (strongest benched); everything else uses a multilingual XNLI checkpoint (~100 languages). `p_contradiction()` takes the max over both premise/hypothesis directions (real contradictions score asymmetrically: 0.44 one way, 0.99 the other on the bench); `bidirectional_entailment()` takes the min (a duplicate entails in both directions); `line_max_contradiction()` scores the top-k cosine-preselected line pairs of two records, rescuing conflicts that blob-level scoring buries (benched: a diagnosis conflict scored 0.58 blob-level, 0.9956 line-level).
- `MNEMOS_DEDUP_CONFIRM=nli`: store-path dedup confirm tier. Bidirectional entailment >= `MNEMOS_NLI_DEDUP_THRESHOLD` (default 0.85) on the top `MNEMOS_NLI_DEDUP_MAX_CANDIDATES` (default 3) candidates by vector distance blocks the store; below it the store proceeds with no fall-through to the coarser scorers. Legacy behavior unchanged when unset or when the NLI runtime is unavailable.
- `MNEMOS_CONTRADICT_MODE=nli`: contradiction detection asks the NLI question directly after the vec gate. Warn + `contradicts` link only at max-direction P(contra) >= `MNEMOS_NLI_CONTRA_THRESHOLD` (default 0.98). No relates band: WEAVE owns topical linking.
- `MNEMOS_NYX_CONTRADICT_FINDER=nli`: phase-4 candidate finder. Drops the legacy cosine-band CEILING (near-identical pairs are where real contradictions live) and scores floor-gated pairs with the line-level finder, recall-first (`MNEMOS_NLI_FINDER_THRESHOLD`, default 0.8, capped at `MNEMOS_NLI_FINDER_MAX_PAIRS` pairs); the existing LLM judge keeps precision. The three benched real contradictions the blob/band approach missed are all caught by this path.
- Optional dependency extra `mnemos[nli]` (torch, transformers, sentencepiece). Every NLI entry point degrades gracefully when the extra is not installed.
- `tests/test_v1015_nli.py`: 19 tests (routing, direction aggregation, store integrations, phase-4 selection), model-free via stub scorers.

### Changed
- Phase-4 cosine gates `CONTRADICT_MIN_SIM`/`CONTRADICT_MAX_SIM` moved from `consolidation/phases.py` to `constants.py` (env-overridable); all new NLI tunables live in `constants.py` as the single settings surface.
- Phase-4 pair selection extracted into `select_contradict_candidates()` (pure, testable).
- `scripts/mnemos_sortkit.py` no longer defaults to a deployment-specific namespace.

## [10.14.0] - 2026-07-02 (external audit fixes: atomic content+vector writes, hybrid vec-only recall, embed_meta migration, exploder boundary tightening)

Response to an independent full-code audit of v10.13.0 (4 parallel reviewers plus a live DB health audit on a second production deployment, on Windows). Every finding was re-verified against source before fixing; the two partially-refuted ones (model provenance "never written", archive-move "pollutes forever") still carried real cores and are fixed too. Each fix ships with a regression test in `tests/test_v1014_audit_fixes.py` (43 tests).

### Fixed
- **Hybrid search discarded vector-only hits when FTS matched nothing.** The merge had no branch for `hybrid` with empty `fts_ids`: control fell through to the FTS-only else and returned the (empty) FTS list while the computed `vec_ids` were thrown away. Typical trigger is a cross-lingual query with zero token overlap, which the multilingual embedder handles fine, so the advertised cross-lingual recall was silently dead in the default mode. Confirmed live with JA/KO/EL queries returning fts=0, vec=10, hybrid=0. Now vec-only hits feed the rerank pool.
- **Content write and vector write were two separate transactions.** `store_memory` committed the row (FTS triggers fire in that same transaction) and only then wrote the vector with its own commit; `update_memory` likewise. A crash or exception between the commits left a keyword-findable but vector-invisible memory (store path) or new content with the old vector still attached (update path); this is the confirmed root cause of a stale-vector incident on the second deployment. `_store_embedding` now takes `commit=False` and joins the caller's transaction; both write paths open `BEGIN IMMEDIATE`, commit once after both writes, and roll back fully on failure. The up-front write lock also closes the check-then-act race on `UNIQUE(source_db, source_id)` between concurrent embedders of the same id.
- **`move_embedding_to_archive` committed the archive insert before the active delete.** A crash in the window left the vector in both `embed_vec` and `embed_vec_arch`, invisible to `archived_missing_embeddings` (which only looks for missing arch copies). Phase-6 `cleanup_orphan_vectors` would have healed it at the next cycle, but the window is now closed properly: one transaction, rollback on failure.
- **QdrantStore had the same split, plus no staleness tracking at all.** SQLite committed, then the network upsert ran with nothing to compensate. Store now hard-deletes the just-committed row when the upsert fails (SQLite and Qdrant cannot share a transaction, so compensate and re-raise), and `text_hash` rides in the Qdrant payload so staleness stays detectable on that backend too (it has no `embed_meta`).
- **`embed_meta` had no back-compat migration and `doctor` could not self-heal.** `CREATE TABLE IF NOT EXISTS` is a no-op on a pre-10.6 `embed_meta` (no `text_hash`/`model` columns) while `_store_embedding` writes `text_hash` unconditionally, so pointing Mnemos at an older DB threw `no such column: text_hash` on every store and update. Worse, `doctor --migrate` died inside `embed_status()` (which selects the missing column) before it could repair anything, and had no `embed_meta` migration anyway. `embed_meta`/`embed_meta_arch` now get the same silent column backfill the `memories` table has had since 10.3.4, and doctor's coverage check is guarded so drift surfaces as a reported issue instead of killing the health check.
- **The mechanical CML exploder false-split on the hot write path.** The statement boundary accepted any prefix letter not glued to an alphanumeric, so `F:free space on C: drive is low` shredded at `C:` (only `C:\` and `C:/` were excluded) and `(P:prefer HE-AAC v2)` tore at the parenthesis. The loss guard stripped `.` as a separator, so it was blind to these placements. A boundary now requires start-of-blob or a preceding `;`/`.` terminator, and periods count as content in the guard. The inlined copy in `scripts/split_single_line_cml.py` is synced. Both production DBs were scanned: no existing shreds, this was latent.
- **Size-split children inherited packed multi-statement lines.** The size splitter is deliberately line-preserving and children are stored with `_no_split` (the chain exploder is a no-op on multi-line text anyway), so a physical line carrying several `;`-chained statements survived remediation un-atomic; 8 such children were produced on the audit deployment. New `splitter.explode_cml_lines()` runs the loss-guarded exploder per line on every split child, in both `_store_split` and `remediate-oversized`.
- **`valid_from` was stored but never enforced, and `valid_until` was off by one day.** No `valid_only` filter checked `valid_from`, so a future-dated fact passed as currently valid; and Phase-4 EVOLVED sets `valid_until = today` with the stated intent of immediate exclusion, but the `>=` filters kept the memory valid until tomorrow. All four filter sites now also require `valid_from <= now` and treat `valid_until` as an exclusive expiry (`>`).
- **Phase-2 merge lineage never reached `nyx_insights`.** `get_merged_sources` (and thus `search(expand_merged=True)`) reads exclusively from `nyx_insights`, but `apply_merge` recorded provenance only in tags, so real merges produced super-memories with permanently empty `merged_from` and tier-2 recall fell back to similarity only. The merge transaction now writes the lineage row.
- **CONTRADICT could auto-archive a verified memory on a steered verdict.** The classifier's SUPERSEDED verdict is derived from memory content interpolated into the prompt, i.e. from data the project's own attribution rule treats as untrusted. Verified and importance>=9 memories now get the `superseded_by` link recorded without the archive (counted as `superseded_skipped`); ordinary memories behave as before.
- **Phase-3 weave bridges were stored active but never embedded.** FTS-only forever, permanently reported `missing` by embed-status, with no backfill path. Extracted `store_bridge_insight()` embeds at creation (best-effort: on embedder failure the bridge still lands and `embed-fill` catches it later).
- **Phase 5 could cite just-archived sources.** `mem_by_id` was reloaded after Phase-2 merges but not after Phase-4 supersedes, so synthesis packed archived memories as active. The orchestrator now reloads after Phase 4 when anything was superseded and Phase 5 is enabled.
- **Two decay-scoring queries interpolated stored column values into SQL.** `julianday('{m['last_accessed']}')` built SQL from a caller-settable field (whitelisted in `update_memory`, reachable via MCP). Parameterized; this closes the one exception to the codebase's otherwise fully parameterized SQL.
- **Every UPDATE re-tokenized the row into FTS, on every read.** The unguarded `AFTER UPDATE` trigger resynced FTS on any column change, and `get_memory` bumps `access_count` per read, so each read paid a full FTS delete+reinsert. The trigger now carries `WHEN new.content IS NOT old.content OR ...` and existing DBs with the unguarded trigger are upgraded in place at connect.
- **One malformed stdin line killed the MCP server loop.** `read_msg` now skips unparseable lines (logged to stderr) instead of raising out of `main()`; EOF still terminates.
- **`tools/call` errors leaked raw `str(e)`.** DB paths and schema fragments went back to the caller. The response now carries the exception class plus a 300-char-truncated message; the full traceback goes to stderr.
- **A caller-supplied catastrophic regex could hang the server forever.** `bulk_rewrite(use_regex=True)` compiled and ran the pattern over all content with no bound, `dry_run` included, on the single-threaded MCP loop. The scan now runs under a SIGALRM time limit (default 30s, `MNEMOS_BULK_REWRITE_TIMEOUT` to tune, no-op on Windows and off the main thread) and returns an error instead of hanging.
- **CLI crashed with `UnicodeEncodeError` on non-UTF-8 stdout.** Any CML glyph aborted whole commands on cp1252-encoded streams (Windows consoles, redirected output). stdout/stderr are reconfigured to UTF-8 with replacement at CLI entry; UTF-8 streams are untouched, and the MCP server was never affected (ASCII-escaped JSON).

### Added
- **`mnemos embed-fill`**: backfills vectors for active memories that have none (the rows embed-status reports as `missing`). `doctor` has recommended this command since 10.3.x without it existing; `reindex-archived` only ever covered the tier-2 index. `--dry-run` and `--limit` supported.
- **Store result warning on embed failure.** A transient embedder failure used to be indistinguishable from success (memory persisted FTS-only, `embedded: false` buried in the result). The result now carries an explicit `warning` pointing at `embed-fill`.
- **Vector model provenance.** `_store_embedding`, `_store_archived_embedding`, the archive moves, and the consolidation writer now all record which embedder produced each vector (the column existed since 10.6 but the primary path never wrote it), and `doctor` flags mixed provenance: a different-dims model fails loudly at insert, but a same-dims `MNEMOS_EMBED_MODEL` swap silently corrupts every KNN comparison, which nothing detected before.

### Known limitations (audited, deliberately not changed)
- Rerank-mode contradiction detection still conflates topical overlap with logical conflict (cross-encoder score >= 0.60 persists a `contradicts` link). The honest fix is an NLI model, not a threshold tweak; `MNEMOS_CONTRADICT_MODE=llm` already distinguishes. Documented here so the false-positive warnings on dense same-subject corpora are a known quantity.

## [10.13.0] - 2026-07-02 (per-phase LLM key + temperature omission: hybrid local/cloud consolidation)

Enables routing a single Nyx phase to a different provider than the rest, which the fidelity-critical MERGE phase needs: a strong cloud model that obeys one-fact-per-line and preserves facts under compression, while base/weave/contradict/triage stay on a cheap local pool. Motivated by a live finding that the local 30B over-compressed real clusters (45->16 / 85->19 line collapse) and mislabeled prefixes on merge.

### Added
- **Per-phase LLM API key** (`MNEMOS_LLM_API_KEY_<PHASE>`). `_get_config` already resolved per-phase model and URL overrides but not the key, so hybrid routing only worked within a single auth domain. Now MERGE can carry a real cloud token (e.g. an Anthropic `sk-ant-` key against `api.anthropic.com/v1/chat/completions`) that reaches only the cloud endpoint, while the global `MNEMOS_LLM_API_KEY` stays a throwaway for the local pool. The secret never touches the local router.
- **Per-phase temperature omission** (`MNEMOS_LLM_OMIT_TEMPERATURE[_<PHASE>]`). Some newer models reject `temperature` as deprecated and 400 the entire request (observed: Anthropic Sonnet 5 on the OpenAI-compat endpoint). When set, `chat()` drops `temperature` from the payload for that phase; the other phases keep it. Regression tests in `tests/test_llm_config.py`.

## [10.12.1] - 2026-07-02 (fixes: load_embeddings rowid crash + over-aggressive store dedup + single-line CML on merge/store)

### Fixed
- **MERGE and store persisted single-line prefix-chained CML.** `explode_cml_chain` (added 10.12.0) was wired into no runtime path, only the standalone repair script and tests. `apply_merge` and `store_memory` ran only the size-guard splitter, which triggers on length (> 4000 chars), not on format, so a short merged blob the local MERGE model chained with `;` (`D:cpu is ...; F:has 64gb ...`, ignoring the one-fact-per-line prompt) was stored single-line and unsplittable, re-introducing exactly what 10.11.0/10.12.0 set out to kill. Both write paths now run the mechanical exploder before the size guard, so single-line chains are normalized to one fact per line at write time no matter what the LLM emits. The prompt asks; the exploder enforces. Verified live: a two-cluster local Nyx merge that previously produced single-line `#8`/`#9` now yields 2- and 3-line memories. Regression test `tests/test_store_explode.py`.
- **Store-time dedup silently dropped distinct memories.** `Mnemos._dedup` fell back to a blanket `score = 0.75` whenever the cross-encoder reranker was disabled or unavailable, so any candidate with a coarse FTS/CML/vector match (the vec gate is a loose cosine ~0.82) was flagged a duplicate and the store was blocked, distinct or not: 11 of 16 unrelated memories were rejected in a fresh-install shakedown (`epsilon runs Ubuntu` vs `epsilon has NVMe drives` both killed). The fallback now derives confidence from the actual vector distance and blocks only within the strong-dup bar `VEC_DEDUP_MAX_DISTANCE`; FTS/CML-only matches no longer block without a real similarity score. `DEDUP_RERANK_THRESHOLD` also raised 0.70 -> 0.85 (0.70 over-blocked related-but-distinct memories the reranker scored 0.81-0.84). Bias: prefer false-store (Nyx merges later) over false-block (silent loss).
- **`consolidation/phases.py::load_embeddings` crashed on every fresh install.** It hardcoded `SELECT embedding FROM embed_vec WHERE id = ?`, but `sqlite_store` creates `embed_vec` as `vec0(embedding float[N])`, which is rowid-keyed with no `id` column, so the first `consolidate` raised `sqlite3.OperationalError: no such column: id`. It now uses the same `_vec_join_col(conn)` detection its sibling `store_embeddings` already used. Legacy DBs with an explicit `id` PK (long-lived stores from the v7/v8 era, e.g. the author's own) were never affected and stay unaffected. Found on the NUC while wiring the local Nyx cycle. Regression test `tests/test_load_embeddings_compat.py` exercises `load_embeddings` against a fresh rowid-schema DB, which legacy CI databases could not reach.

## [10.12.0] - 2026-07-01 (R: restriction prefix, cemelify one-fact-per-line, mechanical CML exploder)

Follow-on to 10.11.0: that release fixed the MERGE prompt to emit one fact per line, but the store-time and Phase 0.5 `cemelify` prompt still instructed "a single compact CML line", so it kept regenerating the unsplittable single-line blobs. This release fixes cemelify at the source, adds a mechanical (no-LLM) exploder for boxes where the MERGE path is unavailable, and adds a distinct `R:` restriction prefix.

### Added
- **`R:` (Restriction) CML prefix** for hard rules and limits, kept distinct from `W:` (Warning, a caution flag); a rule is not a warning. Wired into every prefix-set site in one pass: the cemelify prompt, the MERGE prompt, the `memory_store` MCP tool description, `core.CML_TYPE_PREFIXES`, `cemelify._needs_cemelify`, the splitter statement-boundary regex, and the CML docs (`docs/cml.md`, `docs/agent-instructions.md`).
- **`splitter.explode_cml_chain()`** reformats one physical line of prefix-chained CML into one-fact-per-line CML, stdlib only (no LLM, no DB). It splits before each canonical prefix that starts a new statement; a `;` not followed by a prefix stays intra-fact, so a single fact is never shredded, and a prefix followed by a path separator (a Windows drive letter `C:\` or a `D:/` URL) is not a boundary, so file paths mid-text are not false-split. Loss-guarded: returns the input unchanged unless the separator-free content is preserved exactly. For repairing legacy single-line memories on deployments where the LLM MERGE path is disabled.
- **`scripts/split_single_line_cml.py`** applies `explode_cml_chain` across a whole memory DB: scans for single-line multi-statement CML memories, dry-runs by default, and on `--apply` rewrites each through the Mnemos update API so FTS and vector re-sync. For repairing existing memories on machines where the LLM MERGE path is unavailable.

### Changed
- **cemelify emits one fact per line** instead of a single compact CML line. This is the sibling of the 10.11.0 MERGE fix: with MERGE corrected but cemelify still packing one line, the store-time hook and Phase 0.5 kept producing unsplittable single-line blobs.

### Fixed
- **cemelify `C:` legend corrected to Contact.** The cemelify prompt uniquely defined `C:` as "Constraint or Caveat" while every other site (MCP tool description, MERGE prompt, docs, `core.CML_TYPE_PREFIXES`) defines `C:` as Contact. Verified against the live corpus before changing: 48 of 49 existing `C:` statements are contacts, so this aligns the outlier with no retroactive reinterpretation. Constraints now have their own `R:` prefix.

## [10.11.0] - 2026-07-01 (One-fact-per-line merge, idempotent cemelify, decision merge-protection)

Consolidation overhaul from a live session moving the Nyx cycle onto local inference. The single-line CML format was found to be the root cause of unsplittable oversized memories and of fact loss during merge.

### Changed
- **MERGE emits one fact per line** instead of dense single-line packing. The old prompt ("chain facts densely on one line when topic is shared, pack more facts per line") produced single-line blobs that the line-based splitter could not cut, so same-topic merges became unsplittable oversized memories. One fact per line is splittable and atomic-ready (child extraction becomes a newline split, not an LLM pass), and it preserves MORE: bench went 93.6 -> 95.2% unique-fact preservation, because dense packing was itself dropping facts.
- **Phase 0.5 Cemelify is idempotent.** `_needs_cemelify` no longer re-triggers on already-CML content over 800 chars; a memory whose first line carries a CML prefix is left as-is. Re-rewriting already-CML memories on a weaker local model corrupts exact strings (observed: a benchmark score `56/56` rewritten to `56/64`) for zero normalization gain.
- **Decisions are excluded from Phase 2 merge.** `type='decision'` memories join evergreen, `importance >= SKIP_IMPORTANCE`, and `consolidation_lock` in the merge skip set. They are still woven and contradiction-scanned (they remain in `all_embeddings`), but are never blended and archived, since merging compresses authoritative records lossily.

### Added
- **`MNEMOS_NYX_CEMELIFY` env flag** (default `1`). Set `0` to skip the Phase 0.5 cemelify pass entirely, for corpora whose non-CML population is document-shaped content that should not be compressed to a single line.

## [10.10.1] - 2026-06-28 (Audit hardening: busy_timeout, atomic backup, richer doctor)

Post-v10.10.0 read-through audit fixes, same data-safety theme.

### Fixed
- **`PRAGMA busy_timeout=5000`** on every connection. The MCP server, the CLI, and the Nyx consolidation run are separate processes against one DB; WAL handles reader/writer but concurrent write-vs-write previously raised SQLITE_BUSY immediately instead of waiting. Writes now wait out contention.
- **`backup()` is now atomic.** It VACUUMs INTO a temp sibling then `os.replace()`s it into place, so a failed snapshot (disk full, I/O error) can no longer destroy an existing prior backup at the destination. It also resolves the destination to an absolute path and creates a missing parent directory.
- **`doctor` reports the full quick_check result**, not just the first line: on corruption it shows the first few problems plus a count instead of hiding the extent behind `fetchone()`.
- **Search surfaces a corruption hint.** A raw SQLite "database disk image is malformed" at search time is re-raised with guidance to run `mnemos doctor` and restore the latest `mnemos backup`, instead of an opaque error (the exact failure mode from 2026-06-27).

## [10.10.0] - 2026-06-27 (WAL-safe backup + doctor integrity check)

A live, WAL-mode `memory.db` corrupted in prod during the v10.8/10.9 split-backlog work: `btreeInitPage` error 11, rowids out of order, concentrated in the newest pages. Root cause was NOT the split logic (it is lossless and all writes go through SQLite) but unsafe file-level handling of a live WAL DB: the file was copied/restored without checkpointing its `-wal`/`-shm` (doctor used `shutil.copy2`, compounded by an operator `cp` cascade), so a restored snapshot replayed mismatched WAL frames and tore the btree. Worse, doctor never ran an integrity check, so it reported "healthy" on a malformed DB and the damage only surfaced as a "database disk image is malformed" blow-up at search time.

### Added
- **`SQLiteStore.backup(dest)`** and **`mnemos backup <dest>`**: WAL-safe hot backup via `VACUUM INTO`. Captures the full committed state (including rows still resident in an un-checkpointed WAL) into a single defragmented standalone file that needs no `-wal`/`-shm` sidecar, and is safe to run while the DB is live. Use this instead of `cp` on a live DB.
- `doctor()` runs `PRAGMA quick_check` first, before any read that assumes a sane btree, and reports page/btree corruption under issues. This corruption class is now caught at `mnemos doctor` time instead of surfacing as a malformed-image error at search time.

### Fixed
- `doctor()`'s pre-migration backup no longer uses `shutil.copy2` (an unsafe raw copy of a live WAL `.db`); it uses the new `VACUUM INTO` snapshot, so the safety backup can never itself become the corruption vector.

## [10.9.2] - 2026-06-27 (Sentence-split fallback, embedder-aligned target, cascade fix)

### Added
- Hard-mode sentence splitting (`split_content(..., hard=True)`, `mnemos remediate-oversized --hard`): a single line that exceeds target (the one thing the line splitter cannot break) is split on sentence/clause boundaries as a last resort, sentence-level lossless (verified by `split_preserves_all_sentences`). Atomizes structured single-line blobs that line-splitting leaves whole.

### Changed
- Default `MNEMOS_SPLIT_TARGET` 2800 -> 2400, aligned to the e5-large embedder window (~512 tokens, ~2000-2500 chars). Content past the window is truncated out of the embedding vector, so a larger target silently hurts vector recall.

### Fixed
- Re-split cascade: when a memory that was itself a split child got re-split, the new children inherited the parent's `split-from:#grandparent` tag, and single-match done-detection marked the grandparent (never the parent) as processed, causing infinite re-splitting and duplicate memories. Done-detection now uses `findall` (all ancestors), and child tags strip inherited `split-from`/`split-part` so each child carries exactly one parent marker.

## [10.9.1] - 2026-06-27 (remediate-oversized: --include-archived)

### Added
- `mnemos remediate-oversized --include-archived`: extends the flat backfill to archived (tier-2) memories. Archived originals are kept as lineage anchors; their split children are also archived and embedded into the tier-2 index, never promoted into active search. Used to atomize the archived oversized backlog.

## [10.9.0] - 2026-06-26 (Topic-sort: oversized memories into coherent atomic sub-memories)

The flat size-guard splits a blob into in-order pages. For a sprawling merged catch-all (e.g. a 302k personal memory mixing eight unrelated subjects), pages still mix topics and embed muddily. v10.9.0 adds topic-aware splitting: a router (an LLM) assigns each CML block to a topic, and the same lossless mechanical placement groups them so each resulting memory is about one thing, which embeds to a sharp, retrievable vector.

### Added
- **`topic_sort(content, propose_fn)`** in `mnemos/splitter.py`: groups blocks by router-assigned topic, sub-splits oversized topics, and gates on `split_preserves_all_lines` (multiset losslessness, since topic-sorting reorders). The router only routes; content is never rewritten. Falls back to flat `split_content` if the routing is unavailable or not a perfect cover.
- **`split_preserves_all_lines`**: order-independent lossless check for reordered splits.
- **`scripts/mnemos_sortkit.py`** (`dump` / `place` / `apply`): one-off kit to topic-sort oversized memories. The router (Opus) supplies a block-to-topic grouping; the kit places verbatim, writes a temp result, and `apply` stores each topic as an atomic child with a hierarchical subcategory path (e.g. `personal/janne-dementia`), sibling-linked, archiving the original.
- Topic-sort tests plus a trailing-whitespace regression in `tests/test_splitter.py`.

### Fixed
- `topic_sort` no longer `.strip()`s a topic's joined text, which could alter a content line carrying trailing whitespace at a topic tail and silently force the flat fallback.

### Migration applied (Epsilon prod)
- The 4 active giant memories (302k/82k/82k/66k) topic-sorted into 248 atomic sub-memories across coherent subcategory paths, all lossless, all embedded, originals archived.

## [10.8.0] - 2026-06-26 (Size-guard splitter: atomic memories)

Memories could grow without bound. A handful had ballooned to tens or hundreds of thousands of characters (worst case 302k), which both pollutes an agent's context when loaded and embeds to a blurry averaged vector that retrieval can barely rank. There was no size limit anywhere, and the merge prompt even said "do not truncate".

### Added
- **Lossless size-guard splitter** (`mnemos/splitter.py`). Pure mechanical, no LLM, stdlib only: packs whole CML blocks and lines into chunks of at most `MNEMOS_SPLIT_TARGET` (default 2800) characters, never breaking inside a line, so every non-blank fact line lands in exactly one chunk in original order. `split_is_lossless()` verifies the invariant.
- **Store-path guard.** `core.store_memory` splits content over `MNEMOS_SPLIT_THRESHOLD` (default 4000) into atomic sibling memories, each embedded and FTS-indexed, chained with 'related' links. Skipped for `consolidation_lock`. An internal `_no_split` flag prevents recursion on an un-splittable single line.
- **Consolidation guard.** The Phase 2 merge site (`apply_merge`) never emits an oversized merged memory: it splits losslessly into atomic siblings inside the same transaction and returns the primary id, so the lineage contract is unchanged.
- **`mnemos remediate-oversized`** (`--min-size`, `--max-size`, `--limit`, `--dry-run`): backfill that splits existing oversized active memories into atomic siblings, archives the original (vector moved to the tier-2 index), and re-points links onto the first child. Reuses the same splitter as the live path.
- `tests/test_splitter.py`, `tests/test_v108_split.py` (lossless property, size bound, consolidation_lock skip, store-path split, remediation backfill, dry-run).

### Config
- `MNEMOS_SPLIT_THRESHOLD` (default 4000), `MNEMOS_SPLIT_TARGET` (default 2800), `MNEMOS_SPLIT_ENABLED` (default on).

## [10.7.0] - 2026-06-14 (Tier-2 archived recall: keep merged-away vectors)

Consolidation used to delete the embedding of every memory it archived, so the only path to a merged-away original was `expand_merged`, which joins a found consolidated memory to its sources by lineage. That join can only surface originals whose consolidated parent already ranked in primary search. If consolidation summarized away a detail, a query matching that detail would not rank the parent, and the original became unrecallable by vector search. There was no independent vector path to archived content.

### Added
- **Tier-2 archived vector index.** Archived memories now keep their vectors in a separate index (`embed_vec_arch` / `embed_meta_arch`) instead of being deleted. It is deliberately separate from `embed_vec`: primary KNN over-fetches `k = limit*3` and post-filters status, so mixing the archived bulk (typically most of the corpus) into the primary index would crowd out active hits before the filter ran. A separate index keeps primary search active-only with zero regression.
- **`search_vec_archived()`** on the SQLite store: KNN over the archived index, always `status='archived'`, with the same project/subcategory/layer/type/valid filters as primary vec search.
- **`expand_merged` now runs a real tier-2 vector pass.** Alongside the lineage join, it KNN-searches the archived index with the query embedding and returns matching originals under a new `tier2_recall` key, deduped against primary results and `merged_from`. Archived originals are reachable even when their consolidated parent did not rank.
- **`reindex_archived()`** (CLI `mnemos reindex-archived`): backfill the archived index by embedding every archived memory that lacks an archived-index vector. Idempotent.
- `tests/test_v107_tier2.py`.

### Changed
- **Consolidation moves vectors instead of deleting them.** `cleanup_orphan_vectors` now moves an archived memory's vector from the active index into the archived index, and only deletes a vector outright when the memory row no longer exists at all (a hard delete). Primary search is unaffected: it still queries `embed_vec` (active only).

## [10.6.0] - 2026-06-10 (Namespace integrity + stale-vector detection)

Fixes from a fresh-eyes review of the whole engine. Two of these were silent-divergence bugs in production; the worst one was eating consolidated memories.

### Fixed
- **Nyx consolidation lost memories across the namespace boundary.** All three memory-creating sites in the consolidation phases (merge super-memories, weave bridge insights, synthesis insights) inserted without a `namespace` column, so every Nyx output landed in `default` regardless of `MNEMOS_NAMESPACE`. On a namespaced deployment the cycle archived visible source memories and replaced them with rows that no namespace-filtered search could ever return: silent memory attrition, one consolidation run at a time. The phases now resolve the active namespace exactly like the MCP server and CLI do. (On the production store this had orphaned 231 of 323 active memories, including every consolidated insight since the package migration; repaired by a one-time namespace update.) Note: cluster *selection* still operates store-wide; multi-tenant stores should run one Nyx pass per namespace until selection is namespace-scoped.
- **Stale vectors were undetectable.** `embed_meta.text_hash` existed in the schema and `embed.text_hash()` existed in code, but nothing ever wrote or read them. A content update whose re-embedding failed (model cold-start, OOM) kept the old vector with no record that it no longer matched the text, and `embed_status()` only counted missing embeddings. Now: `store_memory`/`update_memory` thread the canonical embed-text hash into `_store_embedding`; `embed_status()` reports `stale` (hash mismatch) and `unverified` (no recorded hash) alongside `missing`. Hash comparison is prefix-aware so 16-char truncated hashes written by older external tooling verify without forcing a re-embed.
- **`update()` hid re-embed failure.** `store_memory` reported `"embedded": bool`; `update()` reported nothing. It now returns `"embedded"` whenever re-embedding was attempted, plus a warning when the vector is left stale.
- **Hard delete orphaned the link graph.** `delete_memory(hard=True)` removed the memory and its embedding but left `memory_links` rows pointing at the dead id forever. Links are now pruned in the same operation.
- **Linked expansion resurfaced archived content.** Neither `get_links` nor the `include_linked` BFS filtered by status, so archived memories' content appeared in `linked_memories` summaries of active results. Summaries now skip non-active memories.
- **A silent reranker failure mass-wrote spurious links.** `rerank()` degrades by returning documents unscored; `_detect_contradictions` then scored every candidate at sigmoid(0) = 0.5, which lands inside the `relates` band (0.35..0.60), writing a `relates` link for every vec-gated candidate on any reranker hiccup. The pipeline now bails when no document carries a rerank score.

### Added
- `tests/test_v106_features.py` (10 tests; 94 total).

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

## [10.5.2] - 2026-06-09 (Tool-usage logging survives legacy schemas)

### Fixed
- `log_tool_usage` relied on the `called_at` column default, which does not exist in `tool_usage` tables created by pre-package deployments (`called_at TEXT NOT NULL`, no default). On such stores every insert failed the NOT NULL constraint and was silently swallowed by the diagnostics-only except guard, so `MNEMOS_TOOL_USAGE_LOG=1` produced no rows at all. The insert now supplies `called_at` explicitly, working on both legacy and package-created schemas. Found on a production store where tool-usage telemetry had been dark since the migration to the packaged MCP server.

### Fixed
- Phase 6 bookkeeping (`decay_access_counts`, `cleanup_stale_links`, `cleanup_orphan_vectors`) committed unconditionally, so a dry run (`execute=False`) silently decayed access counts and demoted importance on the live store while `log_consolidation_run` (correctly `execute`-gated) recorded nothing. The result was a store that had been mutated but a run that was never logged, so "Last run: never" persisted and every subsequent run re-triaged from scratch. All three Phase 6 mutators now take an `execute` flag (default `True` for backward compatibility) and only write when it is set; a dry run computes and reports the would-be counts without touching the store. Real runs (`execute=True`, including the weekly cron and SQL-only no-LLM runs) are unchanged and continue to log. Adds `tests/test_v105_features.py` (5 tests).

## [10.5.0] - 2026-06-03 (Resource-aware models + standalone SQL-only triage)

Changes aimed at running Mnemos well on small or shared hosts. All new
behaviour is opt-in and defaults to the previous behaviour, so existing
deployments are unaffected unless they set the new variables.

### Added
- **Optional idle model unloading.** `MNEMOS_MODEL_IDLE_TTL` (seconds, default
  `0` = never) lets a background reaper drop the embedder and reranker after
  they sit idle, returning their RSS to the OS (dropping the model frees the
  ONNX session and its arena; `malloc_trim` reclaims the glibc residue). The
  next query pays a one-off reload. Stops a long-lived server from pinning the
  models in RAM while idle on a constrained box.
- **Lazy model warmup.** `MNEMOS_EAGER_WARMUP=0` loads models on first use
  instead of at startup. Default `1` keeps the warm-at-startup behaviour.
- **Memory-pressure guard.** `MNEMOS_MIN_FREE_MB` (default `0` = off) refuses
  to load a model when available memory is below the floor, so the search path
  degrades gracefully (vec-only, then FTS5) instead of risking an OOM on a
  memory-tight host.
- `access_decayed` and `importance_demoted` columns on `consolidation_log`, so
  bookkeeping-only runs record their decay and demotion counts in the main
  audit columns rather than only inside `phase_details`. Existing databases are
  migrated automatically on the next cycle.

### Changed
- **Phase 1 (Triage) now runs standalone.** Triage is pure SQL and no longer
  sits behind the LLM-phase block, so SQL-only deployments
  (`MNEMOS_DISABLE_LLM=1`) get new-memory detection and surge sensing, not just
  Phase 6 bookkeeping.

---

## [10.4.4] - 2026-05-30 (LLM wall-clock budget + per-call timeout override)

Robustness patch for the consolidation LLM client. Caps total wall-clock
per `chat()` call across retries and lets fast paths request a tighter
read timeout. No behavior change on healthy calls.

### Fixed

- **`consolidation/llm.py` adds `MNEMOS_LLM_WALL_BUDGET` (default 480s)
  ceiling on total time spent in a single `chat()` call across all
  retries.** Without it, three 240s read timeouts plus their backoffs
  could burn ~726s on one hung call. On 2026-05-27 this sank
  `memory-dream-midweek.service`: a single LLM call ate ~12min of budget
  via the 3-retry-on-timeout path, the cemelify loop then ran 55min over
  93 candidates, and systemd's `TimeoutStartSec=3600` killed Phase 2A
  mid-merge of Cluster 2. The new budget gives the retry path one full
  retry-with-backoff cycle and then exits, returning `None` so the phase
  fallback continues. Env-tunable per provider.
- **`chat()` and the aliases `haiku_chat` / `sonnet_chat` / `opus_chat`
  accept a `timeout=` kwarg** that overrides the global `LLM_TIMEOUT`
  for a single call. Lets fast/small paths cap themselves without
  globally tightening, which would hurt hierarchical-merge prompts that
  legitimately need the larger window.
- **`cemelify.py` passes `timeout=90`** to its `chat()` call. Phase 0.5
  cemelify items are small (single memory rewrite, ~512 token output)
  and should never need the consolidation-class 240s window; 90s is
  generous for a healthy call and bounds the cost of a hung one. Worst
  case per call drops from ~726s to ~280s, and the wall-budget caps
  that further.

### Operational (companion change on the deployer side, not in this repo)

On Epsilon prod, `memory-dream-midweek.service` and
`memory-consolidate-nightly.service` had `TimeoutStartSec` raised from
3600s to 7200s. Belt-and-suspenders so a worst-case slow run is not
killed mid-phase while the code-level budgets above prevent a single
bad call from dominating.

---

## [10.4.3] - 2026-05-18 (LLM read-timeout hardening)

Robustness patch for the consolidation LLM client. No behavior change on
healthy calls; prevents a known degradation mode under slow providers.

### Fixed

- **`consolidation/llm.py` per-call read timeout raised 60s → 240s,
  env-tunable via `MNEMOS_LLM_TIMEOUT`.** The hardcoded 60s was too tight
  for reasoning-class models (e.g. gpt-5-mini) on large hierarchical-merge
  prompts: a slow-but-completing call timed out across all retries,
  `chat()` returned `None`, and `phases.py` fell back to raw concatenation
  of the unmerged pair, inflating merged memories and dropping merge
  quality without changing the model. Observed in production 2026-05-11
  (DO Gradient latency) and reproduced while evaluating gpt-5-mini for the
  MERGE phase. The retry/backoff logic was already sound; only the timeout
  ceiling was the gap. Operators can tune per provider without a code
  change.

---

## [10.4.2] - 2026-04-18 (CLI honors MNEMOS_NAMESPACE)

Tiny patch fixing a CLI / MCP-server divergence in env handling.

### Fixed

- **`mnemos` CLI now reads `MNEMOS_NAMESPACE` from the environment.**
  The MCP server has always honored `MNEMOS_NAMESPACE` (multi-tenant
  isolation key for the SQLite store), but `cli.py:main()` constructed
  `Mnemos()` with no arguments, silently falling back to
  `DEFAULT_NAMESPACE = "default"`. Result: on a database where memories
  live under a non-default namespace, `mnemos stats` reported 0 / wrong
  totals, `mnemos search` returned no hits, and the CLI was effectively
  invisible to the same data the MCP server was serving. CLI now
  matches the MCP server pattern: `os.environ.get("MNEMOS_NAMESPACE",
  DEFAULT_NAMESPACE)` at startup.

---

## [10.4.1] - 2026-04-18 (flush-on-print in consolidation phases)

Tiny patch release fixing a long-standing observability bug.

### Fixed

- **`print()` calls in `mnemos/consolidation/phases.py` now pass
  `flush=True`.** Eleven cluster/pair logging lines were buffered when
  stdout was a pipe (cron mail, `tee`, monitor stream), making long
  Nyx runs look hung for minutes at a time even though the cycle was
  making steady progress between flushes. The orchestrator's `log()`
  helper has always flushed; the per-phase progress prints did not.
  Discovered when a Phase 2A run on a 700-memory surge appeared to
  stall after "Found 30 tight clusters" but was in fact merging
  silently. PYTHONUNBUFFERED=1 worked around it externally; this is
  the in-package fix.

---

## [10.4.0] - 2026-04-18 (cemelify-on-import, loud-fail, OpenAI default)

Additive feature release plus one intentional behavior change around LLM
configuration. Three new env-var knobs, one new consolidation phase, and a
default model preset for the OpenAI endpoint.

### Added

- **Phase 0.5 (Cemelify) in the Nyx cycle.** New phase between Triage (1)
  and Dedup (2) that scans active memories which either don't start with
  a CML prefix (`F:`/`D:`/`C:`/`L:`/`P:`/`W:`) or are longer than 800
  chars, and rewrites each via `cemelify()`. Skips memories with
  `consolidation_lock=1` (prose-protection convention, matches Phase 2
  semantics). Runs whenever the LLM-dependent block runs, so it inherits
  the new loud-fail behavior below. Logs progress every 100 memories.
  Updates persist via `store.update_memory`; re-embed happens on the next
  bookkeeping pass (deferred by design to keep the phase cheap).
- **`MNEMOS_CEMELIFY_ON_IMPORT=1`** (opt-in env): when set, `store_memory()`
  pipes raw content through `cemelify()` before persistence, so memories
  land already in CML form. Falls back silently to the raw content on any
  LLM failure: this flag never turns LLM into a hard dependency. Skipped
  when the caller sets `consolidation_lock=True`.
- **`mnemos/cemelify.py`** new module exposing `cemelify(content)` -
  a single-entry helper that routes through `consolidation.llm.chat()`,
  inheriting all env routing (API URL, key, model, per-phase overrides,
  the new OpenAI default below).
- **Default model preset for the OpenAI endpoint.** If
  `MNEMOS_LLM_API_URL` points at `api.openai.com` (the default) and
  `MNEMOS_LLM_MODEL` is unset, Mnemos now defaults the model to
  `gpt-4o-mini` (recommended per the consolidation-quality bench in
  `docs/benchmarks.md`, 91.8-97.3% unique-fact preservation at $0.05/run).
  **No default for non-OpenAI endpoints**, since provider-specific model
  naming is too heterogeneous to guess. With this default, an OpenAI user
  only needs to set `MNEMOS_LLM_API_KEY` to be fully configured.

### Changed

- **`mnemos consolidate` now loud-fails when LLM is required but
  unconfigured** (intentional behavior break). Previously the cycle
  logged a warning and silently skipped LLM phases while running Phase 6
  bookkeeping; users reported assuming the full Nyx had run. Now
  `run_nyx_cycle` raises `RuntimeError` and the CLI exits with code 2
  (one-line stderr message, no traceback). Set `MNEMOS_DISABLE_LLM=1` to
  restore the previous silent SQL-only behavior explicitly. This is the
  only backward-incompatible change in v10.4.0; all other additions are
  strictly opt-in.

---

## [10.3.10] - 2026-04-16 (second-pass audit fixes)

Four more real bugs from a deeper second audit pass. All low severity
but all reproducible with concrete triggers.

### Fixed

- **`_vec_fallback_snippet` now guards against non-positive `chars`.**
  Python's `content[:negative]` truncates from the end of the string
  rather than returning empty, producing nonsense output when the
  snippet budget is 0 or negative. The MCP tool schema caps
  `snippet_chars` at 50..2000 but direct Python callers can pass any
  int. Explicit early return at function entry.

- **Atomic marker file for once-per-session UserPromptSubmit guard**
  (`scripts/mnemos-session-hook.sh`, also mirrored in the Epsilon
  reference hook). Previous test-then-touch pattern had a TOCTOU
  window: two concurrent invocations sharing a `CLAUDE_SESSION_ID`
  could both observe "marker absent" and both run priming. `mkdir`
  succeeds for exactly one caller even under race.

- **NUL bytes stripped from content and tags on store.** SQLite tolerates
  them fine, but downstream consumers (jq in shell hooks, strict JSON
  parsers, some display layers) truncate or reject at NUL. Silent
  data loss in recipients is the real risk. Strip on `store_memory()`
  entry so the value reaching SQLite is NUL-free.

- **`linked_depth` clamped to [1, 3] at `search()` entry.** Negative or
  zero `linked_depth` made the BFS guard `if dist >= linked_depth`
  trivially true after the root, silently disabling all link expansion.
  MCP tool schema caps at 1..3; Python callers now get the same
  protection.

### Not fixed (audit false positives, documented for future-me)

- `bulk_rewrite` with `max_affected=1` and all-no-op replacements
  reports `affected=0` - technically correct, semantics are clear.
- `doctor(migrate=True)` with backup failure - correctly aborts and
  reports the error. No silent data loss.

---

## [10.3.9] - 2026-04-16 (bug audit fixes)

Three real bugs from a post-ship audit of v10.1.0–v10.3.8. None of them
were blocking production, but all three were real and would have bitten
eventually.

### Fixed

- **`bulk_rewrite(tags='foo')` no longer leaks across tag boundaries.**
  Previous `tags LIKE '%foo%'` matched memories with tags like
  `unnamed,other` when the caller asked for `name`. Now uses
  `(',' || tags || ',') LIKE '%,foo,%'` for word-boundary match.
  Could have silently rewritten the wrong memories in production if
  a user used the tags filter with a common substring. Highest-severity
  of the three.

- **`_vec_fallback_snippet` returns `""` on empty/whitespace content**
  instead of the cosmetic fragment `" …"`. Triggered when a memory
  with whitespace-only content hit the vec-only snippet path.
  Cosmetic, but now honest.

- **LLM classification path handles whitespace-only responses.** The
  parsing logic `response.strip().lower().split()[0]` would IndexError
  on `""`, `"   "`, or `"\n"` responses. Caught by the outer try/except
  so no user-visible crash, but silently degraded to rerank heuristic
  without telling anyone. Now explicitly checks for empty token list
  and empty word, falling through cleanly when the LLM returns garbage.

### Not changed (false positives from the audit)

Two items flagged by the audit were not actual bugs:
- Multi-hop BFS was accused of exceeding the depth cap. Traced:
  `if dist >= linked_depth: continue` fires before expansion, so
  grandchildren at max depth are collected (correct - depth is
  inclusive) but great-grandchildren are never added. Invariant holds.
- "Malformed LLM classification bypasses validation" - same code path
  as the LLM empty-response bug; the existing
  `if word in self._CONTRADICTION_CLASSES:` check catches `"---"` and
  similar. Folded into the LLM fix above.

---

## [10.3.8] - 2026-04-16 (SQL-based tag aggregation for scale)

### Changed

- **`list_tags` now uses a SQLite recursive CTE** to split the tags CSV
  server-side instead of fetching all rows and aggregating in Python.
  Scales better for large deployments (no O(N) fetch of full tag rows
  into Python memory). At modest sizes (< 5K memories) the difference
  is negligible either way. Python-side fallback preserved for safety
  (triggered only if the CTE hits SQLite's recursion depth limit, which
  shouldn't happen on realistic tag strings).

### Parity

CTE and Python paths verified to return identical results (same tags,
same counts, same example IDs). API signature unchanged:
```python
mnemos.list_tags(project=None, min_count=1, order_by='count', limit=500)
# Returns: [{"tag": str, "count": int, "example_id": int}, ...]
```

### Implementation detail

The recursive CTE walks each memory's `tags` column, emitting one row
per tag by carving off the next substring up to the first comma:
```sql
WITH RECURSIVE split(mid, tag, rest) AS (
  SELECT m.id, '', m.tags || ',' FROM memories m WHERE ...
  UNION ALL
  SELECT mid,
         substr(rest, 1, instr(rest, ',') - 1),
         substr(rest, instr(rest, ',') + 1)
  FROM split WHERE rest != ''
)
SELECT TRIM(tag), COUNT(*), MIN(mid) FROM split ... GROUP BY TRIM(tag)
```

Runs entirely in SQLite. No Python-side string operations on the hot
path.

---

## [10.3.7] - 2026-04-16 (smarter vec-only snippet fallback)

### Changed

- **Vec-only snippet fallback now does sentence-scored picking** instead
  of a blind head slice. When a search hit matched via vec similarity but
  not FTS (so FTS5's `snippet()` returned nothing), the fallback splits
  content into sentences on `. ! ?` and picks the sentence with the
  highest substantive-word overlap with the query (stopwords dropped,
  min token length 3). If no sentence has any matching tokens, falls
  back to head slice as before. Cheap - no extra embedding calls.
- Keeps the existing semantics where exact-token matches still go through
  FTS `snippet()` for BM25-ranked extraction.

### Why this matters

Vec-only hits are by definition the case where FTS didn't match. The old
head-slice fallback returned the FIRST chars of the content regardless of
where the semantic match lived. For long consolidated memories with
multiple sentences on different sub-topics, that was often the least
relevant part of the content. The sentence-pick beats head slice whenever
the query has even one substantive word that appears in the content.

### Limits

- Exact token match only. Lemma-insensitive ("consolidation" does not
  match "consolidates"). Stemming would require adding a dependency;
  not worth it for this fallback path.
- Sentence splitter is regex-based (`(?<=[.!?])\s+`), correct for most
  CML and prose but will split mid-sentence on abbreviations like
  "U.S." or "e.g.". Good enough for this use case.

---

## [10.3.6] - 2026-04-16 (multi-hop `include_linked` with cycle detection)

### Changed

- **`memory_search(include_linked=true, linked_depth=N)`** now does real
  BFS graph traversal up to N hops (default 1, max 3 via MCP tool). Each
  linked memory summary carries a `distance` field (hops from the root
  result) and, for depth>1, a `via` field naming the intermediate node
  that reached it. Cycle detection via visited set prevents infinite
  loops on circular link graphs. Per-result cap of 30 total linked nodes
  to keep response sizes bounded even on well-linked graphs.

### Why this matters

v10.1.0 documented `linked_depth=1` as the only supported depth. For
single-hop relationship inspection that's fine, but graph-aware callers
(e.g., "show me this memory and anything transitively connected within
3 hops") had to do BFS client-side via repeated `memory_get` calls.
Now it's one parameter on `memory_search`.

### Response shape

Each entry in `linked_memories`:
```json
{
  "id": 42,
  "project": "dev",
  "relation": "relates",
  "strength": 0.7,
  "distance": 2,       // hops from root
  "via": 17,           // present when distance > 1; the intermediate node
  "content": "..."     // first 200 chars
}
```

### Limits

- Max 3 hops via MCP (parameter constraint); library API accepts any int
- 30-node cap per result prevents exponential blowup
- Nodes already in the top-level result set are not included as "linked"
  (callers already have them)

---

## [10.3.5] - 2026-04-16 (`mnemos doctor --migrate` + column backfill extended)

### Added

- **`mnemos doctor --migrate`** flag: apply safe fixes for detected schema
  drift. Before touching anything, copies the DB to
  `{db}.bak-pre-doctor-migrate-{timestamp}` so rollback is a file copy.
  Then:
  - Backfills any missing column in `memories` via init_schema's ALTER
    pass (v10.3.4+)
  - Creates any missing aux table (`retrieval_log`, `tool_usage`,
    `consolidation_log`, `nyx_state`)
  - Rebuilds out-of-sync FTS index (INSERT INTO memories_fts(memories_fts)
    VALUES ('rebuild'))
  - Reports which migrations were applied and which issues remain
  Idempotent. Never drops data.

- **Column backfill extended** to include `type`, `last_accessed`,
  `updated_at`. v10.3.4 missed these; pre-v10 DBs that used v8-era
  schema without `type` column would still throw on
  `CREATE INDEX idx_mem_type`.

### Changed

- **`Mnemos.doctor(migrate=False)`** now takes an optional `migrate`
  keyword. Default is inspection-only (existing behavior). When True,
  triggers the migration pass and includes `migrations_applied` + `backup`
  fields in the returned report.

---

## [10.3.4] - 2026-04-16 (graceful init_schema on pre-v10 DBs + calibration dataset)

### Fixed

- **`SQLiteStore.init_schema` now backfills missing columns on pre-v10 DBs**
  before creating any index that references them. Previously, pointing
  Mnemos at a DB that predated v10.x (typically missing the `namespace`
  column) threw "no such column: namespace" on the first CREATE INDEX.
  Now the init pass runs ALTER TABLE ADD COLUMN for any of
  `namespace` / `nyx_processed` / `subcategory` / `valid_from` /
  `valid_until` / `layer` / `consolidation_lock` / `verified` /
  `last_confirmed` that are absent, using the documented defaults. Silent
  migration, no data loss, idempotent (existing columns skipped).

### Added

- **Calibration dataset for contradiction classification**
  (`tests/data/contradiction_calibration.json`). Hand-crafted synthetic
  pairs grouped by expected class: 6 `contradicts`, 4 `refines`,
  4 `evolves`, 6 `relates`, 3 `unrelated`. Includes the canonical
  v10.2.x false-positive case (two dominance insights, complementary
  not conflicting) as the `relates` calibration anchor. No PII, no
  real user memories.

---

## [10.3.3] - 2026-04-16 (revert graceful degrade; state rerank as required)

### Changed

- **Removed the defensive rerank-off graceful degrade added in v10.3.2.**
  The cross-encoder is canonical - Mnemos's benchmark numbers and the
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
    opt-out was silently negated - the ~500 MB the user was trying to
    save got loaded anyway during the first contradiction check.
  - On truly constrained machines where the model import failed, the
    rerank call raised, was caught, and `_detect_contradictions`
    returned `[]` - dropping ALL contradictions silently.

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
  via `jq` on `CLAUDE_TRANSCRIPT` - no LLM call. Next session's briefing
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
  `evolves`, `supersedes`, `enables`). No schema migration needed -
  `memory_links.relation_type` is free-form text, this is a new sentinel
  value.

- **`MNEMOS_CONTRADICT_MODE` env var** with four values:
  - `off` - disable contradiction detection entirely
  - `vec` - Tier 1 only (vec gate, no rerank); all vec-gated candidates
    → `contradicts`. Matches pre-v10.3 behavior for users who explicitly
    want it.
  - `rerank` (default) - Tier 1 + Tier 2. Vec gate + cross-encoder rerank
    with two thresholds:
    - `CONTRADICTION_RERANK_MIN` (0.35): below → skip, not even topical
    - `CONTRADICTION_RERANK_HIGH` (0.60): above → `contradicts` + warn
    - Between MIN and HIGH → `relates`, silent link, no warning
  - `llm` - Tier 1 + Tier 2 + Tier 3 LLM classification. Each rerank
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
  the pattern would modify more memories than allowed - prevents runaway
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
  distinct operation category - not CRUD (doesn't operate on a single
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
  every MCP tool call records `(tool_name, called_at)` - no arguments, no
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
  dispatch when the flag is set. Failures swallowed - diagnostics only.

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
  predate this change - CREATE IF NOT EXISTS makes it idempotent.

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
  No MCP tool yet to emit that flag - the column is reserved.

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
  namespace with usage count and an example memory ID. Prevents tag drift -
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
  relationship graphs - one search call returns the hit plus everything it
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
