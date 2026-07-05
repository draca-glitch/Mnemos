# Changelog

All notable changes to Mnemos. Dates are from the original private development
repository, where the system existed under an internal name (`agent-memory`)
before being open-sourced as Mnemos in this repo.

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
