# Mnemos Architecture

## Design principles

1. **A memory system should be a memory.** It stores data and retrieves it. That is the entire job. No LLM in the retrieval or storage pipeline. The agent thinks; Mnemos remembers.
2. **CML is a language, not a compression algorithm.** Condensed Memory Language is a token-minimal notation the agent writes directly. It is not compressed or encoded; it is just denser English with structural prefixes and operators. Both humans and LLMs read it without decoding. Prose that slips through is shaped into CML at store/ingest time (`cemelify()`), not by the Nyx cycle: since v10.20.0 the cycle never rewrites already-stored content in place.
3. **Retrieval is configurable, not fixed.** Four modes, two flags: BM25 + vector only (lite), BM25 + vector + reranker (clean), BM25 + vector + CML (not recommended), BM25 + vector + reranker + CML (canonical). The reranker is enabled by default but not required for upper-90s numbers; even the lite mode lands in the same tier as any verified no-LLM number in the field.
4. **Hybrid retrieval beats single-method.** Combining lexical (BM25) and semantic (vector) signals via RRF, then reranking with a cross-encoder, consistently outscores either alone.
5. **Curated > Verbatim.** Memories are distilled facts, decisions, and learnings, not raw chat transcripts. Higher signal-to-noise.
6. **CPU-only is a feature, not a constraint.** ONNX models run on every laptop, NAS, Pi 4+, and budget VPS. No GPU monopoly.
7. **Storage is pluggable, retrieval is not.** The pipeline (FTS5 + vec + RRF + rerank) is the same across backends. Only persistence varies.
8. **No auth in the engine.** Mnemos is a memory engine. Authentication is the responsibility of the transport layer (MCP server, HTTP API, etc.).
9. **Forgetting is a feature.** Exponential temporal decay matches how human memory works. Old, unaccessed memories naturally fade.
10. **A system should only be as complicated as it needs to be.** Four hot-path tools plus two maintenance tools, one database file, no LLM in the pipeline, no GPU. Add complexity when the workload demands it, not before.

## Layered architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Client (any MCP)                       │
│  Claude Code  │  Cursor  │  ChatGPT  │  Gemini  │  custom   │
└────────────────────┬────────────────────────────────────────┘
                     │ JSON-RPC 2.0 / stdio
┌────────────────────▼────────────────────────────────────────┐
│                  mnemos.mcp_server                           │
│      4 hot-path tools: store, search, get, update            │
│      + 2 maintenance tools: bulk_rewrite, list_tags          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                     mnemos.core.Mnemos                       │
│  • 3-way dedup on store (NLI confirm tier)                   │
│  • BM25 + vector search (FTS + vec + RRF + rerank)           │
│  • Auto-widen on thin results                                │
│  • Contradiction detection (NLI)                             │
│  • Dynamic importance bumping                                │
└───┬──────────────┬───────────────┬───────────────┬──────────┘
    │              │               │               │
┌───▼─────────┐ ┌──▼────────────┐ ┌▼─────────────┐ ┌▼─────────────┐
│mnemos.embed │ │ mnemos.rerank │ │ mnemos.nli   │ │ mnemos.query │
│ (FastEmbed  │ │ (cross-       │ │ (NLI dedup/  │ │ (FTS5 query  │
│  embedder)  │ │  encoder,     │ │  contradict, │ │  builder)    │
│             │ │  search only) │ │  ONNX)       │ │              │
└─────────────┘ └───────────────┘ └──────────────┘ └──────────────┘
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
- **Access telemetry**: `access_count`, `last_accessed`
- **Explicit verification**: `last_confirmed`, recorded by
  `memory_update(confirmed=true)` after checking a source, never by a read.
  See [memory correctness semantics](memory-correctness.md).

### Memory link
A typed relationship between two memories: `related`, `contradicts`, `supports`, etc. Stored with a strength (0-1).

### Nyx insight
A consolidation event recording which source memories fed a result. Stored in the `nyx_insights` table. Includes `consolidation_type` (`aggregation` for a phase-2 merge, `synthesis` for a phase-5 insight), which the tier-2 `expand_merged` lineage join reads to walk a merged memory back to its sources.

### Vector index tables
Vectors live in a two-tier sqlite-vec layout:
- **Active index**: `embed_vec` (the `vec0` virtual table holding the float vectors) paired with `embed_meta` (row provenance: `source_db`, `source_id`, `text_hash`, `model`, `embedded_at`). Every active memory has one row here.
- **Tier-2 archived index** (v10.7.0): `embed_vec_arch` + `embed_meta_arch`, the same shape, holding vectors for memories that consolidation archived. Keeping the archived vectors in a separate index lets tier-2 recall KNN the originals directly (see tier-2 recall) instead of losing them at merge time. On new stores `embed_vec_arch` is created with a declared primary key (`vec0(id INTEGER PRIMARY KEY, ...)`); older implicit-rowid arch tables keep working through the `_arch_join_col` auto-detection used by every archive-side query.

`embed_meta` and `embed_meta_arch` default `embedded_at` to local time (`datetime('now', 'localtime')`) so a synchronous embed lines up with its memory's `created_at`. Tables created before that DDL carried a UTC default; `mnemos doctor --migrate` detects the UTC dialect from the table SQL and rebuilds it with the localtime default (values converted DST-correctly).

### Store self-check: `mnemos doctor`

`mnemos doctor` runs a read-only health check over the store and reports issues without mutating anything; `mnemos doctor --migrate` applies the safe, reversible repairs behind an automatic pre-migration backup (the backup path is returned so a caller can roll back). It verifies, in order:

- **Integrity** (`PRAGMA quick_check`) first, before any read that assumes a sane btree, catching page/btree corruption from an unsafe copy of a live WAL-mode DB.
- **Schema drift**: the required `memories` columns and the v10.2+ auxiliary tables (`retrieval_log`, `tool_usage`, `consolidation_log`, `nyx_state`); `--migrate` backfills via `init_schema`.
- **Empty-store detection** (v10.23.0): 0 active memories in the resolved namespace is flagged as a likely wrong `MNEMOS_DB` or `MNEMOS_NAMESPACE` rather than blessed as "healthy".
- **FTS sync**: `memories` row count vs `memories_fts`; `--migrate` rebuilds the FTS index.
- **Content/vector coherence** (v10.19.0): re-derives each active memory's embed-text hash from current content and compares it to the stored `embed_meta.text_hash`, catching content edited without a re-embed (direct SQL writes, or a write-path bug), which retrieval cannot otherwise see. Near-total mismatch is reported as an embed-text format change across versions, not row corruption; `--migrate` re-embeds the flagged rows.
- **Vector-model provenance**: flags same-dimension vectors from a different embedder coexisting in one index (which makes every KNN comparison silently meaningless).
- **Archive lifecycle** (v10.24.0): the UTC `embedded_at` dialect migration (above), orphan tier-2 rows left by pre-v10.24 hard deletes, and tier-2 completeness (archived memories missing a vector, or carrying a legacy pre-canonical hash, both pointing at `reindex-archived`).

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
        │  (reranker)          │
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

Candidates from all three signals are pooled and confirmed by a final scorer. The three signals catch different failure modes: keyword overlap finds verbatim repeats, subject matching finds same-topic different-phrasing, and vector similarity finds semantic paraphrases. A single one of them missing a duplicate is normal; all three missing it is rare.

The confirm tier has two modes:

- **`MNEMOS_DEDUP_CONFIRM=nli` (recommended, v10.15+)**: bidirectional entailment through the NLI layer, a candidate blocks the store only when each side entails the other (min-direction P(entailment) >= `MNEMOS_NLI_DEDUP_THRESHOLD`, default 0.85). This asks the duplicate question directly and stops the reranker failure mode of blocking structurally-similar-but-distinct facts (bench: 1 false positive vs 16-21 false blocks for distance-based confirmation). When NLI says "not a duplicate" that verdict is final.
- **Legacy (default)**: the pooled candidates are scored by the Jina cross-encoder (or raw vector distance when the reranker is disabled) and the top score is compared to the dedup threshold. Kept as the zero-extra-setup path.

### Real-time contradiction detection on store

Separately from dedup, every `memory_store()` call also checks whether the new memory conflicts with existing facts on the same topic. The mechanism is non-LLM:

1. Vector similarity gate (L2 distance below `CONTRADICTION_VEC_THRESHOLD`)
2. Same-project filter (cross-project topical overlap is almost always complementary, not contradictory, this is the dominant false-positive killer)
3. A classification tier, selected by `MNEMOS_CONTRADICT_MODE`:
   - **`nli` (recommended, v10.15+)**: max-direction P(contradiction) through the NLI layer; warn + link only at >= `MNEMOS_NLI_CONTRA_THRESHOLD` (default 0.98). A cross-encoder scores topicality ("same subject?"), which is why the legacy mode conflated agreement with conflict; an NLI model scores polarity ("opposite claims?"), which is the actual question (bench: AUC 0.94 vs 0.69).
   - **`rerank` (default)** / **`llm`** / **`vec`**: the legacy tiers, Jina topical-match score with a high-score band, optionally refined by an LLM classifier.

A conflict surfaces as a warning in the response to the caller plus a `memory_links` row with `relation_type='contradicts'` that the next Nyx cycle's Phase 4 batch pass will consider. This is the synchronous counterpart to Phase 4 Contradict: the on-store check catches an incoming memory conflicting with what is already in the store the moment it arrives; the batch pass catches slow-burn evolution that only becomes visible once enough related memories accumulate. Both write into the same `memory_links` infrastructure.

### The NLI decision layer (v10.15+)

`mnemos/nli.py` (opt-in extra `mnemos[nli]`) supplies the scorers behind the `nli` modes above and, since the zero-LLM cycle (v10.17.0), the nightly consolidation path: the phase-2 admission gate (`line_max_duplicate`), the mechanical merge's line dedup (`bidirectional_entailment`), and the phase-4 contradiction finder (`line_max_contradiction`, memoized in `nli_scan_cache` since v10.18.0). Design points:

- **Two models, routed per comparison.** Content that reads as English (cheap stopword heuristic, `is_english()`) is scored by an English ANLI+FEVER-hardened DeBERTa-v3 checkpoint, the strongest benched; anything else goes to a multilingual XNLI mDeBERTa checkpoint (~100 languages). Both load lazily and independently. Pairing this with the [English-primary store convention](english-primary.md) puts the strong model on nearly every decision.
- **Direction aggregation is load-bearing.** Contradiction takes the max over both premise/hypothesis orders (real conflicts score asymmetrically, benched 0.44 one way / 0.99 the other); duplicates take the min of both entailment directions (a duplicate entails mutually; a subset-fact does not, and correctly does not block).
- **Line-level scoring for consolidated records** (`line_max_contradiction`): top-k cosine-preselected line pairs, max P(contradiction). Isolating conflicting statements rescues conflicts that whole-record scoring buries under surrounding text (benched: a real conflict scored 0.58 blob-level, 0.9956 line-level).
- **Backends**: ONNX (preferred; onnxruntime is already in the dependency tree, models exported once via `scripts/export_nli_onnx.py` into `MNEMOS_NLI_ONNX_DIR`) or torch (`mnemos[nli-torch]`). ONNX fp32 is score-identical to torch (parity-gated: probability drift 1e-05, zero threshold flips on 114 bench pairs). int8 quantization is deliberately NOT offered: it collapses DeBERTa-v3 scoring to chance and was rejected by the same gate. `MNEMOS_NLI_BACKEND` pins `auto`/`onnx`/`torch`.
- **Graceful degradation**: with no backend usable, every entry point returns None and the store path falls back to the legacy scorers.
- The reranker is unaffected: topicality is the right signal for search ranking, so Jina v2 keeps that job. NLI replaced it only where the question is polarity, not relevance.

## The Nyx cycle

Modeled on brain sleep stages. Six phases, run on a **two-tier schedule** since v10.17.0.

The **nightly tier runs with zero LLM calls**: phases 1, 2, 4, 6. Cosine similarity nominates candidate pairs by mutual top-k rank, the NLI layer decides (phase-2 admission gate + line-level dedup + phase-4 contradiction finder), a mechanical line union executes merges (selection, never generation), and phase 4 queues contradiction-candidate links for later judging. No API key is needed.

The **weekly tier is the optional LLM-driven enrichment layer**: phases 1, 2, 3, 4, 6, where phase 3 (weave) always calls an LLM, phase 2 becomes generative only under `MNEMOS_MERGE_ENGINE=llm`, and the phase-4 judge drains the nightly contradiction queue. Phase 5 (synthesize) is opt-in on top of that, enabled by the `--nyx` flag on the weekly script.

Which phases actually call an LLM is derived at runtime from the configured engines, not fixed by phase number:

- Phase 3 (weave) and phase 5 (synthesize) always generate text, so they always need an LLM.
- Phase 2 (dedup) needs one only under the legacy `llm` merge engine; the default `mechanical` engine is selection, not generation.
- Phase 4 (contradict) needs one only when its judge resolves to `llm`; the default `auto` judge falls back to `queue` (record-only) when no key is present.
- Phases 1 (triage) and 6 (bookkeeping) are pure SQL.

If no LLM is configured, the LLM-requiring phases are skipped with a grep-able `WARNING` and the core cycle continues (this replaced the pre-v10.17 loud-fail). `mnemos consolidate` behaves the same way: it runs the zero-LLM phases and warns on the ones it must skip rather than failing. `MNEMOS_DISABLE_LLM=1` opts into the same skip silently. Originals are archived, never deleted, throughout.

### Phase 1. Triage

Pure SQL. Reads `consolidation_log` to find the last run timestamp, then counts memories with `created_at > last_run_at` among the active set. Decides surge mode when the new-memory count exceeds `SURGE_THRESHOLD`, which relaxes the per-phase LLM-call budget on the theory that a burst of new memories is worth extra consolidation effort. Returns the list of new-memory IDs for downstream phase use.

### Phase 2. Dedup

Cluster near-duplicate memories and merge each cluster into one. Protection filters apply first: memories with `importance >= 9`, the `evergreen` tag, `consolidation_lock = 1`, or `type = 'decision'` are excluded from merging (decisions are authoritative records, so they are woven and contradiction-scanned but never blended). They stay in the full embedding set so weave and contradict still reach them.

**Candidacy and clustering.** The default `MNEMOS_CANDIDACY=mutual-topk` builds pair candidacy from mutual top-k nearest-neighbor rank (`MNEMOS_CANDIDACY_TOP_K`, default 3) rather than an absolute cosine cutoff, because absolute thresholds are noise in the compressed e5 space (measured: 45% of all active pairs clear 0.78). A 0.5 cosine floor only guards degenerate cases. The legacy `threshold` candidacy keeps the absolute-cutoff behavior with `TIGHT_THRESHOLD = 0.88` (near-duplicate) and `TOPIC_THRESHOLD = 0.75` (same-topic). Clusters are capped at `MAX_CLUSTER_SIZE = 8`.

**NLI admission gate.** Before any merge, each candidate cluster passes the NLI admission gate (`MNEMOS_CLUSTER_GATE=nli`, default; tau `MNEMOS_CLUSTER_GATE_TAU = 0.70`): a member stays only if it shares at least one line-level bidirectional-entailment fact with another member. Members with no shared fact are ejected; clusters left with fewer than two survivors dissolve unmerged. Cosine noise clusters that slipped through rank candidacy die here.

**Merge engine.** The default `MNEMOS_MERGE_ENGINE=mechanical` is a zero-LLM line union: the atomic lines of all cluster members are pooled and deduplicated by bidirectional entailment (`MNEMOS_MECH_MERGE_TAU = 0.90`; lines shorter than `MNEMOS_MECH_MERGE_MIN_LINE_CHARS = 25` dedup by exact match only, a short-enumerated-line NLI failure class), with the newer phrasing winning on a duplicate. Every output line is an input line verbatim, so fact preservation holds by construction rather than by audit. The engine returns nothing (cluster skipped, logged loudly) when the NLI backend is unavailable; there is no silent fallback to an LLM in mechanical mode.

The opt-in `MNEMOS_MERGE_ENGINE=llm` restores the generative path. Clusters of size 2 run one LLM call with the `MERGE_SYSTEM` prompt. Clusters of size 3 or more use **hierarchical pairwise merging**: each step merges two inputs at a time into one output, which then participates as an input at the next level. The LLM never sees more than two memories at once, so the natural per-step "this output should be roughly the size of its inputs" intuition holds even for deep clusters. A flat N-way merge of 8 memories forces the model into aggressive fact-dropping to hit a single-memory target size; the hierarchical variant removes that pressure and preserves specifics. Under the mechanical engine the phase-2B topic tier is retired (aggregating distinct same-topic facts is generative, LLM-tier work); the llm engine keeps both the tight and topic tiers.

**Storage side.** The merged memory is stored with the `consolidated` tag, `consolidation_type = 'aggregation'`, and a `nyx_insights` lineage row naming the source ids (the lineage the tier-2 `expand_merged` join reads). Every source memory is flipped to `status='archived'`, its `tags` appended with `merged-into-<new_id>`, and its vector moved into the tier-2 archived index in the same step (via `archive_memory`, v10.24.0). Nothing is deleted; the sources stay queryable via archive-aware retrieval paths (see tier-2 recall).

### Phase 3. Weave

Scans pairs of memories from different projects for cross-category connections. A candidate pair must have cosine similarity above a weave-specific threshold (lower than the tight/topic thresholds, because cross-domain connections are usually weaker in embedding space than intra-domain duplicates). The LLM is asked whether the pair genuinely reflects a thematic or causal connection; if yes, a `memory_links` row is created with `relation_type='related'` and a strength score. Output is pruned to the top-K per memory to prevent link explosion.

Weave's output is the link. The model also returns a one-line insight, and until v10.43 every accepted pair stored that line as a new `bridge` memory (unless the NLI novelty gate found it entailed by one source). Since v10.44.0 that is opt-in: `MNEMOS_WEAVE_STORE_INSIGHTS=1` stores bridge memories again, at most `MNEMOS_WEAVE_MAX_INSIGHTS` (default 5) per run, the most similar pairs first. The reason is in the 10.44.0 changelog entry: the number of bridges tracked how readily the weave model says yes, and a more agreeable model filled two stores with them in three weeks.

### Phase 4. Contradict

Scans same-project decision/fact pairs and classifies their relationship. It scans the **full active set** (`all_embeddings`), not just the mergeable subset, so protected memories (decisions, verified, `importance >= 9`), exactly the population most worth contradiction-scanning, are included. Pairs already carrying a verdict or candidate link are skipped before scoring, so a cleared pair never re-enters the loop.

**Finding** candidate pairs has two modes (`MNEMOS_NYX_CONTRADICT_FINDER`):

- **`cosine` (default)**: pairs inside a similarity band (`CONTRADICT_MIN_SIM`..`CONTRADICT_MAX_SIM`). Cheap, but the band ceiling excludes near-identical pairs, which is exactly where value-conflicts ("X is 64GB" vs "X is 32GB") live.
- **`nli` (recommended with the NLI layer installed)**: floor-only cosine gate, then the line-level NLI finder scores each pair and only pairs at P(contradiction) >= `MNEMOS_NLI_FINDER_THRESHOLD` (default 0.8, recall-first) advance. Scores are memoized in `nli_scan_cache` keyed on content hashes (v10.18.0), so a static pair is scored once and costs nothing on later nights; `MNEMOS_NLI_FINDER_MAX_PAIRS` budgets only advanced pairs and never-scored ones backfill across subsequent runs.

**Judging** the flagged pairs has two modes (`MNEMOS_CONTRADICT_JUDGE`, default `auto`, which resolves to `llm` when a key is present and `queue` when not):

- **`queue` (zero-LLM nightly tier)**: each flagged pair is recorded as a `contradiction-candidate` link and left inert (no archiving, no `valid_until` edits). A later LLM-judged run consumes the queue.
- **`llm`**: consumes any queued candidates first, then fresh pairs, classifying each with the `CONTRADICT_SYSTEM` prompt into one of **five** labels:
  - `COMPATIBLE`, same topic, different aspects; tombstoned with a `contradiction-cleared` link so it never re-enters the finder/judge loop.
  - `UNRELATED` (v10.21.0), same project but a different subject (exhaustive same-project candidacy necessarily feeds these); also tombstoned `contradiction-cleared`. Added because without a way to say "unrelated" the judge scope-conflated shared vocabulary into false `CONTRADICTS`.
  - `SUPERSEDED`, newer memory updates the same fact with a new value; the older gets a `valid_until` set and is archived, with a `superseded_by` link to the newer. A `verified` or `importance >= 9` older memory keeps its status and only records the link (blast-radius guard, since the verdict derives from untrusted memory content).
  - `EVOLVED`, a belief or preference that genuinely shifted over time; both versions stay queryable, the older gets a `valid_until` plus an `evolves` link so the evolution chain is preserved for user queries like "when did I stop believing X?"
  - `CONTRADICTS`, two memories state incompatible things and it is not clear which is current; flagged with a `contradicts` link but neither is marked superseded. Leaves the disambiguation to the user or a later cycle with more context.

This is the **batch** contradiction pass. The real-time on-store contradiction check (see the [Contradiction detection](../README.md#contradiction-detection-real-time-on-store) section of the README) runs a lighter non-LLM version, vector similarity plus same-project filter plus a classification tier, every time `memory_store()` is called, so obvious conflicts get flagged immediately in the response rather than having to wait for a cycle.

### Phase 5. Synthesize

Opt-in (not in the default phase set; enabled by the weekly script's `--nyx` flag). Feeds clusters of semantically-related memories into the LLM and asks for novel observations: recurring themes, cross-domain patterns, tensions, preferences evolving in parallel across domains. The output is stored as new memories with `type='learning'`, `layer='semantic'`, and a `nyx_insights` row linking back to the source memories that informed the insight. These synthesized memories participate in future searches like any other fact and are themselves eligible for further consolidation in later Nyx cycles, so the system's self-generated observations compound over time rather than being one-shot reports.

### Phase 6. Bookkeeping

Pure SQL, always runs regardless of LLM configuration.

- **Temporal decay**: adjust `importance` downward for memories with recent `access_count` below a threshold, following exponential-decay half-lives (`EPISODIC_HALFLIFE_DAYS` and `SEMANTIC_HALFLIFE_DAYS` in `constants.py`). Nothing is deleted, decayed memories just rank lower in hybrid search until access revives them.
- **Orphan vector hygiene** (tier-2 aware since v10.7.0): a vector whose memory was archived is *moved* from the active index (`embed_vec`/`embed_meta`) into the tier-2 archived index (`embed_vec_arch`/`embed_meta_arch`) so tier-2 recall can still reach the original; a vector is only removed outright when its memory row no longer exists at all (a hard delete). This is the end-of-cycle safety net behind the per-archive move `archive_memory` now does inline.
- **Stale link pruning**: drop `memory_links` rows where either endpoint has been archived, so the link graph reflects the current active memory surface without dragging old archived chains along.
- **Scan-cache cleanup**: drop `nli_scan_cache` rows whose pair references a non-active memory (archived or deleted memories can never re-enter the phase-4 candidate set, so their cached scores are dead weight).

### LLM configuration

When they run in their LLM modes (weave and synthesize always; merge only under `MNEMOS_MERGE_ENGINE=llm`; contradict only under an `llm` judge), these phases call through the `llm_utils` abstraction. `MNEMOS_LLM_MODEL` is the global default; per-phase env var overrides are first-class:

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

Merging is never silent deletion. The archived originals of any merged memory stay in the database (`status='archived'`, `merged-into-<id>` appended to their tags), with their vectors moved into the separate tier-2 archived index (`embed_vec_arch`/`embed_meta_arch`) rather than dropped. Tier-2 recall is opt-in per search, via `memory_search(..., expand_merged=True)`, and reaches the originals through two complementary paths:

1. **Lineage expansion.** For each consolidated result, Mnemos reads its `nyx_insights` source-id list and returns the originals under a `merged_from` key, filtered to exclude any source superseded via `valid_until`. This surfaces the constituents of a merged memory that already ranked.
2. **Direct archived KNN** (v10.7.0). The query embedding is also KNN'd against `embed_vec_arch` directly, so an archived original is reachable even when its consolidated parent did not rank in primary search. Fresh hits come back under a `tier2_recall` key with their vector distances.

The merged memory is the fast path; the archived sources are the precision path. `mnemos reindex-archived` backfills the tier-2 index, embedding any archived memory that has no arch vector yet and re-embedding rows carrying a legacy pre-canonical hash (idempotent). The lifecycle around this index was hardened in v10.24.0: soft-delete/archive moves the vector into tier-2 (`move_embedding_to_archive`), hard-delete purges the tier-2 rows too, and `mnemos doctor` reports archived memories missing a tier-2 vector as an incomplete index pointing at `reindex-archived`.

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
- cross-encoder reranker ONNX (~2-3s cold-load on CPU)

Both are cached in memory after first load. Subsequent queries are fast (<200ms total).

### RAM and disk footprint

Mnemos is CPU-only but it does load real models into RAM. Component breakdown:

| Component | Disk | RAM (loaded) | When |
|---|---|---|---|
| Python + Mnemos package | ~50 MB | ~100-150 MB | Always |
| SQLite + sqlite-vec | trivial | ~10-50 MB (page cache) | Always |
| embedder (ONNX) | default tier ~35 MB, multilingual ~500 MB | ~150 MB / ~1 GB | Always loaded once on startup |
| cross-encoder reranker (ONNX) | default ~600 MB, multilingual ~1.1 GB on disk | ~300-500 MB | Only if `enable_rerank=True` |

**With reranker** (default): **~0.5-0.7 GB resident on the default tier, ~1.5-1.7 GB on the multilingual tier**. First call spools the cross-encoder ONNX (1-2s cold-start), then subsequent reranks are ~50 ms. The MCP server warms it up at boot so the spool cost is paid once at startup rather than on the first user query.

**Without reranker** (opt-out via `MNEMOS_ENABLE_RERANK=0`): **~0.2-0.4 GB resident on the default tier** (~1-1.2 GB multilingual). The R@5 cost looks like half a point; the measured R@1 cost is eleven (81.29 vs 92.40), so treat this as a hardware-forced mode, not a tuning knob.

**Disk**: ~650 MB for the default tier's two ONNX models (~3.3 GB for the multilingual tier), downloaded automatically from HuggingFace on first use and cached under `~/.cache/fastembed`.

**Sub-1 GB hardware**: the default tier (bge-small + gte-modernbert, since v10.33.0) sits well under 1 GB resident and runs on a 2 GB Pi. For 512 MB devices, disable the reranker as well; R@1 drops hard (81 vs 92 measured) but the system still works as a basic memory store. Multilingual stores cannot shrink this way: the multilingual tier (e5-large + jina-v2) is the smallest combination that works for non-English content, measured against every smaller multilingual candidate.

## Multi-source indexing (production pattern)

The author runs Mnemos in **split-storage mode (SQLite + Qdrant)** in production. Mnemos itself only indexes the curated memory layer, but the same FastEmbed pipeline is used to index bulk content via Qdrant collections:

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

**A system should only be as complicated as it needs to be.** That principle runs through every design decision in this project: four hot-path MCP tools plus two maintenance tools instead of nineteen, one SQLite file instead of a separate vector database, no auth in the storage layer instead of a half-baked role system, no LLM in the search path instead of a generative reasoning step nobody asked for. Add complexity when the workload demands it, not before.
