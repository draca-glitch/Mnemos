# Mnemos 10.17.0: Zero-LLM Daily Consolidation Cycle

> Implementation plan, 2026-07-03. Evidence base: benchmarks/weave-bench
> (NLI weave refuted, phase-2 gate validated) + benchmarks/merge-bench
> (mechanical merge cleared: A 24/25 exact, B tau=0.90 knob, C 100%/100%
> by construction). Executed in-session; this doc is the handoff artifact
> if context rolls.

**Goal:** The daily Nyx cycle runs with zero LLM calls. LLM-requiring work
(weave, synthesize, contradiction judging, prose polish) moves to an
explicit weekly / opt-in tier.

**Architecture:** cosine nominates (rank-based mutual top-k), NLI decides
(cluster admission gate + line-level dedup), mechanical union executes
(selection, never generation). Phase 4 finder queues candidates as links;
the LLM judge consumes the queue when a key is present.

## Global constraints

- No em dashes anywhere in code, docs, CHANGELOG (repo rule).
- All tunables live in constants.py with MNEMOS_* env overrides.
- SemVer pre-1.0: 10.17.0 minor, tag + push on ship (tag ceremony).
- Tests: tests/test_v1017_zero_llm.py, stub nli._score_pair (no model
  download in CI). Follow test_v1015_nli.py conventions.
- Never break the LLM path: MNEMOS_MERGE_ENGINE=llm restores 10.16
  behavior exactly.

## Tasks

### Task 1: constants + nli.line_max_duplicate
- constants.py, after NLI_FINDER_MAX_PAIRS block:
  - MERGE_ENGINE_DEFAULT = "mechanical"; env MNEMOS_MERGE_ENGINE
    (values: mechanical | llm), read at call time in phases (mirror
    DEDUP_CONFIRM_DEFAULT pattern).
  - MECH_MERGE_TAU (env MNEMOS_MECH_MERGE_TAU, default 0.90)
  - MECH_MERGE_MIN_LINE_CHARS (env, default 25): lines shorter than this
    dedup by exact match only (merge-bench Arm A failure class).
  - CLUSTER_GATE_DEFAULT = "nli"; env MNEMOS_CLUSTER_GATE (nli | off),
    call-time read. CLUSTER_GATE_TAU (env, default 0.70).
    CLUSTER_GATE_MAX_LINES (env, default 8).
  - CANDIDACY_DEFAULT = "mutual-topk"; env MNEMOS_CANDIDACY
    (mutual-topk | threshold), call-time read. CANDIDACY_TOP_K (env,
    default 3).
  - WEAVE_NOVELTY_TAU (env, default 0.85)
  - CONTRADICT_JUDGE_DEFAULT = "auto"; env MNEMOS_CONTRADICT_JUDGE
    (llm | queue | auto). auto = llm when configured else queue.
- nli.py: public line_max_duplicate(a, b, top_k=8) mirroring
  line_max_contradiction but scoring min-direction P(entail) per line
  pair (max over pairs). Returns None when backend unusable.

### Task 2: mnemos/consolidation/mechanical.py (new)
Port benchmarks/merge-bench/mechanical.py with production hardening:
- mechanical_merge_cluster(cluster_ids, mem_by_id) -> str | None
  Same call surface as merge_cluster. Sort members by created_at,
  explode_cml_chain each, pool lines newest-first, keep line unless
  bidirectional entailment >= MECH_MERGE_TAU with an already-kept line
  (newer phrasing wins). Exact-match fast path. Lexical prefilter
  (Jaccard >= 0.10 or shared digit token) skips NLI on unrelated lines.
  Lines < MECH_MERGE_MIN_LINE_CHARS: exact-match dedup only.
  Returns None when nli.is_available() is False (caller skips cluster,
  loud log; no silent LLM fallback in mechanical mode).
- Result flows into existing apply_merge unchanged (explode + size-guard
  + atomic transaction + lineage row all reused).

### Task 3: NLI cluster admission gate (phases.py)
- nli_cluster_gate(cluster, mem_by_id) -> list[int]: pairwise
  line_max_duplicate over members (content truncated to first
  CLUSTER_GATE_MAX_LINES lines); member admitted when any pairwise score
  >= CLUSTER_GATE_TAU; returns admitted ids ([] or singleton = cluster
  dissolves).
- Wire into phase_dedup between find_clusters and merge: gate each
  cluster, log ejections, merge only admitted subsets of size >= 2.
  MNEMOS_CLUSTER_GATE=off or NLI unavailable: log loudly, pass cluster
  through ungated (legacy).

### Task 4: mutual top-k candidacy (phases.py)
- mutual_topk_adjacency(sim_matrix, k) -> bool matrix: adj[i][j] true iff
  j in i's top-k AND i in j's top-k (ties by similarity).
- find_clusters gains candidacy parameter; "mutual-topk" ANDs the
  adjacency with the threshold test (threshold still floors degenerate
  cases at 0.5); "threshold" = legacy.
- phase_dedup: when MNEMOS_CANDIDACY=mutual-topk AND merge engine is
  mechanical, run a single phase-2 pass (tier 2B topic-merge retired:
  aggregation-merges are generation work, they move to the LLM tier).
  Legacy engine keeps both tiers.

### Task 5: engine switch in phase_dedup + orchestrator llm_phases
- phase_dedup: engine = env MNEMOS_MERGE_ENGINE or MERGE_ENGINE_DEFAULT.
  mechanical -> mechanical_merge_cluster; llm -> merge_cluster (10.16
  path, unchanged).
- orchestrator.consolidate: llm_phases computed dynamically:
  {5} | {3} | ({2} if engine == "llm") | ({4} if judge resolves to llm).
  The no-LLM guard then only strips phases that truly need a key. Log
  the resolved tier plan at start.

### Task 6: phase-4 judge queue mode (phases.py)
- phase_contradict judge modes: llm (today's path) | queue. Queue mode:
  after the NLI finder, INSERT OR IGNORE memory_links
  (relation_type='contradiction-candidate', strength=P(contra)) per
  flagged pair; stats["queued"]. LLM mode first consumes queued
  candidate links (SELECT pairs with relation_type=
  'contradiction-candidate'), judges them, replaces the candidate link
  with the verdict link, then continues with fresh candidates.

### Task 7: weave hygiene (phases.py, weekly tier)
- Staleness guard: exclude from original_ids any id with an outgoing
  superseded_by or evolves link (SELECT DISTINCT source_id FROM
  memory_links WHERE relation_type IN ('superseded_by','evolves')).
- Novelty gate: before store_bridge_insight, when NLI available: skip
  insight (keep link) if max one-direction P(entail) from either source
  content (truncated 700 chars) to the insight text >= WEAVE_NOVELTY_TAU.
  stats["insights_skipped_redundant"].
- store_bridge_insight: layer 'episodic' (INSERT and prep_memory_text),
  bridges decay unless retrieval keeps them alive.

### Task 8: useful-loop (core.py)
- In Mnemos.get (core.py:1122): when enable_retrieval_log, after fetch:
  UPDATE retrieval_log SET useful=1 WHERE memory_id=? AND useful IS NULL
  AND retrieved_at >= datetime('now','localtime','-24 hours').
  Get-after-search inside a day = usefulness signal.

### Task 9: version, CHANGELOG, docs, tests, ship
- pyproject.toml 10.17.0; CHANGELOG entry (evidence-linked); README +
  docs/ARCHITECTURE: zero-LLM daily cycle section, tier table, the
  one-liner: consolidates nightly without any LLM; add a key and it
  dreams weekly.
- tests/test_v1017_zero_llm.py covering: mechanical merge collapse +
  preserve + short-line guard + tau boundary (stubbed _score_pair);
  cluster gate admit/eject/dissolve; mutual_topk_adjacency; dynamic
  llm_phases; judge queue links; novelty gate skip; episodic bridge
  layer; useful-loop update. Full suite green before ship.
- Tag v10.17.0, push commit + tag (tag ceremony = push clearance).

### Task 10: Epsilon ops (server-side, outside repo; confirm with Mikael)
- run-nyx.sh (daily): drop anthropic.env sourcing, export
  MNEMOS_MERGE_ENGINE=mechanical, MNEMOS_CONTRADICT_JUDGE=queue,
  phases 1,2,4,6 via --phases.
- New run-nyx-weekly.sh: LLM env as today, phases 1,2,3,4,6, judge=llm
  (consumes queue), weave hygiene active. systemd timer Sun 03:30.
- CLAUDE.md MNEMOS section update after ship.
