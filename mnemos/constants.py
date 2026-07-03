"""
Centralized configuration for Mnemos.

Single source of truth for all tuning constants, ranking formulas, and
thresholds. Override via environment variables prefixed with MNEMOS_.
"""

import os

# --- Temporal decay ---
# Episodic (time-bound events): half-life ≈ 46 days
# Semantic (distilled knowledge): half-life ≈ 180 days
DECAY_RATE = 0.015           # ln(2)/46 ≈ 0.015
DECAY_RATE_SEMANTIC = 0.00385  # ln(2)/180 ≈ 0.00385
DECAY_FLOOR = 0.1  # old memories never drop below 10% boost

# --- Search behavior ---
HYBRID_MIN_MEMORIES = 10  # activate vector search above this count
RRF_K = 60  # Reciprocal Rank Fusion constant

# --- BM25 FTS weights (content, project, tags) ---
BM25_WEIGHTS = (5.0, 2.0, 3.0)

# --- Dynamic importance: access_count thresholds → minimum importance ---
IMPORTANCE_THRESHOLDS = [(20, 8), (10, 7), (5, 6)]

# --- Vector dedup threshold (L2 distance on normalized vecs) ---
# 0.40 L2 ≈ 0.80 cosine similarity
VEC_DEDUP_MAX_DISTANCE = 0.40

# --- Ranking boost multiplier ---
IMPORTANCE_BOOST = 0.3
ACCESS_BOOST = 0.05
ACCESS_CAP = 20

# --- Confirmation recency boost ---
CONFIRM_BOOST_30D = 0.3
CONFIRM_BOOST_90D = 0.15

# --- Contradiction detection ---
# Tiered classification (v10.3.0, 2026-04-16). The previous single-threshold
# design conflated "same topic" (what the cross-encoder scores) with "actually
# contradicts" (what we care about), producing noisy false positives on
# same-topic-complementary pairs. New scheme:
#
#   vec distance < CONTRADICTION_VEC_THRESHOLD   → candidate, proceed
#   rerank < CONTRADICTION_RERANK_MIN            → skip (not even topical)
#   rerank CONTRADICTION_RERANK_MIN..HIGH        → likely `relates` (silent link, no warning)
#   rerank >= CONTRADICTION_RERANK_HIGH          → likely contradicts (link + warning),
#                                                   OR Tier-3 LLM classification if enabled
#
# The 2026-04-09 autoimprove tuning result (vec 0.35, rr 0.35, same-project
# filter) is preserved as the min gate; the new HIGH threshold introduces
# the silent-link zone that kills false-positive warnings.
CONTRADICTION_VEC_THRESHOLD = 0.35
CONTRADICTION_RERANK_MIN = 0.35    # below: skip entirely
CONTRADICTION_RERANK_HIGH = 0.60   # above: likely contradicts; between: relates

# Backward-compat alias: v10.2.x code that imported the old name stays working.
CONTRADICTION_RERANK_THRESHOLD = CONTRADICTION_RERANK_MIN

# --- Contradiction detection mode ---
# Three-tier classification behaviour:
#   off    - disable contradiction detection entirely
#   vec    - Tier 1 only (vec gate, no rerank); all vec-gated pairs → contradicts.
#            This is the honest opt-out path for users who have disabled the
#            reranker (MNEMOS_ENABLE_RERANK=0).
#   rerank - Tier 1 + Tier 2 (default, recommended): vec + rerank with HIGH
#            threshold distinguishing `contradicts` from `relates`. REQUIRES
#            the cross-encoder: the `relates` silent-link refinement is what
#            rerank buys you. If you disable rerank, use mode=vec instead -
#            mode=rerank will silently return no contradictions.
#   llm    - Tier 1 + Tier 2 + Tier 3 LLM classification: moderate-score pairs
#            are classified by LLM into {contradicts, refines, evolves,
#            relates, unrelated}. REQUIRES both the cross-encoder AND
#            MNEMOS_LLM_* env vars configured.
DEFAULT_CONTRADICT_MODE = os.environ.get(
    "MNEMOS_CONTRADICT_MODE", "rerank"
).lower()

# --- Dedup rerank threshold ---
DEDUP_RERANK_THRESHOLD = 0.85  # raised from 0.70: 0.70 over-blocked related-but-distinct memories (reranker scored genuinely distinct facts 0.81-0.84). Bias toward false-store (Nyx merges dups later) over false-block (silent memory loss).

# --- NLI decision layer (v10.15.0, bench-backed: benchmarks/nli-bench) ---
# The NLI layer replaces the cross-encoder for the store DECISION questions
# (duplicate? contradiction?). The reranker stays for search ranking, where
# topicality is the right signal. Language routing is agnostic: English
# content uses NLI_EN_MODEL (strongest benched), everything else uses the
# multilingual NLI_MULTI_MODEL.
NLI_EN_MODEL = os.environ.get(
    "MNEMOS_NLI_EN_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
NLI_MULTI_MODEL = os.environ.get(
    "MNEMOS_NLI_MULTI_MODEL",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
NLI_MAX_LENGTH = 512

# Store-path thresholds. Contradiction: max-direction P(contra) at 0.98
# scored AUC 0.939 with 2 false positives on the bench. Dedup: min-direction
# P(entail) (bidirectional entailment) at 0.85 scored AUC 0.983 with 1 false
# positive, vs 16-21 false blocks for the raw vec-distance blocker.
NLI_CONTRA_THRESHOLD = float(os.environ.get("MNEMOS_NLI_CONTRA_THRESHOLD", "0.98"))
NLI_DEDUP_THRESHOLD = float(os.environ.get("MNEMOS_NLI_DEDUP_THRESHOLD", "0.85"))
NLI_DEDUP_MAX_CANDIDATES = int(os.environ.get("MNEMOS_NLI_DEDUP_MAX_CANDIDATES", "3"))

# --- ONNX Runtime memory arena ---
# ONNX Runtime's default CPU memory arena grows with use and never shrinks
# while a session stays loaded, so RSS climbs during active periods and the
# idle reaper never gets a window on a busy host (reported from a 7.3GB
# system: 700MB -> 4.8GB RSS, OOM + swap; contributed by balaianu/Mnemos).
# MNEMOS_DISABLE_MEM_ARENA=1 passes enable_cpu_mem_arena=False to every
# ONNX session Mnemos creates (e5 embedder, Jina reranker, both NLI
# scorers): each inference allocates from the system and returns it.
# Trade-off ~10-15% slower inference for bounded RSS. Opt-in, default 0,
# same contract as MNEMOS_MODEL_IDLE_TTL and MNEMOS_MIN_FREE_MB.
DISABLE_MEM_ARENA = os.environ.get("MNEMOS_DISABLE_MEM_ARENA", "0") == "1"

# Dedup confirm tier: 'rerank' (legacy cross-encoder / vec fallback) or 'nli'.
# Read at call time so tests and long-lived processes can flip it via env.
DEDUP_CONFIRM_DEFAULT = "rerank"

# Nyx phase-4 contradiction finder. 'cosine' keeps the legacy similarity
# band; 'nli' drops the band ceiling (near-identical pairs are where real
# contradictions live) and scores candidate pairs with the line-level NLI
# finder, recall-first, before the LLM judge.
NYX_CONTRADICT_FINDER_DEFAULT = "cosine"
NLI_FINDER_THRESHOLD = float(os.environ.get("MNEMOS_NLI_FINDER_THRESHOLD", "0.8"))
NLI_FINDER_MAX_PAIRS = int(os.environ.get("MNEMOS_NLI_FINDER_MAX_PAIRS", "200"))

# Phase-4 cosine gates (moved here from consolidation/phases.py so tunables
# live in one place). Floor applies in both finder modes; ceiling only in
# legacy cosine mode.
CONTRADICT_MIN_SIM = float(os.environ.get("MNEMOS_CONTRADICT_MIN_SIM", "0.60"))
CONTRADICT_MAX_SIM = float(os.environ.get("MNEMOS_CONTRADICT_MAX_SIM", "0.85"))

# --- Zero-LLM daily cycle (v10.17.0). Evidence: benchmarks/weave-bench
# (phase-2 NLI gate validated on production clusters) and
# benchmarks/merge-bench (mechanical merge 24/25 exact recovery; the one
# semantic false-duplicate at 0.851 sets the 0.90 tau; short enumerated
# list lines are the other failure class, hence the exact-only floor). ---
# Phase 2 merge engine: 'mechanical' (line-union NLI dedup, selection only,
# provable preservation) or 'llm' (10.16 generative path). Read at call
# time (mirror DEDUP_CONFIRM_DEFAULT) so tests and long-lived processes
# can flip via env.
MERGE_ENGINE_DEFAULT = "mechanical"
MECH_MERGE_TAU = float(os.environ.get("MNEMOS_MECH_MERGE_TAU", "0.90"))
MECH_MERGE_MIN_LINE_CHARS = int(os.environ.get(
    "MNEMOS_MECH_MERGE_MIN_LINE_CHARS", "25"))
# Phase 2 cluster admission gate: members must share at least one
# line-level bidirectional-entailment fact or they are ejected before any
# merge. 'nli' or 'off' (legacy pass-through), read at call time.
CLUSTER_GATE_DEFAULT = "nli"
CLUSTER_GATE_TAU = float(os.environ.get("MNEMOS_CLUSTER_GATE_TAU", "0.70"))
CLUSTER_GATE_MAX_LINES = int(os.environ.get(
    "MNEMOS_CLUSTER_GATE_MAX_LINES", "8"))
# Phase 2 candidacy: 'mutual-topk' (rank-based, immune to the compressed
# e5 cosine space where 45% of all pairs clear 0.78) or 'threshold'
# (legacy absolute cutoffs). Read at call time.
CANDIDACY_DEFAULT = "mutual-topk"
CANDIDACY_TOP_K = int(os.environ.get("MNEMOS_CANDIDACY_TOP_K", "3"))
# Phase 3 insight novelty gate: a bridge insight entailed by either source
# alone is a restatement; the link is kept, the insight memory is not.
WEAVE_NOVELTY_TAU = float(os.environ.get("MNEMOS_WEAVE_NOVELTY_TAU", "0.85"))
# Phase 4 judge: 'llm' (judge immediately), 'queue' (record
# contradiction-candidate links for a later LLM-tier run), 'auto'
# (llm when an LLM is configured, queue otherwise). Read at call time.
CONTRADICT_JUDGE_DEFAULT = "auto"

# --- Nyx cycle tunables (v10.15.1: centralized from consolidation/phases.py;
# this file is the single settings surface for every tunable in the package) ---
# Phase 2 clustering
TIGHT_THRESHOLD = float(os.environ.get("MNEMOS_TIGHT_THRESHOLD", "0.88"))   # Tier 1A near-duplicate dedup
TOPIC_THRESHOLD = float(os.environ.get("MNEMOS_TOPIC_THRESHOLD", "0.75"))   # Tier 1B same-topic merge
# Phase 3 weave
WEAVE_MIN_SIMILARITY = float(os.environ.get("MNEMOS_WEAVE_MIN_SIMILARITY", "0.55"))
WEAVE_TOP_K = int(os.environ.get("MNEMOS_WEAVE_TOP_K", "3"))
# Phase 5 synthesis packet size
NYX_PACKET_SIZE = int(os.environ.get("MNEMOS_NYX_PACKET_SIZE", "25"))
# LLM call budgets per run
NORMAL_MAX_CALLS = int(os.environ.get("MNEMOS_NORMAL_MAX_CALLS", "30"))
SURGE_MAX_CALLS = int(os.environ.get("MNEMOS_SURGE_MAX_CALLS", "80"))
SURGE_THRESHOLD = int(os.environ.get("MNEMOS_SURGE_THRESHOLD", "50"))       # new-memory count that triggers surge mode

# --- Ingest tunables (centralized from ingest.py) ---
INGEST_CHUNK_CHARS = int(os.environ.get("MNEMOS_INGEST_CHUNK_CHARS", "2000"))
INGEST_DEFAULT_PROJECT = os.environ.get("MNEMOS_INGEST_DEFAULT_PROJECT", "ingested")
INGEST_MAX_READ_BYTES = int(os.environ.get(
    "MNEMOS_INGEST_MAX_READ_BYTES", str(50 * 1024 * 1024)))  # larger files should be chunked

# --- Default valid enums (these are conventions, not enforced by storage) ---
# Projects are free-form strings. The list below is just a sensible starter
# set; add or remove categories to match how you organize your memory. The
# storage layer does not enforce membership.
DEFAULT_PROJECTS = {
    "dev", "finance", "food", "health", "home", "personal",
    "relationships", "server", "travel", "work", "writing"
}
VALID_TYPES = {"fact", "decision", "learning", "preference", "todo"}
VALID_LAYERS = {"episodic", "semantic"}

# --- Consolidation ---
SKIP_IMPORTANCE = 9  # skip memories with importance >= this from consolidation
MAX_CLUSTERS_PER_RUN = 10
MAX_CLUSTER_SIZE = 8

# --- Embedding model ---
FASTEMBED_MODEL = os.environ.get("MNEMOS_EMBED_MODEL", "intfloat/multilingual-e5-large")
FASTEMBED_DIMS = 1024
FASTEMBED_CACHE = os.environ.get(
    "MNEMOS_EMBED_CACHE",
    os.path.expanduser("~/.cache/fastembed")
)

# --- Reranker model ---
RERANKER_MODEL = os.environ.get(
    "MNEMOS_RERANKER_MODEL",
    "jinaai/jina-reranker-v2-base-multilingual"
)

# --- Reranker enable flag ---
# Default ON: the cross-encoder reranker is part of the canonical Mnemos
# pipeline and the configuration the benchmark numbers are reported on.
# Disable by exporting MNEMOS_ENABLE_RERANK=0 in your environment, only if
# you are running on sub-1 GB / Pi-class hardware where the extra ~500 MB
# of RAM is unaffordable. Single source of truth: both core.py and
# mcp_server.py read this constant, so disabling here disables it everywhere.
DEFAULT_ENABLE_RERANK = os.environ.get(
    "MNEMOS_ENABLE_RERANK", "1"
).lower() in ("1", "true", "yes", "on")

# --- Retrieval logging (opt-in) ---
# When enabled, every successful search records (query, returned memory IDs)
# to a `retrieval_log` table. This feeds benchmark generation and retrieval
# quality analysis over real queries. Default off for privacy: queries may
# contain sensitive content, and users should opt in consciously.
#   export MNEMOS_RETRIEVAL_LOG=1   # enable
DEFAULT_RETRIEVAL_LOG = os.environ.get(
    "MNEMOS_RETRIEVAL_LOG", "0"
).lower() in ("1", "true", "yes", "on")

# --- Tool usage logging (opt-in, minimal diagnostic telemetry) ---
# When enabled, every MCP tool call records (tool_name, timestamp) to a
# `tool_usage` table. Purpose: health-check tooling can answer "has the MCP
# server been responsive?" without parsing stdin/stdout logs. Cost: one
# INSERT per tool call, bytes per day of storage. Contains no query text,
# no memory content, no IDs - only the tool name + when it was called.
# Default off for consistency with retrieval_log, though privacy footprint
# is essentially zero since no user content is captured.
#   export MNEMOS_TOOL_USAGE_LOG=1   # enable
DEFAULT_TOOL_USAGE_LOG = os.environ.get(
    "MNEMOS_TOOL_USAGE_LOG", "0"
).lower() in ("1", "true", "yes", "on")

# --- Storage paths ---
DEFAULT_DB_PATH = os.environ.get(
    "MNEMOS_DB",
    os.path.expanduser("~/.mnemos/memory.db")
)
DEFAULT_NAMESPACE = "default"

# --- CML mode ---
# "on"  (default): MCP tool description teaches CML, Nyx cycle cemelifies on
#                  merge and uses CML-prefixed insights on synthesis, dedup
#                  matches on CML subject prefixes, the whole system assumes
#                  CML as the default storage convention.
# "off": prose-only deployment. MCP tool description stops teaching CML,
#        Nyx cycle merges and synthesis prompts instruct prose output,
#        dedup falls back to FTS + vector signals only (no CML-subject
#        branch). Single coordinated flag; flip everything at once.
CML_MODE = os.environ.get("MNEMOS_CML_MODE", "on").lower()
if CML_MODE not in ("on", "off"):
    CML_MODE = "on"

# --- SQL templates (pre-built from constants above, used by SQLiteStore) ---
DAYS_AGE_SQL = "julianday('now','localtime') - julianday(COALESCE(m.last_accessed, m.created_at))"

TEMPORAL_DECAY_SQL = f"""(CASE WHEN m.tags LIKE '%evergreen%' THEN 1.5
    WHEN m.layer = 'semantic' THEN 1.5 * MAX({DECAY_FLOOR}, exp(-{DECAY_RATE_SEMANTIC} * {DAYS_AGE_SQL}))
    ELSE 1.5 * MAX({DECAY_FLOOR}, exp(-{DECAY_RATE} * {DAYS_AGE_SQL})) END)"""

_CONFIRM_DAYS = "julianday('now','localtime') - julianday(m.last_confirmed)"
CONFIRMATION_BOOST_SQL = f"""(CASE
    WHEN m.last_confirmed IS NULL THEN 0
    WHEN {_CONFIRM_DAYS} <= 30 THEN {CONFIRM_BOOST_30D}
    WHEN {_CONFIRM_DAYS} <= 90 THEN {CONFIRM_BOOST_90D}
    ELSE 0 END)"""

BM25_CALL = f"bm25(memories_fts, {BM25_WEIGHTS[0]}, {BM25_WEIGHTS[1]}, {BM25_WEIGHTS[2]})"

RANKING_ORDER_SQL = f"""ORDER BY ({BM25_CALL}
    - (m.importance * {IMPORTANCE_BOOST})
    - (MIN(m.access_count, {ACCESS_CAP}) * {ACCESS_BOOST})
    - {TEMPORAL_DECAY_SQL}
    - {CONFIRMATION_BOOST_SQL})"""
