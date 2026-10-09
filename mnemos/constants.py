"""
Centralized configuration for Mnemos.

Single source of truth for all tuning constants, ranking formulas, and
thresholds. Override via environment variables prefixed with MNEMOS_.
"""

import os
import re

# --- Temporal decay ---
# Episodic (time-bound events): half-life ≈ 46 days
# Semantic (distilled knowledge): half-life ≈ 180 days
DECAY_RATE = 0.015           # ln(2)/46 ≈ 0.015
DECAY_RATE_SEMANTIC = 0.00385  # ln(2)/180 ≈ 0.00385
DECAY_FLOOR = 0.1  # old memories never drop below 10% boost

# --- Search behavior ---
HYBRID_MIN_MEMORIES = 10  # vector search joins FTS at this count of active memories
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

# --- Link output ---
# Nyx bookkeeping relations: they record that a pair was judged, not that
# the memories relate. On a mature store they outnumber semantic links
# roughly 4:1, so search output drops them unless the caller opts in.
AUDIT_LINK_RELATIONS = frozenset({"contradiction-cleared"})

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

# The legacy rerank-confirm tiers (dedup above, contradiction bands below)
# sigmoid a raw cross-encoder logit and compare it to these constants, and
# logit scales are per-model: the numbers were tuned on jina-v2. Measured on
# gte-modernbert (the v10.33.0 default), identical pairs score 0.92-0.98 and
# UNRELATED pairs median 0.68 with outliers past 0.85, and the identical and
# related ranges OVERLAP, so no threshold separates them: the sigmoid tier is
# structurally unsuited to that model, not merely mistuned. The store path
# therefore only uses the rerank tier on models it was calibrated for, and
# otherwise prefers NLI when available, else the vec-distance tier.
RERANK_CONFIRM_CALIBRATED = ("jina-reranker-v2",)


def rerank_confirm_calibrated(model_id=None):
    m = (model_id if model_id is not None else RERANKER_MODEL).lower()
    return any(t in m for t in RERANK_CONFIRM_CALIBRATED)

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
# MNEMOS_DISABLE_MEM_ARENA passes enable_cpu_mem_arena=False to every ONNX
# session Mnemos creates (embedder, reranker, both NLI scorers): each
# inference allocates from the system and returns it.
# Opt-in, default 0, same contract as MNEMOS_MODEL_IDLE_TTL and
# MNEMOS_MIN_FREE_MB. Strongly recommended for any long-lived process: the
# arena is unbounded in practice, not merely large. Measured on one host,
# same workload: 21.2 GB RSS after ~25 minutes with the arena on and no
# shrink when idle, against 2.85 GB flat over 3.5 hours with it off. The
# reported 700MB -> 4.8GB OOM is the same failure on smaller hardware.
# It is not the default because it is not free: measured at +49% embedding
# latency (interleaved A/B, 4 paired rounds), far above the 10-15% previously
# assumed here. Short-lived batch processes that exit before growth matters
# should leave it off and keep the throughput.
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
# An absolute floor is a property of the embedding model, not of the corpus,
# so no constant serves bge-small, e5-large and gte alike. 0.60 was calibrated
# on a space that is not e5's: measured on a 746-row e5-large store, the
# same-project cosines span [0.74, 0.97] with a median of 0.86, so 0.60 admits
# 100.0% of 65,960 pairs and phase 4 degenerates into an exhaustive scan.
# Self-calibrate instead: take the floor from the store's own distribution, so
# the gate keeps its selectivity across any embedder. Setting
# MNEMOS_CONTRADICT_MIN_SIM pins an absolute floor and turns calibration off.
CONTRADICT_MIN_SIM_EXPLICIT = "MNEMOS_CONTRADICT_MIN_SIM" in os.environ
CONTRADICT_SIM_PERCENTILE = float(
    os.environ.get("MNEMOS_CONTRADICT_SIM_PERCENTILE", "99.0"))
# Never let calibration admit fewer than this many pairs on a small store,
# where a high percentile of few pairs is noise rather than selection.
CONTRADICT_MIN_CANDIDATES = int(
    os.environ.get("MNEMOS_CONTRADICT_MIN_CANDIDATES", "20"))

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
# DEFAULT SWITCHED to the light tier in v10.33.0 (benchmarked 2026-08-19,
# see docs/usage.md model tiers and issue #4). bge-small ties e5-large on
# LongMemEval R@1 92.40 vs 92.01 WITH the reranker in the pipeline, at 33M
# params against 560M. The reranker is what makes this safe: without one,
# e5 is clearly better (R@1 88.43 vs 81.29), so do not pair this default
# with MNEMOS_ENABLE_RERANK=0 on quality-sensitive stores.
# Populated stores are unaffected: they pin to the model their vectors were
# built with (v10.30.0) until their owner runs `mnemos reembed`.
# Multilingual stores should use the multilingual tier:
#   MNEMOS_EMBED_MODEL=intfloat/multilingual-e5-large MNEMOS_EMBED_DIMS=1024
#   MNEMOS_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
FASTEMBED_MODEL = os.environ.get("MNEMOS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_MODEL_EXPLICIT = "MNEMOS_EMBED_MODEL" in os.environ
# Dimensionality of FASTEMBED_MODEL. Must match the model: e5-large and
# bge-large are 1024, bge-base 768, bge-small and MiniLM 384. Changing
# either this or the model requires a full re-embed. A mismatch does not
# corrupt anything silently: sqlite-vec rejects the insert ("Expected 1024
# dimensions but received 384") and embed() raises first with the value to
# set, but the store is left without vectors until it is corrected.
FASTEMBED_DIMS = int(os.environ.get("MNEMOS_EMBED_DIMS", "384"))
EMBED_DIMS_EXPLICIT = "MNEMOS_EMBED_DIMS" in os.environ
# --- CML operator normalization (opt-in) ---
# CML's relational operators are unicode symbols. SentencePiece vocabularies
# (e5, Jina reranker: 250k) carry most of them (not all: even 250k XLM-R
# lacks the because-operator and the FTS5 snippet markers, see
# rerank._probe_cml_support); English WordPiece vocabularies
# (bge/gte/mxbai/nomic/MiniLM: 30522) map ALL of them to a single
# shared [UNK]. Measured on a 720-memory production store: 901 UNK tokens, and
# because [UNK] has one embedding, "migration confirmed" and "migration
# rejected" become the same vector.
# With this on, the operators are replaced by the words they stand for BEFORE
# tokenization, for the EMBEDDER ONLY. Stored content, FTS5, the reranker and
# what the agent reads back are all untouched. Cost measured at +0.13% tokens.
# DEFAULT ON since v10.30.0. Safe to default because it only seeds NEW stores:
# a populated store pins itself to the provenance its vectors were built with
# (see embed.adopt_store_config), so upgrading does not silently re-interpret
# an existing index. To migrate an existing store, export
# MNEMOS_EMBED_NORMALIZE_CML=1 explicitly (an exported value outranks the
# store) and run `mnemos reembed`. Set it to 0 to disable.
EMBED_NORMALIZE_CML = os.environ.get("MNEMOS_EMBED_NORMALIZE_CML", "1") == "1"
# An explicitly set knob is a deliberate instruction and outranks whatever a
# populated store was built with; an unset one is only a seed for new stores.
EMBED_NORMALIZE_EXPLICIT = "MNEMOS_EMBED_NORMALIZE_CML" in os.environ

FASTEMBED_CACHE = os.environ.get(
    "MNEMOS_EMBED_CACHE",
    os.path.expanduser("~/.cache/fastembed")
)

# --- Reranker model ---
# gte-modernbert (149M) BEATS jina-v2 (278M) on the discriminating LongMemEval
# categories at half the size (preference R@1 80.00 vs 73.33). It is English-
# only (byte-level BPE, so CML operators fragment into byte pieces rather
# than [UNK]; the load-time probe detects that and spells them out, and maps
# the query to match); unlike the embedder, the reranker is NOT pinned per store (it touches
# no stored data), so existing multilingual deployments must set
# MNEMOS_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual explicitly
# or their non-English scoring degrades on the next restart. doctor and the
# store path warn when that combination meets non-English content.
# Not in fastembed's built-in catalogue: rerank.py registers it at load.
RERANKER_MODEL = os.environ.get(
    "MNEMOS_RERANKER_MODEL",
    "Alibaba-NLP/gte-reranker-modernbert-base"
)
# Explicit env is an instruction and outranks a store-declared preference;
# unset means the default is only a seed, same contract as the embedder vars.
RERANKER_EXPLICIT = "MNEMOS_RERANKER_MODEL" in os.environ

# --- Reranker / embedder tensor shape bounds ---
# The ONNX Runtime CPU arena keeps the peak allocation of every tensor SHAPE it
# has ever seen and never returns it. Shape here is (batch, sequence length),
# and left alone both axes are free-running: fastembed defaults to batch 64 for
# reranking and 256 for embedding, and nothing bounded sequence length at all,
# so a wide search over long memories set a new high-water mark that the
# process then held forever. Measured on a shared long-lived server: one
# limit=20 search (a 60-document rerank pool) claimed 1.6 GB permanently, while
# an identical-shape repeat cost 3% of that.
# Bounding both axes collapses shape space to a small fixed set, so the arena
# stops finding new highs and the RSS ceiling becomes a real ceiling rather
# than a running maximum. This matters most under the shared HTTP transport
# (v10.27.0): stdio harnesses used to reset the arena for free every time a
# session ended, and a server that never exits removed that accident.
# Set MAX_CHARS to 0 to disable truncation.
RERANK_BATCH = int(os.environ.get("MNEMOS_RERANK_BATCH", "16"))
# The sequence axis is NOT unbounded, which is the easy thing to get wrong:
# Jina v2 caps at 1024 tokens and fastembed truncates there. What is free is
# the PADDING. fastembed calls enable_padding() with no length, so every batch
# pads to its own longest member, and one long memory in a batch of 16 drags
# the other 15 up with it. Shape therefore varies continuously in [1, 1024]
# with batch composition, which is what the arena keeps finding new highs in.
# Pinning truncation AND padding to one length collapses that to a single
# shape. 512 rather than 1024 because attention is quadratic: (512/1024)^2 is
# a quarter of the peak. Set to 0 to leave fastembed's dynamic padding alone.
RERANK_MAX_TOKENS = int(os.environ.get("MNEMOS_RERANK_MAX_TOKENS", "512"))
# Superseded by the token pin above and kept only as an escape hatch: clipping
# characters is a weak proxy for clipping tokens, and on a store of CML the
# ratio varies enough that a char budget does not bound the tensor.
RERANK_MAX_CHARS = int(os.environ.get("MNEMOS_RERANK_MAX_CHARS", "0"))
EMBED_BATCH = int(os.environ.get("MNEMOS_EMBED_BATCH", "32"))

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

# --- Self memory (v10.42.0) ---
# An agent's memories about itself (traits it claims, habits it has observed,
# commitments about its own behaviour) live in their own namespace,
# "self:<agent>", beside the user's store in the same database, so backup,
# move and the Nyx cycle (run with MNEMOS_NAMESPACE=self:<agent>) cover them.
# Off by default: the MCP `self` flag and the CLI `--self` switch refuse until
# MNEMOS_SELF is on, and turning it off hides the namespace without deleting a
# row. The agent is the MCP client's declared clientInfo.name; MNEMOS_AGENT
# names the CLI and is the fallback for a client that declares no name.
#   export MNEMOS_SELF=1
#   export MNEMOS_AGENT=claude-code   # CLI identity / fallback
DEFAULT_SELF = os.environ.get(
    "MNEMOS_SELF", "0"
).lower() in ("1", "true", "yes", "on")
DEFAULT_AGENT = os.environ.get("MNEMOS_AGENT") or None
SELF_NAMESPACE_PREFIX = "self:"
_AGENT_SLUG = re.compile(r"[^a-z0-9._-]+")


def agent_slug(name: str) -> str:
    """Namespace-safe form of a client name: lower case, runs of anything
    outside [a-z0-9._-] become one dash, trimmed, at most 40 chars."""
    slug = _AGENT_SLUG.sub("-", (name or "").strip().lower()).strip("-._")
    return slug[:40]


def self_namespace(agent: str) -> str:
    slug = agent_slug(agent)
    if not slug:
        raise ValueError("agent name is empty after sanitising")
    return SELF_NAMESPACE_PREFIX + slug

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
