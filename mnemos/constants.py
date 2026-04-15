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
# Tightened 2026-04-09 after autoimprove iteration on a 15-pair eval set:
# baseline (vec 0.60, rr 0.50) scored F1=0.500 with 1 FP and 3 FNs; winning
# config (vec 0.35, rr 0.35, require_same_project=True) scored F1=1.0 with
# 0 FPs and 0 FNs. Tight vec gate + strict same-project filter eliminates
# cross-project topical neighbors that the cross-encoder over-rates.
CONTRADICTION_VEC_THRESHOLD = 0.35
CONTRADICTION_RERANK_THRESHOLD = 0.35

# --- Dedup rerank threshold ---
DEDUP_RERANK_THRESHOLD = 0.70

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
