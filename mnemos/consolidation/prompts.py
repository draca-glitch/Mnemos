"""
LLM prompts for the Dream Cycle consolidation system.

Separated from logic for easy tuning. All prompts use CML notation
and are designed for Claude Opus via DO Gradient API.

CML notation key:
  D: decision, C: contact, F: config/fact, L: learning, P: preference, W: warning
  Symbols: → ↔ ← ∵ ∴ △ ⚠ @ ✓ ✗ ~ ∅ … ; > #N
"""

# --- Phase 1: Dedup Merge ---

MERGE_SYSTEM = """You consolidate multiple memories into one.

CRITICAL SIZE RULE: The output must be roughly the size of ONE of the input memories, NOT the sum of all of them. 2 memories in = 1 memory-sized output. 4 memories in = still 1 memory-sized output. If you're producing something close to the combined input length, you're concatenating, not consolidating. That is wrong.

These memories were clustered because they're semantically similar. Extract what matters, merge overlapping facts into single statements, and discard the rest. The originals are archived (never deleted). aggressive compression is safe.

The reader is an AI agent starting a fresh session. It needs enough to orient and act, not a complete record.

CML notation. Type prefixes: D: decision, C: contact, F: config/fact, L: learning, P: preference, W: warning. Symbols: → ↔ ← ∵ ∴ △ ⚠ @ ✓ ✗ ~ ∅ … ; > #N (∵=because, ∴=therefore, ~=approximate/uncertain, …=continuation)

KEEP: concrete data (names, paths, IDs, config values), current state, decisions (one-phrase reason)
DROP: overlap between memories (merge once), superseded states, reasoning/history, detail that lives in referenced files
CONFLICTS: most recent wins, mark with △

Output ONLY the CML text. No explanations, no markdown."""

# --- Phase 2: Thematic Weaving ---

WEAVE_SYSTEM = """You analyze pairs of memories from different life domains to find genuine cross-domain connections.

You will see two memories from different categories (e.g., health + relationships, work + personal, dev + writing). Your job is to determine if there is a GENUINE thematic connection. not superficial topic overlap, but a meaningful pattern, cause-effect, behavioral through-line, or shared underlying dynamic.

Examples of genuine connections:
- Sleep patterns → energy levels → workout consistency
- Control-seeking in server infrastructure → control-seeking in personal habits
- Career change motivation → underlying values about meaningful work
- Writing themes (a recurring topic in personal notes) → actual beliefs revealed through actions

Examples of NOT genuine connections:
- Both mention the same city (geographic coincidence)
- Both were created in the same month (temporal coincidence)
- Both contain technical details (format similarity)

If connected, respond EXACTLY in this format:
LINK_TYPE: <one of: evolves|informs|contradicts|enables|reflects>
STRENGTH: <0.3 to 1.0>
INSIGHT: <one sentence describing the connection. this may be stored as a new memory>

Link types:
- evolves: A developed into or led to B over time
- informs: A provides context that changes how B should be interpreted
- contradicts: A and B are in tension or state incompatible things
- enables: A is a prerequisite or enabler of B
- reflects: A in domain X mirrors a pattern in B from domain Y

If NOT genuinely connected, respond exactly: NO_LINK"""

# --- Phase 3: Contradiction Scan ---

CONTRADICT_SYSTEM = """You compare two memories on the same topic from different times to detect evolution, contradiction, or supersession.

These memories are from the SAME category and are about similar topics. Classify their relationship:

SUPERSEDED. the newer memory completely replaces the older (same information, updated values/state)
EVOLVED. a decision or understanding genuinely changed over time (the reasoning or conclusion shifted)
CONTRADICTS. they state incompatible things and it's unclear which is current
COMPATIBLE. they're about the same topic but don't actually conflict

Respond EXACTLY in this format:
<SUPERSEDED|EVOLVED|CONTRADICTS|COMPATIBLE>|<one-line explanation>

Examples:
SUPERSEDED|Database engine changed from Postgres 14 to Postgres 16; old version specifics are obsolete
EVOLVED|Initial deployment used a single region; now multi-region with reasoning shifted toward redundancy over latency
CONTRADICTS|Memory #42 says weekly backup; #89 says daily backup; unclear which is current
COMPATIBLE|Both describe the same server setup from different angles"""

# --- Phase 4: Insight Synthesis ---

SYNTHESIS_SYSTEM = """You are performing "dream consolidation" on a person's memory corpus.

You see memories spanning multiple life domains. health, relationships, work, technology, philosophy, writing, finances, etc. Your job is to notice what the person themselves might not have noticed: patterns, recurring themes, behavioral tendencies, tensions, and cross-domain connections.

Generate 2-4 concise insights. Each MUST:
- Connect at least 2 different categories (e.g., work patterns reflecting in relationships)
- Be genuinely novel. not a restatement of any single memory
- Be framed as an observation, not advice
- Use CML notation: L: prefix for learnings
- Include which memory IDs informed the insight

Example output (illustrative only, with placeholder memory IDs):
L: Pattern: technical preference for explicit configuration (#101, #142) mirrors decision-making style in personal habits (#118) → both may stem from a preference for low-ambiguity systems
L: Tension: states a clear preference for minimalism in tooling (#205) while accumulating broad infrastructure responsibilities (#211) → unresolved question of whether the minimalism is aspirational or actual

You may receive a `MNEMOS_USER_PROFILE` description of who the user is at the
end of this prompt to help you reason about patterns. If no profile is given,
infer cautiously and prefer observations over claims.

Output ONLY the CML insights. No preamble, no markdown."""

# --- Phase 0: Triage ---

TRIAGE_SYSTEM = """You classify memories into rough topic clusters for efficient processing.

For each memory, assign:
1. A topic tag (2-3 words, e.g., "health-supplements", "server-config", "relationship-pattern")
2. A priority: HIGH (contains decisions, unique insights, personal revelations) or LOW (routine facts, configs, logs)

Respond as JSON array:
[{"id": 1, "topic": "health-supplements", "priority": "HIGH"}, ...]

Be concise. This is preprocessing, not analysis."""
