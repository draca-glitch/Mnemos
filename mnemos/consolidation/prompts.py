"""
LLM prompts for the Nyx Cycle consolidation system.

Separated from logic for easy tuning. Two variants exist for merge and
synthesis, picked at call time by MNEMOS_CML_MODE:
  "on"  (default): CML-output prompts (MERGE_SYSTEM, SYNTHESIS_SYSTEM)
  "off":           prose-output prompts (MERGE_SYSTEM_PROSE,
                   SYNTHESIS_SYSTEM_PROSE), same preservation rules, no
                   CML prefixes or relation symbols in the output

Classification prompts (WEAVE_SYSTEM, CONTRADICT_SYSTEM, TRIAGE_SYSTEM)
emit structured labels, not CML, and are unaffected by the mode switch.

CML notation key:
  D: decision, C: contact, F: config/fact, L: learning, P: preference, W: warning
  Symbols: → ↔ ← ∵ ∴ △ ⚠ @ ✓ ✗ ~ ∅ … ; > #N
"""

# --- Phase 1: Dedup Merge ---

MERGE_SYSTEM = """You consolidate memories into one CML block. Your job is UNIQUE INFORMATION PRESERVATION with dense formatting.

THE ONE RULE YOU MUST NOT BREAK:
Every unique piece of information from any input memory must survive in your output. If a detail exists in ANY input and does NOT exist in your output, you have failed the merge. Overlap between memories (same fact stated twice) can be collapsed to one line. Content unique to a single input memory must NEVER be dropped, ever, for any reason, regardless of how minor it seems.

Before emitting your output, audit yourself: for each input memory, can you point to every distinct fact, specific, or detail it contained and show where it appears in the output? If not, add what's missing.

WHAT TO PRESERVE (non-negotiable):
- Every name (person, service, product, tool, place)
- Every number, amount, quantity, version, measurement
- Every date, time, duration, deadline
- Every path, URL, IP, port, ID, identifier
- Every decision and its recorded reason
- Every distinct event or transaction (3 separate events → 3 items listed, not "some events")
- Every contact detail
- Every stated preference
- Every configuration value
- Every distinct nuance, caveat, or qualifier

WHAT TO COMPRESS (shrink these):
- Prose structure, sentences, transitions, narrative glue
- Redundant phrasing between overlapping memories (same fact twice = one line)
- Reasoning-history that no longer matters, meta-commentary
- Explanatory filler around a fact (keep the fact, drop the explanation)

SIZE: output should be shorter than the concatenated inputs (prose compresses well), but grow with cluster size is expected, 6 memories of distinct content cannot fit into 1 memory-sized output without data loss. Do not truncate to hit any arbitrary size.

CONFLICTS: most recent value wins, use △ to mark changed state and briefly note prior value.

CML notation. Prefixes: D: decision, C: contact, F: config/fact, L: learning, P: preference, W: warning. Symbols: → ↔ ← ∵ ∴ △ ⚠ @ ✓ ✗ ~ ∅ … ; > #N (∵=because, ∴=therefore, ~=approximate/uncertain, …=continuation)

Use ; to chain facts densely on one line when topic is shared. Newline for distinct topics. Pack more facts per line than per memory.

Output ONLY the CML text. No explanations, no fencing, no preamble, no self-commentary."""


MERGE_SYSTEM_PROSE = """You consolidate memories into one paragraph of clear English prose. Your job is UNIQUE INFORMATION PRESERVATION with natural writing.

THE ONE RULE YOU MUST NOT BREAK:
Every unique piece of information from any input memory must survive in your output. If a detail exists in ANY input and does NOT exist in your output, you have failed the merge. Overlap between memories (same fact stated twice) can be collapsed to one sentence. Content unique to a single input memory must NEVER be dropped, ever, for any reason, regardless of how minor it seems.

Before emitting your output, audit yourself: for each input memory, can you point to every distinct fact, specific, or detail it contained and show where it appears in the output? If not, add what's missing.

WHAT TO PRESERVE (non-negotiable):
- Every name (person, service, product, tool, place)
- Every number, amount, quantity, version, measurement
- Every date, time, duration, deadline
- Every path, URL, IP, port, ID, identifier
- Every decision and its recorded reason
- Every distinct event or transaction (3 separate events means 3 items mentioned, not "some events")
- Every contact detail
- Every stated preference
- Every configuration value
- Every distinct nuance, caveat, or qualifier

WHAT TO COMPRESS (shrink these):
- Redundant phrasing between overlapping memories (same fact twice = one sentence)
- Reasoning-history that no longer matters, meta-commentary
- Explanatory filler around a fact (keep the fact, drop the explanation)

SIZE: output should be shorter than the concatenated inputs (redundancy compresses well), but growth with cluster size is expected. 6 memories of distinct content cannot fit into 1 memory-sized output without data loss. Do not truncate to hit any arbitrary size.

CONFLICTS: most recent value wins, briefly note the prior value.

Write in natural clear English prose. Do NOT use CML prefixes (D:, C:, F:, L:, P:, W:) or relation symbols (→, ∵, ∴, △, ⚠, ↔). Output ONLY the prose text. No explanations, no fencing, no preamble, no self-commentary."""

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

SYNTHESIS_SYSTEM = """You are performing "Nyx consolidation" on a person's memory corpus.

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


SYNTHESIS_SYSTEM_PROSE = """You are performing "Nyx consolidation" on a person's memory corpus.

You see memories spanning multiple life domains. Health, relationships, work, technology, philosophy, writing, finances, etc. Your job is to notice what the person themselves might not have noticed: patterns, recurring themes, behavioral tendencies, tensions, and cross-domain connections.

Generate 2-4 concise insights. Each MUST:
- Connect at least 2 different categories (e.g., work patterns reflecting in relationships)
- Be genuinely novel, not a restatement of any single memory
- Be framed as an observation, not advice
- Be written as a single clear English sentence (no CML prefixes, no relation symbols)
- Include which memory IDs informed the insight, in parentheses

Separate insights with a blank line so the parser can detect them.

Example output (illustrative only, with placeholder memory IDs):
Pattern: technical preference for explicit configuration (#101, #142) mirrors decision-making style in personal habits (#118); both may stem from a preference for low-ambiguity systems.

Tension: stated preference for minimalism in tooling (#205) while accumulating broad infrastructure responsibilities (#211); unresolved question of whether the minimalism is aspirational or actual.

You may receive a MNEMOS_USER_PROFILE description of who the user is at the end of this prompt to help you reason about patterns. If no profile is given, infer cautiously and prefer observations over claims.

Output ONLY the prose insights separated by blank lines. No preamble, no markdown, no CML prefixes."""

# --- Phase 0: Triage ---

TRIAGE_SYSTEM = """You classify memories into rough topic clusters for efficient processing.

For each memory, assign:
1. A topic tag (2-3 words, e.g., "health-supplements", "server-config", "relationship-pattern")
2. A priority: HIGH (contains decisions, unique insights, personal revelations) or LOW (routine facts, configs, logs)

Respond as JSON array:
[{"id": 1, "topic": "health-supplements", "priority": "HIGH"}, ...]

Be concise. This is preprocessing, not analysis."""
