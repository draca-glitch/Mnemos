<p align="center">
  <img src="assets/mnemos-logo.svg" alt="Mnemos" width="220">
</p>

# Mnemos

> **The last memory you'll ever need.**
>
> A state-of-the-art persistent memory system for AI agents.
> Named after Mnemosyne (Greek: μνήμη, *memory*).

**Runs on any computer. Works with any AI.**

- **CPU-only**: no GPU required. Embeddings and reranking via ONNX models that run on a regular laptop, NAS, Raspberry Pi 4+, or budget VPS. Even the optional consolidation LLM can run locally on CPU in "slow mode" (1-5 tok/s on a 32B model with enough RAM) since the dream cycle is a background job and quality matters more than speed
- **MCP-native**: works with Claude Code, Cursor, ChatGPT Desktop, Gemini, or any MCP-compatible AI client out of the box
- **CLI-friendly**: a full `mnemos` command-line tool ships alongside the MCP server, so you can store, search, ingest, and consolidate from any shell, script, or cron job, with or without an AI client attached
- **100% local**: no API calls, no telemetry, no cloud dependencies. Your memory stays on your machine
- **State-of-the-art**: 98.1% Recall@5 on LongMemEval, competitive with the best published systems

> **Why v10 in a brand-new repository?** Mnemos has been in private production for a long time. Each of the nine internal versions involved real experimentation: adding features, removing the ones that did not pull their weight, evaluating retrieval quality, and iterating on what actually made the system smarter rather than just bigger. v10 is what was running on the day it was benchmarked at 98.1% R@5, and what is shipping here. As for why the repository itself has exactly one commit: I did not use git for this project until I created my first GitHub account on April 10, 2026, the same day I pushed Mnemos. The system has months of history behind it. The repository does not. See [CHANGELOG.md](CHANGELOG.md) for the real timeline.

> **Why "Mnemos"?** The first reason is straightforward: **Mnemosyne** (μνημοσύνη) is the Greek goddess of memory and mother of the nine Muses. Her name literally means "remembrance". A memory system named after her writes itself.
>
> The second reason is a wink. **Mythos** is the rumored name of Anthropic's next Claude model. Mythos tells stories, Mnemos remembers them. Same Greek mythology bench, same family of words, complementary roles: if Mythos becomes the model that powers your AI assistant, Mnemos is the memory it draws from. The naming was already in place before any of that surfaced publicly, but I was not going to pretend the pairing was not too good to keep. To be clear: this project has no affiliation with Anthropic. It is appreciation, not partnership. Just two names from the same root, doing the two things memory and storytelling have always done together.

## Origin: scratching a real itch, then "wait, mine's better"

Mnemos started because the default `memory.md` approach felt deeply inefficient. A flat markdown file that Claude (or any AI) reads on every session is fine for a handful of facts, but it does not scale, it does not rank, it does not forget anything that should be forgotten, and it does not actually let the model *remember the user* in any meaningful way. I wanted my AI assistant to know who I am, what I care about, what I have decided, what I prefer, the way a long-term colleague would. Not just to re-read a static text file at every session start.

So I started building. Over months it grew into a real retrieval system: FTS5 for lexical search, sqlite-vec for semantic search, RRF fusion, a cross-encoder for high-precision dedup, exponential temporal decay, a weekly dream cycle for consolidation. It has been running in private production powering personal AI agents for a long time before this repo existed. **I was never planning to publish it.** It was just my own infrastructure, built for my own use, by someone who got tired of `memory.md`.

Then [MemPalace](https://github.com/milla-jovovich/mempalace) surfaced, claiming state-of-the-art retrieval with a hierarchical memory metaphor. I read their README and benchmarks and had the strangest reaction: *"wait, mine is already better."* Not from arrogance, just from looking at the architecture. They were missing hybrid retrieval, missing reranking, missing decay, missing dedup. So I pointed Mnemos at the same LongMemEval dataset they had used. First run, no tuning: **98.1% Recall@5**.

That result is what tipped this from "private side project I had no plans to share" into "I should clean this up and put it on GitHub for public scrutiny". If you are reading this README, that comparison is the only reason. The honest comparison:

|                              | MemPalace | **Mnemos** |
|---|---|---|
| Storage backend              | ChromaDB only | **SQLite (default), Qdrant, Postgres-ready** |
| Vector search                | ChromaDB defaults | **FastEmbed e5-large (1024-dim, ONNX)** |
| Lexical search (BM25)        | None | **FTS5 with stemming + AND/OR fallback** |
| Hybrid retrieval             | None | **RRF fusion (k=60)** |
| Cross-encoder reranking      | None | **Jina Reranker v2 multilingual** |
| Deduplication                | None | **3-way: FTS + CML + vector → reranker** |
| Temporal decay               | None | **Exponential, separate half-lives for episodic/semantic** |
| Validity windows             | None | **`valid_from` / `valid_until` per fact** |
| Contradiction detection      | None | **Auto-detect on store + memory_links** |
| Hierarchical organization    | Wing/Room metadata | **project/subcategory metadata** |
| MCP tools                    | 19 (manual navigation) | **4 (CRUD + search)** |
| Consolidation                | Mining modes | **5-phase dream cycle (LLM-driven)** |
| Auto-widen on thin results   | None | **Cross-project fallback** |
| **LongMemEval R@5**          | **96.6%** (raw mode, per their README)<br>~3.4% error rate | **98.1%** (hybrid mode, first run, no tuning)<br>~1.9% error rate, **roughly 44% fewer errors** |

Strip the metaphor away and to me MemPalace sounds like a **gimmick layer on top of ChromaDB**. The "wings, rooms, halls, closets, tunnels" terminology is mnemonic theater, not architecture; underneath, it is metadata fields plus tool wrappers around ChromaDB's default metadata filtering. The 19 MCP tools they expose are navigation primitives that force the LLM to manually traverse the palace step by step instead of just running a search. Their headline "+34% palace boost" is, by their own transparency note, what you get from any standard ChromaDB metadata filter.

Their AAAK abbreviation system, which was originally pitched as a token-compression breakthrough, **actually regresses retrieval quality** (84.2% vs 96.6% raw by their own benchmarks) and the original "30x compression" claim was retracted as misleading after launch.

The retrieval gap matters more than the percentages suggest. **Mnemos hybrid 98.1% vs MemPalace raw 96.6%** is a 1.5 percentage point spread on the same dataset, but on a benchmark that already sits in the high nineties, that translates to roughly **44% fewer errors per query** (1.9% vs 3.4% miss rate). That is the actual difference between "ChromaDB with metadata filters" and "a real hybrid retrieval pipeline". Mnemos's 98.1% is also reported on a **first run with no parameter tuning** against the LongMemEval set, which is the most conservative version of the number you can publish.

> A note on terminology: **MemPalace's "raw mode"** is pure ChromaDB vector similarity over verbatim conversation text, the "store everything, search via embeddings" baseline they document as their primary mode and headline benchmark (96.6% R@5). **Mnemos's "hybrid mode"** is FTS5 BM25 + e5-large vector similarity + Reciprocal Rank Fusion, with no cross-encoder rerank in the search path, scored on the same 470-question non-abstention LongMemEval set. If MemPalace ever publishes a hybrid retrieval mode of their own with a verifiable benchmark number, I will update this section.

Mnemos skips the metaphor and ships the retrieval pipeline that actually scores. **Fewer tools, deeper plumbing, more honest defaults.** **An AI does not need a palace, it needs efficiency.**

To be fair: hierarchical organization, temporal validity, and contradiction handling are good ideas. That is why I had them months ago. Project-based hierarchy shipped in Mnemos v6 back in February, and exponential decay, `last_confirmed` recency, and the dream cycle Phase 3 contradiction detector all shipped in v8 a few weeks later. v10 sharpened the terminology (`subcategory`, `valid_from` / `valid_until`, real-time contradiction on store), but my underlying code has been running in production for months.

```
┌─────────────────────────────────────────────────────────────┐
│  System                            R@5     Errors            │
├─────────────────────────────────────────────────────────────┤
│  Mnemos (hybrid)                   98.1%    1.9%  ← this repo│
│  MemPalace raw ChromaDB            96.6%    3.4%             │
│  Dense retrieval (paper baseline)  ~90.0%  ~10%              │
│  BM25 baseline (paper)             ~85.0%  ~15%              │
└─────────────────────────────────────────────────────────────┘
```

## What is Mnemos?

A local, privacy-preserving memory system for Claude Code (and other MCP-compatible AI clients). Mnemos remembers what you've discussed across sessions and surfaces the right context when you need it, without sending anything to the cloud.

Unlike systems that just dump conversations into a vector database, Mnemos:

- **Curates memories** rather than storing verbatim chats: beliefs, decisions, preferences, learnings
- **Combines hybrid retrieval**: FTS5 BM25 + e5-large vectors + Reciprocal Rank Fusion + Jina cross-encoder reranking
- **Models human forgetting**: exponential temporal decay with separate half-lives for episodic vs semantic memories. Sounds worse than it is. No memory is ever actually deleted; old ones just rank lower in search until you bring them back. See [Forgetting](#forgetting-nothing-gets-deleted-automatically) below.
- **Consolidates during "sleep"**: weekly LLM-driven dream cycle merges related memories into compressed representations
- **Detects contradictions automatically**: flags when new information conflicts with existing facts
- **Tracks temporal validity**: facts can have expiry dates and historical "valid_from" windows

## Architecture

```
                         ┌─────────────────┐
                         │  Claude Code    │
                         │   (or any MCP   │
                         │     client)     │
                         └────────┬────────┘
                                  │ JSON-RPC 2.0
                         ┌────────▼────────┐
                         │  Mnemos MCP     │
                         │     Server      │
                         │   (4 tools)     │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
        ┌─────▼─────┐      ┌──────▼──────┐    ┌──────▼──────┐
        │   FTS5    │      │   sqlite-   │    │    Jina     │
        │  BM25     │      │     vec     │    │   Reranker  │
        │ (lexical) │      │  (vectors)  │    │   (Jina v2) │
        └─────┬─────┘      └──────┬──────┘    └──────┬──────┘
              │                   │                   │
              └─────► RRF Merge ◄─┘                   │
                       │                              │
                       └──────► Cross-Encoder ◄───────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │   Results    │
                                └──────────────┘
```

### Single-database design

Everything lives in one SQLite file (`memory.db`):

- `memories`: main table with content, project, importance, type, layer, validity windows
- `memories_fts`: FTS5 virtual table for BM25 lexical search
- `embed_vec`: sqlite-vec table with 1024-dim vectors (FastEmbed e5-large)
- `memory_links`: cross-memory relationships (related, contradicts, supports)
- `dream_insights`: consolidation history and source tracking
- `consolidation_log`: audit trail of weekly dream cycles

Atomic transactions, no synchronization between separate stores.

## The 4 MCP tools

```python
memory_store(project, content, tags?, importance?, type?, subcategory?,
             valid_from?, valid_until?, verified?, layer?)
memory_search(query, project?, subcategory?, type?, layer?,
              valid_only?, search_mode?, limit?)
memory_get(id)
memory_update(id, [any field])
```

That's the entire surface. No `navigate_to_wing`, no `open_room`, no `list_halls`. Just CRUD plus search. Hierarchy is metadata, not architecture.

### Why 4 tools and not 19

> Teach me one way to do ten things, not ten ways to do one thing.

MemPalace exposes **19 MCP tools** for navigating its memory palace metaphor: tools to enter wings, open rooms, list halls, traverse closets, follow tunnels, and so on. Reading the tool list honestly feels like sitting down to play *King's Quest* in 1984: *> open door. > take key. > look closet*. Mnemos exposes **4**.

This is not about minimalism for its own sake. It is about how AI clients actually use tools:

1. **Every tool definition burns context tokens.** The full schema for 19 tools costs hundreds to thousands of tokens on every single request, on every single session, forever. Four tools is roughly a fifth of that overhead.
2. **More tools means more choice paralysis.** When the model has 19 ways to look something up, it has to reason about which tool fits, often picks suboptimally, and sometimes chains multiple navigation calls when one search would have answered the question. Four orthogonal tools (store, search, get, update) leave no room for ambiguity.
3. **Surface area is bug area.** Each tool is a contract you have to maintain, document, and not break. A pluggable storage backend is hard enough without 19 tool signatures pinned to a particular metaphor.
4. **The metaphor is not the system.** "Memory palace" is a mnemonic device for human memorization, not a database design pattern. Hierarchies are perfectly representable as `project` plus `subcategory` columns, filtered at query time. You do not need a tool called `enter_wing()` for that. It is metadata.

The four Mnemos tools cover the entire CRUD-plus-search surface that any memory system needs. Hierarchical filtering, type filtering, validity windows, namespaces, layers, and rerank modes are all parameters on the existing search tool, not new tools. Adding capability means adding optional parameters, never new tools.

If you ever feel constrained by four tools, the right reaction is "what parameter is missing from `memory_search`", not "I need a `memory_traverse_subcategory_tunnel` tool". So far the answer has always been a parameter.

## Features

### Hybrid retrieval pipeline
1. **FTS5 BM25** with AND-default + OR-fallback for multi-term queries
2. **Vector similarity** via FastEmbed e5-large (1024-dim, ONNX, CPU-native, ~7ms/embed)
3. **RRF fusion** (Reciprocal Rank Fusion, k=60) merges both rankings
4. **Cross-encoder reranking** (Jina v2 multilingual) for final precision (**opt-in** in the public package; the cross-encoder must be loaded into memory and adds roughly 50ms per search. See [Memory usage](#memory-usage) for what enabling it costs in RAM. Toggle with `MNEMOS_ENABLE_RERANK=1` or `Mnemos(enable_rerank=True)`)

### Why these specific models (and how to swap them)

The embedding model and the reranker are both **swappable**. Mnemos talks to FastEmbed-compatible models for embeddings and to any cross-encoder loadable through FastEmbed for reranking, so you can plug in whatever you prefer. The defaults are picked, not mandated.

I went with **`intfloat/multilingual-e5-large`** as the embedder for very specific reasons:

- **Truly multilingual**, not English-with-token-mapping. I store memories in both English and Swedish in production, and the semantic match needs to work *across* languages too ("frukost" should match "breakfast"). e5-large handles 100+ languages with the same vector space, which is rare for high-quality models
- **1024 dimensions** is a sweet spot. High enough to capture nuance, low enough that brute-force search in SQLite stays fast and storage stays reasonable
- **Available as ONNX** so it runs on CPU at ~7ms per embedding without a GPU runtime or PyTorch dependency
- **Apache 2.0 license**, no surprises
- **Battle-tested** on MTEB and many production retrieval systems

For the reranker I picked **`jinaai/jina-reranker-v2-base-multilingual`** for matching reasons: multilingual, ONNX-friendly, very small (sub-300MB), permissive license, and tuned for query-document relevance scoring rather than generic similarity.

If your use case is English-only or domain-specific, you can swap either one. Drop in a FastEmbed-compatible model name in `MNEMOS_EMBED_MODEL` and `MNEMOS_RERANKER_MODEL` env vars, or pass them directly when constructing the embed pipeline. Smaller models (BGE-small, all-MiniLM) will be faster but less precise. Larger models (BGE-large, GTE-large) will be slower but might score higher on specific benchmarks. Mnemos does not care, the rest of the pipeline is identical.

### Temporal modeling
- **Exponential decay** with separate half-lives:
  - Episodic memories: ~46 days (events, conversations)
  - Semantic memories: ~180 days (distilled knowledge)
- **Validity windows**: facts can have `valid_from` / `valid_until` for time-bound truths
- **`valid_only` filter**: exclude expired facts at query time

### Hierarchical organization
- Top-level projects are free-form strings. A starter set is provided (`dev`, `finance`, `food`, `health`, `home`, `personal`, `relationships`, `server`, `travel`, `work`, `writing`) but you can add or remove categories to match how you think about your memory. The storage layer does not enforce the list.
- Optional sub-categories per project (e.g., `dev/myapp`, `finance/crypto`)
- Free-form, no schema migration needed

### Dream cycle consolidation
Modeled on brain sleep stages, runs weekly:
1. **Triage**: detect new memories since last run
2. **Dedup**: merge near-duplicates (cosine ≥0.88 tight, ≥0.75 topic)
3. **Weave**: find cross-category relationships, create memory_links
4. **Contradict**: detect temporal evolution, mark superseded facts
5. **Synthesize**: generate cross-domain insights via LLM
6. **Bookkeeping**: decay old memories, cleanup orphans

### Adaptive learning: how Mnemos gets to know you

The dream cycle is not just maintenance, it is where Mnemos actually learns about the user behind the memories. Phases 2 and 4 quietly build a model of how you think, what you care about, and how your interests connect to each other.

> Personal note: I implemented the dream cycle in Mnemos v8 in February 2026, several weeks before Anthropic shipped any equivalent background-consolidation behavior into Claude. I genuinely laughed when their announcement landed and I realized I had quietly been running the same idea on my own server for weeks. Except mine works better. Hear that, Anthropic? I do not say any of this to claim invention. The "memory consolidates during sleep" concept is borrowed straight from neuroscience and a dozen prior research papers. I just want to be clear that this part of Mnemos was built independently and predates the closest commercial parallel I know about. Because a good idea is a good idea, and some things are so objectively right that everyone working on the problem ends up in the same place.

**Phase 2 (Weave)** scans every active memory for cross-category patterns. When it finds two memories that are semantically close but live in different projects (for example, a `health` memory about sleep and a `work` memory about meeting fatigue), it creates a `memory_links` row tagged `related`. Over weeks, these links form an implicit graph of "things this user mentally connects". Search results enrich themselves with these links, so a query about productivity surfaces relevant health context too, even if you never asked for it.

**Phase 4 (Synthesize)** goes a step further. It feeds clusters of related memories into the LLM with a prompt that asks for novel cross-domain observations: themes, recurring concerns, evolving preferences, contradictions you might not have noticed yourself. The synthesized insights are stored as new `semantic` memories, which then participate in future searches like any other fact. The system is, in effect, writing notes about you for later use.

A few practical consequences:

- **Preferences stick.** If you tell Mnemos once that you dislike a particular framework, that preference shows up in future search results when relevant, even years later.
- **Patterns emerge.** "User often discusses dental visits in the same week as financial planning" is the kind of thing the synthesizer notices and writes down.
- **Contradictions are visible.** Phase 3 flags "user used to say X, now says Y" as a temporal evolution, not a duplicate. Both versions stay queryable.
- **Tier-2 recall preserves nuance.** When a synthesis or merge compresses several originals, the originals are archived but linked. A future query can drill back down for the verbatim detail if needed.

This is opt-in and entirely local. The dream cycle only runs when you invoke `mnemos consolidate --execute`, and only the LLM-powered phases (1 through 4) need an API endpoint. Phase 5 bookkeeping always works without an LLM. If you never run consolidation at all, Mnemos behaves like a static memory store with no adaptive layer, and you lose nothing else.

### Contradiction detection
On every store, Mnemos checks if the new memory contradicts existing facts on the same topic (via vector similarity + cross-encoder). Conflicts are flagged in the response and stored as `memory_links` with `relation_type='contradicts'`.

### Auto-widen
If a project-filtered search returns fewer than 3 results, Mnemos automatically broadens the search to all projects. Reduces "I know I told you about X but you can't find it" failures.

### 3-way deduplication
Before storing, Mnemos checks for duplicates via:
1. FTS5 keyword overlap
2. CML-format subject matching
3. Vector cosine similarity

A cross-encoder ranks all candidates and returns a single confidence score.

## CML: token-minimal memory format

Every memory in Mnemos is text that an AI client will eventually read into its context window. Tokens are the actual currency of any LLM-backed system, and the memory store is the place where those tokens accumulate forever. A bloated memory format costs you context budget on every single retrieval, on every single session, for the entire lifetime of the project. Multiplied by thousands of memories, the difference between a verbose format and a compressed one is enormous.

So Mnemos uses **CML (Compressed Memory Language)** as a soft convention for writing memories. CML is not a parser, not a schema, not an encoder, not a compressor. It is just a tiny set of prefixes and symbols that the writer (you, or the AI assistant on your behalf) uses to pack common semantic patterns into the smallest number of tokens that still preserves meaning. The dedup pipeline understands the conventions well enough to flag conflicts on the same subject, but nothing in Mnemos compiles, validates, or transforms CML. It is just text that happens to be denser.

### Type prefixes (one or two characters)
| Prefix | Meaning |
|---|---|
| `F:` | Fact / config (technical configurations, system state, attributes) |
| `D:` | Decision (with reason) |
| `C:` | Contact (people, organizations, relationships) |
| `L:` | Learning (insight, lesson, pattern observed) |
| `P:` | Preference / pattern (what the user likes, prefers, repeatedly does) |
| `W:` | Warning (safety, gotcha, risk) |

### Relation symbols
| Symbol | Meaning |
|---|---|
| `→` | Leads to, results in, points to (causal) |
| `↔` | Mutual, bidirectional, relates to |
| `←` | Back-reference, originated from |
| `∵` | Because, due to (cause) |
| `∴` | Therefore, so, conclude (logical) |
| `△` | Changed, superseded, evolved |
| `⚠` | Warning, risk, gotcha |
| `@` | At (location, time, host, file) |
| `✓` | Confirmed, verified, working |
| `✗` | Failed, broken, negated |
| `~` | Approximate, uncertain, tentative |
| `∅` | None, empty, missing |
| `…` | Continuation, more, non-exhaustive |
| `;` `>` | Separators inside a line |
| `#42` | Reference to memory ID 42 |

### Quantitative shorthand
| Symbol | Meaning |
|---|---|
| `≥` `≤` | At least, at most |
| `≈` | Approximately |
| `≠` | Not equal, differs from |
| `↑` `↓` | Increased / decreased |
| `×` | Times, by, repeated |

### Less tokens, same detail

The key insight is that **compressed should never mean lossy**. CML drops the connective tissue that English requires for grammatical sentences ("currently", "the", "was", "approximately", "based on") but keeps every single piece of information that actually carries meaning. Both a human and an LLM can read the compressed version and reconstruct exactly the same understanding.

Compare a real engineering learning, the kind of thing you actually want your AI to remember:

```
Verbose prose (74 tokens):

"After three failed attempts to get the Stripe webhook integration working
 with my existing FastAPI middleware, I learned that Stripe requires the
 raw request body for signature verification, but my middleware was reading
 the body stream before it reached the webhook handler. The solution was to
 disable body parsing for the /webhooks/stripe route specifically using a
 custom dependency override."
```

```
CML (28 tokens):

"L: Stripe webhook sig verification needs raw body @FastAPI
 ∵ middleware consumed body stream → fix: disable body parsing
 for /webhooks/stripe via custom dependency"
```

**62% fewer tokens. Zero *actual* information lost.** Every entity, every cause, every fix, every constraint is still there. What got dropped was English filler ("currently", "the", "approximately", "based on") that carries no semantic weight. The `L:` prefix tells the AI this is a learning. The `∵` tells it the next clause is the cause. The `→` tells it the next clause is the resolution. An LLM reading this gets exactly the same actionable knowledge it would get from the prose version, and you paid less than half the context budget for it.

Now multiply that ratio across 1,000 active memories returned in briefings, search results, session priming, and dream cycle synthesis runs over months. The compounding token savings are not a rounding error; they are the difference between an AI that has plenty of context budget for actual reasoning and one that spends most of its window re-reading bookkeeping.

### Soft convention, hard rewards

CML is **not enforced, but it is the default and highly recommended.** You can store any free-form text you want and Mnemos will index and retrieve it just fine, but everything in the system (CLI hints, dedup, dream cycle, briefing format) assumes you are writing in CML and works better when you do. There are real incentives to follow the convention:

- CML is the format Mnemos expects you to write in. Examples, hints, briefings, and the dream cycle's merged super-memories all use it. The CLI still stores non-CML input rather than rejecting it, but prints a one-line nudge so you notice the convention exists and start using it
- **The dream cycle normalizes memories to CML automatically.** When Phase 1 merges duplicates or Phase 2/4 generate insights, the LLM is explicitly instructed to output only CML. So even if you store a memory in free-form prose, the next consolidation pass rewrites it (and any duplicates) into a single compact CML super-memory. Over time, the entire active store converges toward CML without you doing anything
- The dream cycle uses the same conventions when it merges memories, so consolidated super-memories stay compact
- Reranker results are tighter on CML inputs because the cross-encoder sees consistent structural cues

If you write your memories in CML, the system rewards you with smaller context bills, better dedup precision, and more reliable consolidation. If you do not, everything still works; you just leave compression on the table.

## Forgetting: nothing gets deleted automatically

A common misconception about decay-based memory systems is that old memories disappear. In Mnemos they don't. **Decay is a ranking modifier, not a deletion mechanism.**

When a memory ages, its temporal-decay boost shrinks, so it ranks lower in search results. The memory itself stays in the database. There is a floor at 10% of the maximum boost, so even a five-year-old fact still gets a small ranking nudge if it matches a query exactly. This mirrors how human memory actually works: you have not forgotten your old school address, you just no longer associate it with "where I live".

The decay curve uses two half-lives:

| Layer | Half-life | day 30 | day 90 | day 365 |
|---|---|---|---|---|
| Episodic (events, conversations) | 46 days | 0.96 | 0.39 | 0.15 (floor) |
| Semantic (distilled knowledge) | 180 days | 1.34 | 1.06 | 0.38 |

The only ways memories ever leave the active set:

- `mnemos delete <id>` archives a memory (still in DB, hidden from default search)
- `mnemos delete <id> --hard` permanently removes the row and its vector
- The optional dream cycle Phase 1 merges near-duplicates into a consolidated super-memory; the originals get archived but remain queryable via `dream_insights`
- `mnemos doctor` flags memories untouched for 90+ days as "stale" but takes no action

What protects a memory from sliding down in rank:

- **Access**: every search hit and `memory_get` resets the decay clock
- **`evergreen` tag**: skips decay entirely, for permanent facts (birthdays, blood type, addresses)
- **`semantic` layer**: 4x slower decay than `episodic` events
- **`consolidation_lock=1`**: protects from dream cycle merging
- **High importance (≥9)**: never demoted by Phase 5 bookkeeping

In short: Mnemos does not forget. It just stops bringing up things you have not used in a while, and lets you decide what to throw away.

## Beyond memories: one search across all your stuff

Mnemos started life as a curated memory store, but the same retrieval pipeline (FTS5 + vector + RRF + reranker) works for **any text content you want to search semantically**. The `mnemos ingest` command turns Mnemos into a unified semantic search layer for everything you might want your AI assistant to look through, not just CML facts you typed in by hand.

Out of the box you can index:

```bash
# A folder of notes (Obsidian, Logseq, plain markdown, anything)
mnemos ingest ~/notes --project notes --pattern "*.md" --recursive

# A code repository
mnemos ingest ~/projects/myapp --project code --pattern "*.py" --recursive --chunk 1500

# Documentation
mnemos ingest ~/docs --project docs --recursive

# A single file
mnemos ingest ~/important-doc.txt --project reference
```

After ingestion, the same `memory_search` MCP tool finds the new content alongside your curated memories. Your AI client can search across **memories, notes, code, and docs in one query**. Project filters keep them separate when you want, but a default search hits everything.

Built-in extractors handle plain-text formats (txt, md, py, js, json, yaml, html, sql, and a dozen more). For binary or structured formats like PDF, EPUB, or eml, you can register a custom extractor without modifying core:

```python
from mnemos.ingest import register_extractor
import pypdf

def extract_pdf(path):
    reader = pypdf.PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)

register_extractor(".pdf", extract_pdf)
```

For very large bulk content (mail archives, document collections of 100K+ items), bypass `mnemos ingest` and write directly against the storage layer for maximum throughput:

```python
from mnemos.storage import SQLiteStore
from mnemos.storage.base import Memory
from mnemos.embed import embed

store = SQLiteStore()
for doc in your_bulk_source:
    text = extract_text(doc)
    embedding = embed([text], prefix="passage")[0]
    store.store_memory(
        Memory(project="mail", content=text, subcategory=doc.folder),
        embedding=embedding,
    )
```

This is exactly how I run Mnemos in production: a single SQLite store for curated memories plus several Qdrant collections for bulk indexed mail, project documents, personal notes, ebooks, and work files (8 collections, ~500K vectors total). All searched through the same Mnemos hybrid pipeline, all from the same 4 MCP tools, all CPU-only.

**Mnemos is not just a memory system. It is a unified, local, CPU-only semantic retrieval layer for everything you own that has text in it.**

## Storage backends

Mnemos has a pluggable storage layer. Choose the backend that fits your scale:

| Backend | When to use | Status |
|---|---|---|
| **SQLiteStore** (default) | Personal use, up to ~10K memories on SSD, ~5K on HDD | ✅ Production |
| **QdrantStore** | Need HNSW vector search at 25K-millions of memories. Already running Qdrant for other indexing (mail, docs, notes) | ✅ Reference impl |
| **PostgresStore** | Multi-tenant production, ACID, MVCC concurrency | 🚧 Stub, contributions welcome |

In practice, **most users will never hit the SQLite ceiling**. The dream cycle continuously merges duplicates and similar memories into compact super-memories, so the active count grows much slower than the raw number of things you store. After months of heavy personal use my own active set sits in the low four digits, with thousands more archived but still queryable. For a normal user storing a handful of facts per day, reaching 5,000 active memories takes years, and "needing to switch to a real database" is a problem most people will never have. SQLite is also one less thing to install, run, secure, back up, and keep alive: the entire memory store is just a file you can `cp` somewhere safe.

I run Mnemos in **hybrid SQLite + Qdrant mode** in production, but only because I also use the same pipeline to index bulk external content (mail, project documents, personal notes, ebooks, work files; 8 collections, ~500K vectors total). SQLite holds the curated memory store. Qdrant holds the bulk content. The retrieval pipeline is identical regardless of backend; only the vector store changes.

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

## Memory usage

Mnemos is CPU-only but it does load real models into RAM. There are two distinct levels.

| Component | Disk | RAM (loaded) | When |
|---|---|---|---|
| Python + Mnemos package | ~50 MB | ~100-150 MB | Always |
| SQLite + sqlite-vec | trivial | ~10-50 MB (page cache) | Always |
| **e5-large embedder** (ONNX) | ~500 MB | **~800 MB-1 GB** | Always loaded once on startup |
| **Jina cross-encoder reranker** (ONNX) | ~280 MB | **~400-500 MB** | Only if `enable_rerank=True` |

**Without reranker** (default in the public package): **~1-1.2 GB resident** after warmup. Comfortably runs on a 4 GB Raspberry Pi 4, a small VPS, or any modern laptop.

**With reranker** (opt-in): **~1.5-1.7 GB resident**. Still fits on most consumer hardware, but you should not enable it on a 1 GB micro-VPS or a Pi Zero. Also note that the first time the reranker is touched in a session, it takes **1-2 seconds to spool up** (ONNX model load + first inference) before the first reranked query becomes ~50 ms. Mnemos warms it up at MCP server startup if you set `MNEMOS_ENABLE_RERANK=1`, so the spool cost is paid once at boot, not on the first user query.

**Disk requirements**: about 800 MB total for both ONNX models, downloaded automatically from HuggingFace on first use and cached under `~/.cache/fastembed`.

You can run Mnemos on a sub-1 GB machine by skipping the reranker entirely, or save another 500 MB by swapping `intfloat/multilingual-e5-large` for a smaller embedder like `BAAI/bge-small-en-v1.5` (English only, 33 MB ONNX, ~150 MB RAM). Set `MNEMOS_EMBED_MODEL` to override. Retrieval quality drops, but the system still works.

## Multi-user / Auth

Mnemos core is **single-tenant by design**. All operations are scoped to a `namespace` (default `"default"`); multi-user deployments simply use different namespaces. **There is no auth in the storage layer.** Authentication is the responsibility of the transport layer (the MCP server you run, the HTTP API you build on top, and so on).

This is the same separation Postgres uses (no auth in the query engine, all roles externally) and SQLite uses (file system permissions). It keeps the memory engine simple and lets you wrap it however your deployment needs.

## Installation

### Requirements
- Python 3.11+
- ~500MB disk for FastEmbed model + Jina reranker (auto-downloaded on first use)
- **No GPU required**: everything runs on CPU via ONNX

### Setup

```bash
git clone https://github.com/draca-glitch/mnemos.git
cd mnemos
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Register with Claude Code (or any MCP client)

Add to `~/.claude.json` (or your MCP client's config):

```json
{
  "mcpServers": {
    "mnemos": {
      "type": "stdio",
      "command": "/path/to/venv/bin/mnemos",
      "args": ["serve"],
      "env": {
        "MNEMOS_DB": "/home/you/.mnemos/memory.db",
        "MNEMOS_NAMESPACE": "default"
      }
    }
  }
}
```

Restart your AI client. The 4 tools (`memory_store`, `memory_search`, `memory_get`, `memory_update`) will be available.

### CLI usage

```bash
# Core CRUD
mnemos add --project dev "F:Mnemos uses sqlite-vec for vectors"
mnemos search "vector storage" --project dev
mnemos get 42
mnemos update 42 --importance 8
mnemos delete 42                    # archive
mnemos delete 42 --hard             # permanent

# Discovery / introspection
mnemos stats                         # active/archived counts per project
mnemos map                           # topic index by project + subcategory
mnemos digest --days 7               # recent memories
mnemos briefing                      # compact ~370-token session-start summary
mnemos embed-status                  # vector coverage report
mnemos doctor                        # health check (schema, FTS, embeddings, stale)

# Predictive priming (used by session hooks)
mnemos prime "current task description"

# Dream cycle consolidation (requires LLM, see below)
mnemos consolidate                   # dry run, default phases
mnemos consolidate --execute         # apply changes
mnemos consolidate --dream --execute # include synthesis (Phase 4)

# MCP server (typically invoked by your AI client, not directly)
mnemos serve
```

### Optional: dream cycle consolidation

Mnemos can run a 5-phase weekly **dream cycle** that merges related memories,
detects contradictions, and synthesizes cross-domain insights. Phases 1-4 require
an LLM. **Phase 5 (Bookkeeping) always runs without an LLM** and handles vector
cleanup, decay, and stale link pruning purely in SQL.

**Pick a smart model, not a fast one.** The dream cycle is intentionally a
background job; it runs weekly (or whenever you trigger it) and the entire
purpose is to take its time *thinking* about your memories. The quality of the
consolidation, dedup, and cross-domain synthesis is bounded by how good the LLM
is at reasoning, not by how fast it answers. A slow but capable model
(Qwen 2.5 32B locally, Claude Sonnet/Opus, GPT-4o, DeepSeek R1) will produce
much better results than a fast lightweight model. Latency does not matter
here. Quality does. If your dream cycle takes 20 minutes to run instead of 2,
that is fine, because it is running while you are asleep or doing something
else.

**You do not need a GPU even for the LLM.** Modern 32B-class models like
Qwen 2.5 32B, Llama 3.1 70B (quantized), or DeepSeek R1 distill variants can
run entirely on CPU through Ollama or llama.cpp, as long as you have enough
RAM (usually 32-64 GB depending on quantization). On a typical desktop or
small server without a graphics card, expect generation speeds of around
1-5 tokens per second instead of the 50-100 tok/s you would get on a GPU.
For Mnemos that is completely fine. The dream cycle does not care if Phase 1
takes 30 seconds or 5 minutes per merge cluster, as long as the merge is
correct. **Mnemos remains GPU-free end to end**, including consolidation,
if you choose a local model.

To enable LLM-powered phases, set environment variables for any OpenAI-compatible
endpoint:

```bash
# OpenAI
export MNEMOS_LLM_API_URL=https://api.openai.com/v1/chat/completions
export MNEMOS_LLM_API_KEY=sk-...
export MNEMOS_LLM_MODEL=gpt-4o-mini

# Local Ollama (free, runs on your machine)
export MNEMOS_LLM_API_URL=http://localhost:11434/v1/chat/completions
export MNEMOS_LLM_API_KEY=ollama
export MNEMOS_LLM_MODEL=qwen2.5:14b

# Or use OpenRouter, DigitalOcean Gradient, Together.ai, Groq, Fireworks, etc.
```

Without these set, `mnemos consolidate` will skip LLM phases and only run
bookkeeping. The core memory features (store/search/get/update) **never require
an LLM**. Mnemos itself only uses local CPU models for embeddings and reranking.

### Session hooks (Claude Code)

Inject relevant memories at session start automatically:

```bash
chmod +x scripts/mnemos-session-hook.sh
```

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{
      "type": "command",
      "command": "/path/to/mnemos/scripts/mnemos-session-hook.sh start"
    }]
  }
}
```

## Benchmark

Run the LongMemEval benchmark yourself:

```bash
cd benchmarks
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -O longmemeval_s.json

# Default hybrid: FTS5 + vector + RRF (no cross-encoder in the search path)
python longmemeval_bench.py --mode hybrid

# Hybrid + cross-encoder reranker on top of the hybrid candidates
python longmemeval_bench.py --mode hybrid+rerank
```

Mnemos was benchmarked in two modes against the LongMemEval 470-question hybrid set. The two modes use the exact same retrieval pipeline up to RRF; the only difference is whether the Jina cross-encoder reorders the top candidates before returning them.

### `--mode hybrid` (default in the public package, no cross-encoder)

| Question Type | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| knowledge-update | 98.6% | 100.0% | 100.0% | 100.0% | 72 |
| multi-session | 94.2% | 99.2% | 100.0% | 100.0% | 121 |
| single-session-assistant | 92.9% | 96.4% | 96.4% | 96.4% | 56 |
| single-session-preference | 60.0% | 86.7% | 86.7% | 96.7% | 30 |
| single-session-user | 98.4% | 100.0% | 100.0% | 100.0% | 64 |
| temporal-reasoning | 89.8% | 94.5% | 97.6% | 98.4% | 127 |
| **Overall** | **91.9%** | **97.0%** | **98.1%** | **98.9%** | **470** |

### `--mode hybrid+rerank` (opt-in, cross-encoder over top RRF candidates)

> Currently running. Numbers will be filled in once the full 470-question run completes; early signal is that it lifts R@1 and R@5 most on the weakest category (single-session-preference, which sits at 86.7% in pure hybrid). I will update this table with the actual measured numbers instead of speculating.

| Question Type | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| knowledge-update | TBD | TBD | TBD | TBD | 72 |
| multi-session | TBD | TBD | TBD | TBD | 121 |
| single-session-assistant | TBD | TBD | TBD | TBD | 56 |
| single-session-preference | TBD | TBD | TBD | TBD | 30 |
| single-session-user | TBD | TBD | TBD | TBD | 64 |
| temporal-reasoning | TBD | TBD | TBD | TBD | 127 |
| **Overall** | **TBD** | **TBD** | **TBD** | **TBD** | **470** |

### Tradeoff: when is `hybrid+rerank` worth it?

Hybrid alone is already at 98.1% R@5, which is competitive with state-of-the-art systems and uses no extra model. Reranker mode is for the last mile, when you specifically care about R@1 and the precision of the very top result.

| | hybrid | hybrid+rerank |
|---|---|---|
| RAM (loaded models) | ~1-1.2 GB | ~1.5-1.7 GB |
| First-query spool time | ~0 ms (embedder warm) | **+1-2 s** to load reranker once |
| Per-query latency | ~5-30 ms | **~50-80 ms** (+50 ms cross-encoder) |
| Recall@5 | 98.1% | (pending benchmark) |
| Best for | "show me anything relevant" queries | "give me the single best answer" queries |

If you are running on a tight RAM budget (Raspberry Pi, sub-1 GB VPS, embedded device), keep hybrid mode. If you have spare RAM and care about top-1 precision, enable rerank with `MNEMOS_ENABLE_RERANK=1`. The 50 ms per query is barely perceptible, the 1-2 s spool cost is paid once at MCP server startup, and the quality improvement on the weakest categories is real.

**The choice is explicit.** Mnemos lets you pick the hit rate you want against the resources you can spare:

- **~98% R@5** with hybrid alone, ~1 GB RAM, ~5-30 ms per query, runs on a Pi
- **(targeting) 99%+ R@5** with hybrid + reranker, ~1.5-1.7 GB RAM, ~50-80 ms per query, ~1-2 s spool at startup

Both numbers are achieved on the same retrieval pipeline, with the same code, by toggling a single flag. You are not locked in either direction.

## How Mnemos compares to other systems

| System | R@5 | Error rate | Approach | GPU? | Local? |
|---|---|---|---|---|---|
| **Mnemos (this repo)** | **98.1%** | **1.9%** | FTS5 + vector + RRF (+ optional rerank) | ❌ No | ✅ Yes |
| MemPalace raw ChromaDB | 96.6% | 3.4% | Vector-only over verbatim text | ❌ No | ✅ Yes |
| Mastra observational memory | ~95% | ~5% | RL-based | ✅ Yes | ❌ No |
| Dense retrieval (paper baseline) | ~90% | ~10% | Vector-only | varies | varies |
| BM25 baseline (paper) | ~85% | ~15% | Lexical-only | ❌ No | ✅ Yes |

Mnemos is the only system in this list that combines **state-of-the-art retrieval** with **CPU-only inference**, **no cloud calls**, a **pluggable storage backend**, and a benchmark number that is **a first run with no parameter tuning** against the dataset. The 1.5pp lead over MemPalace raw is roughly **44% fewer errors per query** (1.9% vs 3.4% miss rate), which on a benchmark already in the high nineties is a meaningful gap.

## License

MIT: see [LICENSE](LICENSE).

## Credits

Built on top of:
- [FastEmbed](https://github.com/qdrant/fastembed): multilingual e5-large ONNX embeddings
- [sqlite-vec](https://github.com/asg017/sqlite-vec): vector search in SQLite
- [Jina Reranker v2](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual): cross-encoder
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval): benchmark dataset
