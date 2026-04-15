# CML: token-minimal memory format

> Condensed Memory Language. A soft convention for writing memories denser, not a parser or a compressor.

Every memory in Mnemos is text that an AI client will eventually read into its context window. Tokens are the actual currency of any LLM-backed system, and the memory store is the place where those tokens accumulate forever. A bloated memory format costs you context budget on every single retrieval, on every single session, for the entire lifetime of the project. Multiplied by thousands of memories, the difference between a verbose format and a condensed one is enormous.

So Mnemos uses **CML (Condensed Memory Language)** as a soft convention for writing memories. CML is not a parser, not a schema, not an encoder, not a compressor. It is just a tiny set of prefixes and symbols that the writer (you, or preferably the AI assistant on your behalf) uses to pack common semantic patterns into the smallest number of tokens that still preserves meaning. The dedup pipeline understands the conventions well enough to flag conflicts on the same subject, but nothing in Mnemos compiles, validates, or transforms CML. It is just text that happens to be denser.

> **A disclaimer.** I am not claiming CML is the best possible notation for this problem. What I *am* claiming is that **compression is not the answer; efficient language is.**
>
> The mechanism matters. AAAK and similar compression approaches work **on text**: they apply regex-based abbreviation rules and entity codes (`KAI` for `Kai`, etc.) to strings that already exist. The retriever then has to score against text that no longer matches the original surface form. BM25 loses its tokens, the bi-encoder embeds something it never saw at training time, and the cross-encoder has no idea what `KAI` means. That mechanism is why AAAK regresses retrieval on LongMemEval (84.2% R@5 vs raw 96.6%, by MemPalace's own published numbers).
>
> CML works **on the writer**, not on the text. The agent or LLM is instructed to write in CML notation in the first place. No transformation step, no encoder, no decoder, no entity table, no regex pass. Every token in a CML memory is recognizable English (`L: webhook sig needs raw body @FastAPI ∵ middleware ate stream → fix: disable body parse` reads as plain English to a human and as a structurally-tagged learning to an LLM, with no decoding step in either case), so FTS5 indexes the original tokens, the bi-encoder embeds text it can handle, and the cross-encoder scores against text that means what it says. **Compression removes signal the retriever needs. Efficient language preserves it while saying less.**
>
> There are almost certainly better notations to discover, or ways to extend CML further that would sharpen it beyond what it does today. This is where my own exploration has reached so far. If you find something better, open an issue; the grammar is small enough to evolve without breaking anything downstream.

> **CML is not a theoretical claim, it was tested.** The question "does the retrieval pipeline actually work as well on CML-stored memories as on plain-prose memories?" is answered by the `--cml` LongMemEval modes in [`benchmarks/longmemeval_bench.py`](../benchmarks/longmemeval_bench.py). Each LongMemEval session is cemelified via a one-time Claude Haiku call (cached on disk by SHA256 hash), stored in the normal Mnemos pipeline, and then the full benchmark runs against that CML-formatted corpus. Both `--mode hybrid --cml` and `--mode hybrid+rerank --cml` land in the same R@5 tier as the non-CML modes, despite the bi-encoder being mildly out of distribution on structural CML, because the cross-encoder handles CML's structural prefixes and operators as explicit relation markers. Result JSONs are committed: [`results_hybrid_session_cml.json`](../benchmarks/results_hybrid_session_cml.json) and [`results_hybrid+rerank_session_cml.json`](../benchmarks/results_hybrid+rerank_session_cml.json). CML is not a compressor the retriever has to decode, it is just denser English text, and the pipeline handles it cleanly. The orthogonal question, "does the MERGE prompt produce valid CML consistently across LLM tiers?", is answered by the [Consolidation quality bench](benchmarks.md#consolidation-quality-fact-preservation-in-the-nyx-cycle).

> **Brief aside: I had to invent a new English word to write this README.** The verb is **`cemelify`**, pronounced as `See-M-El` + `ify`, which is the three letter names of `CML` spoken aloud and condensed into a single syllable. It means "rewrite verbose text in CML notation". To be clear, this is not a translation in the language-to-language sense; it is more like a compaction, or a "remove redundant words" function. The output is still recognizable English, just denser, with the connective tissue stripped out and replaced with a small set of prefixes and operators that an LLM (or a human) can read directly. I wanted a single verb for this action because writing out "rewrite in CML form" everywhere felt clumsy, and existing verbs like *condense*, *canonicalize*, or *densify* are all close but none of them include the structural prefix-tags-and-operators part that CML adds on top of just shortening. So I made one up.

## Type prefixes (one or two characters)

| Prefix | Meaning |
|---|---|
| `F:` | Fact / config (technical configurations, system state, attributes) |
| `D:` | Decision (with reason) |
| `C:` | Contact (people, organizations, relationships) |
| `L:` | Learning (insight, lesson, pattern observed) |
| `P:` | Preference / pattern (what the user likes, prefers, repeatedly does) |
| `W:` | Warning (safety, gotcha, risk) |

## Relation symbols

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

## Quantitative shorthand

| Symbol | Meaning |
|---|---|
| `≥` `≤` | At least, at most |
| `≈` | Approximately |
| `≠` | Not equal, differs from |
| `↑` `↓` | Increased / decreased |
| `×` | Times, by, repeated |

## Less tokens, same detail (for any reader)

CML is **lossless for any reader of the text**. It drops the connective junk tissue that English requires for grammatical sentences to sound correct but that carries no informational value ("currently", "the", "was", "approximately", "based on") but keeps every single piece of information that actually carries meaning. Both a human and an LLM can read the condensed version and reconstruct exactly the same understanding the prose carried.

This is not just a framing claim. Rewriting 20 hand-curated prose memories (209 atomic facts across two styles: 15 fact-dense production-style notes and 5 longer narrative-style entries) into CML and then checking which facts survived, **Claude Opus 4.6 preserves 100% overall**, Sonnet 4.6 preserves 98.1%, Haiku 4.5 preserves 98.1%, and gpt-4o-mini preserves 95.2%. The compression is input-dependent: fact-dense production prose condenses 15–26% (Claude tier and gpt-4o-mini all in that band), and longer narrative prose condenses **41–61%** (gpt-4o at 0.39×, gpt-4o-mini at 0.48×, Sonnet at 0.52×, Haiku at 0.55×, Opus at 0.59×). Full per-subset table across 8 models and per-memory breakdowns in the [CML fidelity benchmark](../benchmarks/README.md#4-cml-fidelity-format-level-content-parity--cml_fidelity_benchpy). In practice the agent writes directly in CML, so there is no transformation step at all, the bench is just the cleanest way to prove that the format is expressive enough to hold everything a prose memory would have held.

Compare a real engineering learning, the kind of thing you actually want your AI to remember:

```
Verbose prose (73 tokens, cl100k_base):

"After three failed attempts to get the Stripe webhook integration working
 with my existing FastAPI middleware, I learned that Stripe requires the
 raw request body for signature verification, but my middleware was reading
 the body stream before it reached the webhook handler. The solution was to
 disable body parsing for the /webhooks/stripe route specifically using a
 custom dependency override."
```

```
CML (39 tokens, cl100k_base):

"L: Stripe webhook sig verification needs raw body @FastAPI (3 tries)
 ∵ middleware consumed body stream → fix: disable body parsing
 for /webhooks/stripe via custom dependency override"
```

**About 47% fewer tokens. Zero *actual* information lost.** Every entity, every cause, every fix, every constraint is still there, including the gotcha context (`(3 tries)`) and the FastAPI mechanism name (`override`). What got dropped was English filler ("currently", "the", "approximately", "based on") that carries no semantic weight. The `L:` prefix tells the AI this is a learning. The `∵` tells it the next clause is the cause. The `→` tells it the next clause is the resolution. An LLM reading this gets exactly the same actionable knowledge it would get from the prose version, and you paid roughly half the context budget for it.

How much CML actually shrinks the input depends on how much filler the input had. A technical sentence like the Stripe one above still has narrative glue ("After three failed attempts...", "I learned that...", "The solution was to...") and condenses by about 47%. Longer narrative prose with more connective tissue ("I was sitting at my desk thinking about...", "as I was saying...", "in any case...") condenses harder, up to about **61%** smaller in the bench (gpt-4o on narrative inputs lands at 0.39× at 90.6% preservation; the four top Claude/OpenAI-budget tiers cluster in the 0.48–0.59× range at 92–100% preservation). Fact-dense prose that already reads like a production memory entry (short, already minimal on filler) has less room to shrink: the CML fidelity bench in `benchmarks/` measures 14% to 26% savings against that kind of input, depending on the model doing the cemelification. Across realistic mixed memory inputs, expect savings anywhere in the **14% to ~60%** range depending on how narrative-vs-dense the original text was. The denser your prose, the smaller the win; the more it sounds like English narrative, the bigger the win.

Now multiply that ratio across 1,000 active memories returned in briefings, search results, session priming, and Nyx cycle synthesis runs over months. The compounding token savings are not a rounding error; they are the difference between an AI that has plenty of context budget for actual reasoning and one that spends most of its window re-reading bookkeeping.

## CML and the reranker: complementary, not co-required

CML is lossless for any *reader*, but it is mildly out of distribution for the *bi-encoder* that powers the first stage of semantic retrieval. e5-large was trained on natural-language pairs (Wikipedia, MS MARCO, mC4); it never saw structural prefixes like `F:`, `D:`, `L:` or operators like `→` `∵` `@` during pretraining. The condensed form embeds slightly less cleanly than equivalent prose, and on LongMemEval that translates to a small recall difference if you run pure BM25+vector mode (no reranker) on CML-stored memories.

**The cross-encoder reranker is the cherry on top, not a hard requirement.** CML's structural prefixes act as explicit relation markers the cross-encoder can attend to, which is exactly what reranker architectures are designed to exploit. With the reranker enabled, CML's small bi-encoder penalty disappears and the canonical CML+rerank configuration beats the non-CML rerank configuration on every metric except R@10 where they tie. CML and the reranker are complementary, using both gets you the best number, but you can ship either one alone, or neither, and still land in the same tier as every other public memory system's strongest verified number on this benchmark.

The combined picture:

- **CML alone, no reranker**: essentially the same recall as the no-CML lite mode, with **15% to 60% fewer tokens** stored and reinjected per memory (the spread depends on how narrative the source prose is). Net positive whenever context budget matters at all, which is most of the time.
- **CML + reranker (canonical)**: recall is *higher* than the non-CML reranker baseline at every published k except R@10 (tie), AND you still get the 15% to 60% token savings. The honest cost of running the reranker, compared to skipping it, is about **+50 ms per query** and roughly **+500 MB resident RAM** for the cross-encoder model. The RAM cost is negligible on any modern laptop or server and only matters on Pi-class hardware. The latency cost is barely perceptible to a human and is paid once per search, not per memory. For the recall improvement and the token savings combined, it is the right trade for almost every deployment.

For any active memory system the bottleneck eventually becomes context budget, not raw retrieval recall. Combining CML with the reranker gives you the best of both: upper-90s retrieval recall and a fraction of the tokens. Full per-mode benchmark numbers in [benchmarks.md](benchmarks.md).

## Soft convention, hard rewards

CML is the default convention everything in Mnemos assumes, and in practice the system enforces it through machinery rather than through schema validation. The CLI hints in CML. The dedup pipeline matches CML subject prefixes for conflict detection. The Nyx cycle cemelifies any prose that slipped through into CML during weekly consolidation. Briefings, digests, and the topic map all render in CML. The MCP tool description teaches the agent CML conventions so it writes in CML directly.

So if you store a memory in plain prose, Mnemos will accept it and search it just fine, but over time the system pulls it toward CML form unless you actively prevent that. **You can opt out**: don't run the Nyx cycle's LLM phases, customize the agent's system prompt to write in plain prose, and you have a prose-only deployment. But by default the corpus is or is becoming CML, and the rest of the system is built around that assumption.

> **Single-flag opt-out**: the `MNEMOS_CML_MODE` environment variable flips every CML-related surface that Mnemos controls, in one place. Set `MNEMOS_CML_MODE=off` (default is `on`) and: the MCP `memory_store` tool description drops its CML guidance and tells the agent to write clear natural prose instead, the `consolidation_lock` field descriptions on `memory_store` and `memory_update` swap from "prevent cemelification" to "prevent merging", the Nyx cycle merge and synthesis phases switch to prose-output prompts that still preserve every atomic fact, the dedup CML-subject branch falls back to FTS + vector signals only, and the Phase 3 Weave bridge insights drop the `L:` prefix. One coordinated flag, prose-only deployment is a first-class configuration, not a degraded mode.
>
> One surface Mnemos cannot toggle for you: *your AI client's own system prompt* (Claude Code's `CLAUDE.md`, Cursor rules, custom Claude Desktop system prompts, etc.). Those are user-owned and outside the MCP tool interface. See [`agent-instructions.md`](agent-instructions.md) for copy-paste CLAUDE.md blocks for both CML and prose modes; use the matching block for whichever mode you run, and swap if you flip the flag. Mnemos itself only instructs the AI through the tool descriptions it exposes, which `MNEMOS_CML_MODE` covers.

There are real incentives to follow the convention rather than fight it:

- CML is the format Mnemos expects you to write in. Examples, hints, briefings, and the Nyx cycle's merged super-memories all use it. The CLI still stores non-CML input rather than rejecting it, but prints a one-line nudge so you notice the convention exists and start using it
- **The Nyx cycle cemelifies memories automatically.** When Phase 2 merges duplicates or Phase 3/5 generate relationships and insights, the LLM is explicitly instructed to output only CML. So even if you store a memory in free-form prose, the next consolidation pass rewrites it (and any duplicates) into a single compact CML super-memory. Over time, the entire active store converges toward CML without you doing anything
- The Nyx cycle uses the same conventions when it merges memories, so consolidated super-memories stay compact
- Reranker results are tighter on CML inputs because the cross-encoder sees consistent structural cues

If you (or, more realistically, the AI agent on your behalf) write memories in CML, the system rewards you with smaller context bills, better dedup precision, and more reliable consolidation. If you do not, everything still works; you just leave the savings on the table until the Nyx cycle catches up.
