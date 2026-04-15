# Origin

> Why Mnemos exists, why the name, and why v10 in a brand-new repository.

## Scratching a real itch, then "wait, hold my beer"

Mnemos started because the default `memory.md` approach felt deeply inefficient and dumb. A flat markdown file that Claude (or any AI) reads on every session is fine for a handful of facts, but it does not scale, it does not rank, it does not forget anything that should be forgotten, and it does not actually let the model *remember the user* in any meaningful way. I wanted my AI assistant to know who I am, what I care about, what I have decided, what I prefer, the way a long-term colleague would. Not just to re-read a static text file at every session start.

So I started building my idea of a memory system. Over months it grew into a real retrieval system: FTS5 for lexical search, sqlite-vec for semantic search, RRF fusion, a cross-encoder for high-precision dedup, exponential temporal decay, a weekly Nyx cycle for consolidation. It has been running in private production powering personal AI agents for months before this repo existed. **I was never planning to publish it.** It was just my own infrastructure, built for my own use, by someone who got tired of `memory.md`. In fact I imported my entire ChatGPT history into it just recently, just to see what would happen, and watching the Nyx cycle quietly stitch together patterns across years of unrelated conversations was the moment I realized this thing was doing something I had not seen anywhere else.

Then MemPalace surfaced, claiming state-of-the-art retrieval with a hierarchical memory metaphor. I read their README and benchmarks and had the strangest reaction: *"wait, hold my beer."* I had never benchmarked Mnemos against anything before, because until a few days ago I did not even know LongMemEval existed; I had just been building the system for my own use and trusting my own gut about whether it was getting better. So I pointed Mnemos at the same LongMemEval dataset MemPalace had used. First run, no tuning, no preprocessing of the benchmark data: a recall figure that landed comfortably in the high-90s on verbatim LongMemEval transcripts.

That result is what tipped this from "private side project I had no plans to share" into "I should clean this up and put it on GitHub for public scrutiny". If you are reading this, the MemPalace comparison is the reason this repo exists at all. See [comparison.md](comparison.md) for the head-to-head and the full benchmark table.

## A note on scope

Mnemos is a personal memory system. It is not trying to compete with production-grade multi-tenant memory services like [Mastra](https://mastra.ai), [Mem0](https://mem0.ai), [Supermemory](https://supermemory.ai), [Hindsight](https://hindsight.ai), and the other commercial systems in this space. Those are built for teams, hosted APIs, enterprise SLAs, and commercial support contracts, and Mnemos will not hold a candle to them on that axis.

**For personal Claude (or any MCP-compatible AI) memory on a single machine, though, it is more than enough, and that is the audience it is built for.** If you are running memory for a support-agent fleet with thousands of customers, reach for one of the commercial options instead. The benchmark comparisons are about the retrieval-recall axis specifically, the one axis where a personal tool and a multi-tenant service can meaningfully be measured against each other, not a claim that Mnemos replaces what they do.

## Why v10 in a brand-new repository?

Mnemos has been in private production for months. Each of the nine internal versions involved real experimentation: adding features, removing the ones that did not pull their weight, evaluating retrieval quality, and iterating on what actually made the system smarter rather than just bigger. v10 is the state that was running on the day I decided to publish, the existing architecture, a packaging and documentation pass, and the benchmark numbers cited here, all generated from that same code.

As for the repository history: I had never used git for coding before this project. The entire v1 to v9 history of Mnemos lived as plain files on a home server, iterated on manually, never under version control. I opened my first GitHub account and pushed this repository in April 2026. The system has months of internal history behind it. The repository does not. See [CHANGELOG.md](../CHANGELOG.md) for the incomplete timeline; I was not used to professional coding workflows and the changelog only captures what I happened to write down at the time.

## Why "Mnemos"?

The first reason is straightforward: **Mnemosyne** (μνημοσύνη) is the Greek goddess of memory and mother of the nine Muses. Her name literally means "remembrance". A memory system named after her writes itself.

The second reason is a wink. **Mythos** is the rumored name of Anthropic's next Claude model. Mythos tells stories, Mnemos remembers them. Same Greek mythology bench, same family of words, complementary roles: if Mythos becomes the model that powers your AI assistant, Mnemos is the memory it draws from. The naming was already in place before any of that surfaced publicly, but I was not going to pretend the pairing was not too good to keep.

To be clear: this project has no affiliation with Anthropic. It is appreciation, not partnership. Just two names from the same root, doing the two things memory and storytelling have always done together. And I am really looking forward to Mythos. If it lands the way I am hoping, the pairing of a storytelling-oriented Claude on top of a curated Mnemos store is exactly the kind of setup I would want to use myself. So if anyone at Anthropic is reading this, I'd love to evaluate it. I promise only to use it for good.
