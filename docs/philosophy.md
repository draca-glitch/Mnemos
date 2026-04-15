# Philosophy

> A memory system should be a memory. Not an agent, not a reasoning palace.

## What is Mnemos?

**A memory system should be a memory.** It stores data and retrieves it. That is the entire job.

The AI memory field seems to have gradually drifted toward treating memory as a reasoning surface. Based on what the major memory systems publish about their own architectures, most of the memory products people are shipping today appear to do significant reasoning work inside the memory layer itself: calling LLMs during retrieval, augmenting queries with inference steps, exposing nineteen tools where four would do, and burning tokens to redo work the agent on the other end of the conversation has already done. I have not audited every competitor's source code, nor do I intend to, and I do not claim to know exactly how each one behaves at runtime. But from their public docs, their benchmarks, and their tool surfaces, the field looks like it has slowly forgotten what a memory is, which is ironic.

Your brain does not run inference every time you try to remember where you put your keys. It runs lookup. The reasoning happens around the memory, not inside it. You think *"where did I put my keys?"*, you query memory, memory hands you back *"on the kitchen counter"*, and then you reason about what to do next. Memory itself is dumb, fast, and reliable.

Mnemos is built that way on purpose. The agent thinks. Mnemos remembers. The agent decides what is worth storing and how to phrase it; Mnemos receives those bytes and persists them. The agent decides what to ask about; Mnemos finds the right items and hands them back. There is no LLM call inside the retrieval or storage pipeline. The search path does use two small models (`e5-large` for embeddings and Jina v2 for cross-encoder reranking) but neither is a generative language model. They are discriminative scorers that take inputs and emit numbers, locally, on CPU, in milliseconds, without generating any tokens or making any API calls. When a query comes in, the answer just *happens*. No model is "working" to think it through. There is no reasoning step inside Mnemos that duplicates what the agent on the other side already did. There are four MCP tools (`store`, `search`, `get`, `update`) because those are the verbs a memory has. Everything else is the agent's job, not the memory's.

Not a reasoning palace. Not an agent. Not a thinking creature. **A memory.**

## Why 4 tools and not 45

I have always lived by the teaching: *teach me one way to do ten things, not ten ways to do one thing.* It applies to the kitchen knife you keep sharp instead of buying ten gadgets, the language you speak fluently instead of dabbling in five, the tool in your toolbox you actually know how to use, the keyboard shortcuts you can hit without looking, the APIs you design, and the memory system you trust your AI assistant to. PUBG showed us a frying pan can be used as a weapon, not just for frying things. Same energy.

Some memory systems expose upwards of **19 MCP tools** for navigating their internal metaphors: tools to enter wings, open rooms, list halls, traverse closets, follow tunnels, and so on. Reading a tool list that long honestly felt like sitting down to play *King's Quest* in 1984: *> look at door > open door > look in room > pick up key > use key with lock > open closet > look in closet*. Mnemos exposes **4**.

This is not about minimalism for its own sake. It is about how AI clients actually use tools:

1. **Every tool definition burns context tokens.** The full schema for 19 tools costs hundreds to thousands of tokens on every single request, on every single session, forever. Four tools is roughly a fifth of that overhead.
2. **More tools means more choice paralysis.** When the model has 19 ways to look something up, it has to reason about which tool fits, often picks suboptimally, and sometimes chains multiple navigation calls when one search would have answered the question. Four orthogonal tools (store, search, get, update) leave no room for ambiguity.
3. **Surface area is bug area.** Each tool is a contract you have to maintain, document, and not break. A pluggable storage backend is hard enough without 19 tool signatures pinned to a particular metaphor.
4. **The metaphor is not the system.** "Memory palace" is a mnemonic device for human memorization, not a database design pattern. Hierarchies are perfectly representable as `project` plus `subcategory` columns, filtered at query time. You do not need a tool called `enter_wing()` for that. It is metadata.

The four Mnemos tools cover the entire CRUD-plus-search surface that any memory system needs. Hierarchical filtering, type filtering, validity windows, namespaces, layers, and rerank modes are all parameters on the existing search tool, not new tools. Adding capability means adding optional parameters, never new tools.

If you ever feel constrained by four tools, the right reaction is "what parameter is missing from `memory_search`", not "I need a `memory_traverse_subcategory_tunnel` tool". So far the answer has always been a parameter.

## Adaptive learning: how Mnemos gets to know you

> **A note on the name.** **Nyx** (Νύξ, *"Night"*) is the Greek primordial goddess of Night, older even than Zeus. In the Theogony she is the mother of Hypnos (sleep) and the Oneiroi (dreams), among others. Where Mnemosyne is memory itself, Nyx is the quiet time when memory gets worked on. Naming the consolidation cycle after her is not ornamental: it literally is the process that happens in the dark while nothing else is using the system, consolidating, weaving, and letting the day's input settle into something that will be retrievable tomorrow.

The Nyx cycle is not just maintenance, it is where Mnemos actually learns about the user behind the memories. Phases 3, 4, and 5 quietly build a model of how you think, what you change your mind about, and how your interests connect to each other:

- **Weave** (Phase 3) finds semantically-close memories across different projects and stores the link. Over weeks this forms an implicit graph of "things this user mentally connects" that enriches future search results.
- **Contradict** (Phase 4) catches slow-burn temporal evolution, beliefs or preferences that shifted over time. Both versions stay queryable; the older gets a `valid_until` marker and a link to its successor. Nothing is silently overwritten.
- **Synthesize** (Phase 5) feeds clusters of related memories into the LLM and asks for novel cross-domain observations: themes, recurring concerns, patterns you might not have noticed yourself. The insights are stored as new `semantic` memories and participate in future searches like any other fact.

Run Mnemos for a few months and the surprising thing happens: it starts knowing things about you that you do not know about yourself. Not in an oracle way, just the mundane way any long-running pattern detector eventually outpaces a human looking at the same data. It sees the link between your sleep notes and your bad-mood entries before you do. It surfaces the recurring concern you have been brushing off for six months. It notices the friend you only ever mention when you are tired. The synthesizer is writing notes about you for later use, and some of those notes are uncomfortably accurate.

Opt-in and entirely local. The Nyx cycle only runs when you invoke `mnemos consolidate --execute`, and only the LLM-powered phases need an API endpoint. Phase 1 (Triage) and Phase 6 (Bookkeeping) are pure SQL and always run. If you never invoke consolidation at all, Mnemos behaves like a static memory store with no adaptive layer and loses nothing else.

> Personal note: I implemented the Nyx cycle in Mnemos v8 in February 2026, several weeks before Anthropic shipped any equivalent background-consolidation behavior into Claude. I genuinely laughed when their announcement landed and I realized I had quietly been running the same idea on my own server for weeks. Except mine works better. Hear that, Anthropic? I do not say any of this to claim invention. The "memory consolidates during sleep" concept is borrowed straight from neuroscience and a dozen prior research papers. I just want to be clear that this part of Mnemos was built independently and predates the closest commercial parallel I know about. Because a good idea is a good idea, and some things are so objectively right that everyone working on the problem ends up in the same place.
