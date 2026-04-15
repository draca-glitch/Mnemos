# How Mnemos compares

> Architecture and benchmark comparisons against MemPalace and the broader memory-system field. See [origin.md](origin.md) for why this comparison exists at all.

## MemPalace, side by side

I am comparing against MemPalace specifically because they are, by their own account and by the articles written about them, the current number one in the AI personal memory space. If you are going to benchmark yourself, you benchmark against whoever is on top.

> *Note: the more I read about MemPalace and dig into their actual architecture and reproducible benchmarks, the less the "number one" framing seems to hold up. The headline articles and the substance underneath are not telling the same story. I am leaving the comparison in because it is the reason this repo exists, but treat the framing below as historical context for why Mnemos got published, not as a current claim that MemPalace is the bar to clear.*

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
| Consolidation                | Mining modes | **6-phase Nyx cycle (LLM-driven)** |
| Auto-widen on thin results   | None | **Cross-project fallback** |

The article I read about MemPalace, the one that changed my mind on whether to release my system or not, talked about it like it was the next best thing since AI sliced bread. When I actually looked at the architecture, my reaction was different. To me MemPalace sounds like overly complex mimicry of how the human brain organizes memories spatially, when the AI on the other end neither wants nor needs a spatial metaphor to retrieve information. It just needs a good search, and the complexity may carry more structural overhead than the retrieval task requires. Their headline "+34% palace boost" is, by their own transparency note, what you get from any standard ChromaDB metadata filter.

Mnemos skips the metaphors and ships the retrieval pipeline that actually scores. **Fewer tools, deeper plumbing, more honest defaults.** An AI neither needs nor wants to function like a human brain. **It does not need a palace, it needs efficiency.**

## How Mnemos compares to other systems

| System | R@1 | R@3 | R@5 | R@10 | Approach | Local? |
|---|---|---|---|---|---|---|
| **Mnemos canonical (BM25+vector+rerank+CML)** | **95.74%** | **98.72%** | **99.15%** | **99.15%** | FTS5 + sqlite-vec + RRF + cross-encoder, cemelification | ✅ Yes |
| **Mnemos clean (BM25+vector+rerank)** | **94.3%** | **98.1%** | **98.94%** | **99.15%** | Same pipeline, no CML preprocessing of test data | ✅ Yes |
| **Mnemos lite (BM25+vector only)** | 92.13% | 97.02% | 98.30% | 98.94% | FTS5 + sqlite-vec + RRF, no reranker | ✅ Yes |
| MemPalace raw ChromaDB (no LLM) | n/a | n/a | 96.6% | n/a | Vector-only over verbatim text | ✅ Yes |
| MemPalace + Haiku rerank (LLM in loop) | n/a | n/a | 100% | n/a | Vector + Claude Haiku LLM rerank, **per-query Anthropic API call**, ~500 outbound calls per benchmark run | ⚠️ Local + remote LLM API |
| LongMemEval paper Stella+fact baseline | n/a | n/a | ~73% | n/a | Vector + value/fact expansion | ✅ Yes |
| Dense retrieval (paper baseline) | n/a | n/a | ~90% | n/a | Vector-only | varies |
| BM25 baseline (paper) | n/a | n/a | ~85% | n/a | Lexical-only | ✅ Yes |
| Mastra, Supermemory, Mem0, OMEGA, Emergence, Zep, Letta, Cognee, EverMemOS, TiMem, etc. | not published | not published | not published | not published | LLM-judge QA accuracy only, no retrieval recall | varies |

A few things worth noting about this table:

1. **Mnemos appears to be the only memory system in the field publishing a complete R@1 / R@3 / R@5 / R@10 sweep on LongMemEval.** MemPalace publishes only R@5 plus per-type R@10, and almost everyone else publishes only LLM-judge QA accuracy (a different metric that bundles retrieval and generation).

2. The competitor architectural notes are based on each system's own published documentation and benchmark methodology, not on source code audits. I have not read every competitor's source. The Mnemos source is in this repository and you can verify everything in the Mnemos rows directly.

3. **The 98.94% number (clean BM25+vector+rerank) requires no asterisk of any kind.** No CML preprocessing of test data, no LLM in the search path, no tuning. It is the most defensible single number in the table.

4. **The 99.15% number (canonical BM25+vector+rerank+CML) is the configuration the system was designed to run in**, and the one I use in production. For the benchmark to work in this mode it does involve a one-time LLM call per session to convert raw LongMemEval transcripts into CML at storage time (real users do not pay this cost; their agent writes CML directly). See [How the CML benchmark actually works](benchmarks.md#how-the-cml-benchmark-actually-works-the-part-with-the-asterisk) for the full disclosure.

5. **Mnemos performs zero LLM calls of any kind in the production search path.** The cross-encoder reranker is a discriminative scorer (Jina v2, ~570 MB ONNX, CPU-only), not a generative language model. Among systems in the upper-90s **retrieval recall** tier, Mnemos appears to be the only one I could verify that never calls a generative model in its retrieval pipeline.

6. **On end-to-end QA accuracy (the metric Mastra/Emergence/Mem0/etc. publish), Mnemos's paired-with-Sonnet pipeline scores 77.4%**, competitive with the middle of the published field, below Emergence's reported 95%, above LongMemEval paper baselines. The QA accuracy gap vs retrieval reflects answerer-side reading comprehension on multi-session aggregation, not a retrieval failure. See [End-to-end QA accuracy](benchmarks.md#end-to-end-qa-accuracy-retrieval--answerer--judge) for full numbers and methodology.

7. **Abstention handling is largely undisclosed in the field.** LongMemEval's 500 questions include 30 abstention cases (correct behavior = refusal when the retrieved context doesn't contain the answer). Retrieval metrics (R@K) skip these since there's no gold session to retrieve. QA-accuracy numbers should include them but most systems don't disclose whether their reported accuracy is on all 500 or just the 470 non-abstention questions. Mnemos publishes both, the [QA section](benchmarks.md#end-to-end-qa-accuracy-retrieval--answerer--judge) reports 77.4% on all 500 with Sonnet+raw, and **96.7% accuracy specifically on the 30 abstention questions** (the answerer correctly refuses when retrieved context lacks the answer, rather than fabricating). A system confidently answering questions whose correct answer is "I don't know" is a quiet quality problem the field has not been forced to benchmark.

8. **On why QA numbers do not get re-run every time something changes**: the end-to-end QA benchmark costs real money. A single full 500-question QA run against Sonnet + Opus-judge is ~$20-30 in API fees, and running a sweep of answerer/judge pairs multiplies that. I cannot afford to re-run these numbers on every code change. The retrieval benchmarks are free in API terms (they run locally on CPU with zero outbound calls) but they are not free in wall-clock time, a full run is **10 to 15 hours** on the reference hardware, so even the retrieval numbers are not re-run on every commit; they get revalidated when something material changes in the pipeline, and the QA numbers are revalidated less often still. If you reproduce either benchmark against your own account, your results will be directionally the same but may differ by a few percentage points due to model non-determinism (QA side) or to hardware / embedder-build differences (retrieval side); the [`benchmarks/`](../benchmarks/) directory has the runners and the result JSONs so you can verify whichever dimension you care about.

The combined picture: **Mnemos is the only memory system in this comparison I could verify with (a) upper-90s retrieval recall on LongMemEval, (b) end-to-end QA numbers published alongside retrieval numbers rather than in place of them, (c) fully local CPU-only operation with no generative LLM in the search path, and (d) a stable four-tool MCP surface under 1.5 K tokens of system-prompt overhead.** Other systems may match it on one or two of those axes; based on what they publish, I have not found one that matches on all four.
