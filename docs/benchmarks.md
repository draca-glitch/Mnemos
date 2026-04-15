# Benchmarks

> Multiple metric classes, clearly separated. Per-mode methodology, LoCoMo, end-to-end QA, consolidation quality, CML fidelity. See [comparison.md](comparison.md) for cross-system comparison tables.

> **Multiple metric classes, clearly separated.** Memory systems get benchmarked on fundamentally different things, and conflating them produces misleading comparisons. Mnemos publishes across all of them.
>
> - **Retrieval recall (R@K, NDCG@K) on LongMemEval.** Does the system find the right memory? Deterministic, no LLM in the measurement path. 470 non-abstention questions. Mnemos's headline numbers live here.
> - **Retrieval recall (R@K) on LoCoMo.** Same metric class, different dataset. 1,540 evaluable QA pairs across 10 long conversations (19-32 sessions each), with 446 adversarial questions excluded by methodology. Top-K capped at 10 to avoid the bypass where K ≥ session count makes retrieval trivial. Numbers in [`benchmarks/README.md`](../benchmarks/README.md#3-locomo-retrieval-recall-locomo_benchpy).
> - **End-to-end QA accuracy (LLM-judge).** Given the retrieved context, does the answerer produce the correct answer? Bundled metric, measures retrieval + answerer reading comprehension + judge fairness. LongMemEval supports this on all 500 questions including abstention. This is what Mastra, Emergence, Mem0, Supermemory, and most of the field publish. Mnemos also publishes these numbers below.
> - **Consolidation quality (fact preservation).** How well does the offline Nyx cycle merge related memories without losing specifics? Our own internal metric, reported against 30 historical merge events from the Mnemos production database.
> - **CML fidelity.** Does CML preserve the same atomic-fact content as equivalent prose? 20-memory hand-curated corpus, 209 facts, 8 models tested. See [`benchmarks/README.md`](../benchmarks/README.md#5-cml-fidelity-format-level-content-parity-cml_fidelity_benchpy).
>
> **Do not compare R@K numbers against QA accuracy numbers.** They measure different things. 98.94% R@5 is not directly comparable to 86% QA accuracy, one measures retrieval, the other measures a full pipeline including an answering LLM. Any table or chart that puts these numbers in the same column is metric-mixing.

> **What each benchmark measures, and what it doesn't.** Mnemos's core job is narrow: store memories, retrieve the right ones when asked. That is exactly what R@K on **LongMemEval** measures, and the constructed-haystack format is a fair fit for what Mnemos does. **LoCoMo** measures the same retrieval task on a different data shape (one long continuous conversation, much longer sessions), which is why CML preprocessing matters more there. **Abstention accuracy** is not a retrieval metric at all, it measures whether the answerer LLM correctly refuses when context lacks the answer; R@K skips abstention questions by definition. The one thing none of these benchmarks exercises is the *Nyx cycle*: the offline consolidation that merges duplicates, links across categories, and keeps a long-running store curated. That is why the consolidation-quality bench (below) is self-graded against the production database. The retrieval numbers above measure retrieval honestly; they just don't measure the optional consolidation layer.

## Run it yourself

```bash
cd benchmarks
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -O longmemeval_s.json

# Lightweight: BM25 + vector only, no reranker, no CML preprocessing
python longmemeval_bench.py --mode hybrid

# Cleanest unmanipulated benchmark: BM25 + vector + cross-encoder reranker, no preprocessing
python longmemeval_bench.py --mode hybrid+rerank

# Hybrid with cemelification at storage time, no reranker
python longmemeval_bench.py --mode hybrid --cml

# Canonical Mnemos: BM25 + vector + reranker + CML, the configuration the system was designed to run in
python longmemeval_bench.py --mode hybrid+rerank --cml
```

Mnemos was benchmarked across **four configurations** against the LongMemEval 470-question session-granularity set. Every run uses the exact same code, the same models, the same RRF weights, the same reranker thresholds, and **zero parameter tuning against the benchmark**. No held-out validation set was peeked at. No thresholds were swept. No models were swapped to chase a number. **All four are first-run results. The benchmark just ran.**

> **A note on Mnemos's natural habitat.** In production, Mnemos's curated memory store is **always CML**. The agent writes memories in CML directly via `memory_store()`, the Nyx cycle keeps it that way in the background, and by the time anything queries the store, the bytes are already CML. **The two non-CML benchmark modes (`--mode hybrid` and `--mode hybrid+rerank`) are therefore Mnemos being benchmarked outside its natural habitat**, running on raw verbatim conversation prose that the production memory store would never actually hold. Those modes correspond instead to how `mnemos ingest` indexes external content (mail, documents, ebooks, notes, prose that was never CML to begin with). They are published for three reasons: (1) the cleanest possible measurement, with no preprocessing of the test data and no LLM call anywhere in the pipeline, which is the most skeptic-proof number we can offer; (2) the fact that **Mnemos remains competitive with verified retrieval-recall numbers in the field even when handicapped by running on prose it was not designed to store as memories**; (3) and the strong non-CML numbers (98.30% / 98.94% R@5) confirm that **Mnemos handles verbose natural-language prose just fine**, which is exactly what the `mnemos ingest` pipeline relies on for indexing external content like mail, documents, ebooks, and notes, content that is and stays in prose form. The CML modes (`--mode hybrid --cml` and `--mode hybrid+rerank --cml`) are the production configuration for the curated memory store, where Mnemos is in its native form.

**One methodology disclosure for the CML modes**: they cemelify the LongMemEval transcripts at storage time before running the benchmark. The benchmark data itself is converted from raw verbatim conversation prose into CML notation via a one-time Claude Haiku 4.5 call per session, cached on disk, then handed to the standard storage pipeline. This is how Mnemos works in production (the Nyx cycle does the same thing in the background to anything that slipped through as prose). It does mean the two `--cml` runs are technically operating on a *transformed* version of the test data, but that transformation matches the production state of every memory in the store. See [How the CML benchmark actually works](#how-the-cml-benchmark-actually-works-the-part-with-the-asterisk) below for the full methodology.

The four result JSON files (with `completed_at` timestamps from the actual runs) live in [`benchmarks/`](../benchmarks/) so anyone can verify the numbers below match the source data.

### `--mode hybrid` (BM25 + vector only, no reranker, no CML)

The lightest configuration Mnemos ships. FTS5 BM25 + sqlite-vec + RRF fusion, no cross-encoder, no CML preprocessing of stored data. ~1 GB RAM, ~5 to 30 ms per query, runs on a Raspberry Pi.

> **Important: this is not how Mnemos runs the curated memory store.** The memory store is always CML + rerank (canonical, see below). This mode is relevant for `mnemos ingest`-indexed external content (mail, documents, ebooks, notes), prose that was never CML to begin with, and that you want indexed on a tight RAM budget.

| Question Type | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| knowledge-update | 98.61% | 100.0% | 100.0% | 100.0% | 72 |
| multi-session | 94.21% | 98.35% | 100.0% | 100.0% | 121 |
| single-session-assistant | 94.64% | 96.43% | 96.43% | 96.43% | 56 |
| single-session-preference | 60.00% | 86.67% | 86.67% | 96.67% | 30 |
| single-session-user | 98.44% | 100.0% | 100.0% | 100.0% | 64 |
| temporal-reasoning | 89.76% | 95.28% | 98.43% | 98.43% | 127 |
| **Overall** | **92.13%** | **97.02%** | **98.30%** | **98.94%** | **470** |

NDCG@5 = 0.9612, NDCG@10 = 0.9590. Zero pipeline failures. Source: [`benchmarks/results_hybrid_session.json`](../benchmarks/results_hybrid_session.json)

### `--mode hybrid+rerank` (cleanest unmanipulated measurement)

Hybrid retrieval plus the Jina v2 cross-encoder reranker. **No preprocessing of LongMemEval test data, no LLM calls anywhere in the pipeline, no parameter tuning.** The benchmark just runs and the numbers come out. The cross-encoder is a discriminative scorer, not a generative language model: it takes a query and a candidate document and emits a relevance score, locally, on CPU, in ~50 ms, without generating any tokens. **There is no LLM in the search path.** The 98.94% number is achieved without any model "working" to think the answer through. The search just happens. Completed 2026-04-10 16:04.

> **Important: this is not how Mnemos runs the curated memory store either.** The memory store is always CML + rerank (canonical, see below). Like `--mode hybrid` above, this configuration is relevant for `mnemos ingest`-indexed bulk content where the source is prose and the cross-encoder is worth the extra RAM. It is published here primarily as the skeptic-proof number: the cleanest possible measurement of the pipeline on raw LongMemEval data with nothing processed, tuned, or preprocessed.

| Question Type | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| knowledge-update | 98.6% | 100.0% | 100.0% | 100.0% | 72 |
| multi-session | 98.3% | 100.0% | 100.0% | 100.0% | 121 |
| single-session-assistant | 92.9% | 96.4% | 96.4% | 98.2% | 56 |
| single-session-preference | 73.3% | 86.7% | 100.0% | 100.0% | 30 |
| single-session-user | 98.4% | 100.0% | 100.0% | 100.0% | 64 |
| temporal-reasoning | 91.3% | 97.6% | 97.6% | 97.6% | 127 |
| **Overall** | **94.3%** | **98.1%** | **98.94%** | **99.15%** | **470** |

Source: [`benchmarks/results_hybrid+rerank_session.json`](../benchmarks/results_hybrid+rerank_session.json)

**98.94% R@5 with no asterisk of any kind.** It is also at or above every other verified clean retrieval-recall number on LongMemEval I was able to confirm from public sources. The closest comparable is MemPalace raw ChromaDB at 96.6% (which, per the MemPalace maintainers, measures ChromaDB's default sentence-transformer embeddings with no MemPalace-specific code path involved); most other systems in the field either publish only LLM-judge QA accuracy (a different metric class that bundles retrieval and generation) or rely on a generative LLM in their highest-scoring configurations.

### `--mode hybrid --cml` (BM25 + vector with cemelification, no reranker)

Same BM25 + vector pipeline as `--mode hybrid` above, but with stored memories converted to CML notation at storage time using the same Nyx cycle conversion path that runs in production.

| Question Type | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| knowledge-update | 98.61% | 100.0% | 100.0% | 100.0% | 72 |
| multi-session | 96.69% | 99.17% | 100.0% | 100.0% | 121 |
| single-session-assistant | 96.43% | 100.0% | 100.0% | 100.0% | 56 |
| single-session-preference | 53.33% | 76.67% | 80.00% | 83.33% | 30 |
| single-session-user | 100.0% | 100.0% | 100.0% | 100.0% | 64 |
| temporal-reasoning | 90.55% | 96.85% | 97.64% | 97.64% | 127 |
| **Overall** | **92.98%** | **97.45%** | **98.09%** | **98.30%** | **470** |

NDCG@5 = 0.9615, NDCG@10 = 0.9573. Zero pipeline failures. Source: [`benchmarks/results_hybrid_session_cml.json`](../benchmarks/results_hybrid_session_cml.json)

R@5 sits essentially tied with the no-CML BM25+vector mode (98.09% vs 98.30%), but **R@1 on `single-session-preference` collapses to 53.33%**, the lowest single-cell number in any of the four configurations Mnemos publishes. The reason: the bi-encoder (e5-large) was trained on natural-language pairs and is mildly out of distribution on CML's condensed structural form. Without a reranker to recover the gap, the bi-encoder ranks the right session at slot 1 only about half the time on the precision-heavy preference questions. **The cross-encoder reranker more than rescues this in the canonical configuration below**, preference R@1 jumps from 53.33% to 80.00% the moment the reranker is added back, because CML's structural prefixes that confused the bi-encoder become explicit relation markers the cross-encoder can attend to. This is the worst way to run Mnemos: cemelifying memories at storage time but then running retrieval without the cross-encoder, which is the exact partner that turns CML's structural prefixes from a bi-encoder confusion into a cross-encoder advantage. This row is published anyway as an honest ablation, because it is the only configuration of the four I would not personally use, and pretending otherwise would be the kind of selective reporting this README is built to avoid.

### `--mode hybrid+rerank --cml` (canonical Mnemos, the configuration the system was designed to run in)

This is what real Mnemos looks like in actual production use: stored memories already in CML form (the agent writes them that way at storage time, no Nyx cycle required), retrieval through the standard BM25 + vector + cross-encoder rerank pipeline, no LLM in the search path. **The configuration is what I actually run in production.** For the benchmark, the LongMemEval transcripts are cemelified once at storage time to match this production state; the [methodology section below](#how-the-cml-benchmark-actually-works-the-part-with-the-asterisk) explains exactly how and why.

| Question Type | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| knowledge-update | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 72 |
| multi-session | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 121 |
| single-session-user | 96.88% | **100.0%** | **100.0%** | **100.0%** | 64 |
| single-session-assistant | 92.86% | 98.21% | **100.0%** | **100.0%** | 56 |
| single-session-preference | 80.00% | 93.33% | 96.67% | 96.67% | 30 |
| temporal-reasoning | 93.70% | 97.64% | 97.64% | 97.64% | 127 |
| **Overall** | **95.74%** | **98.72%** | **99.15%** | **99.15%** | **470** |

NDCG@5 = 0.9781, NDCG@10 = 0.9774. Zero pipeline failures. Source: [`benchmarks/results_hybrid+rerank_session_cml.json`](../benchmarks/results_hybrid+rerank_session_cml.json)

> **The number to take away from this section is 99.15%.** That is what Mnemos does in actual production use, in the canonical configuration the system was designed to run in. The 98.94% above is what it does in a deliberately raw measurement mode that Mnemos was not intended to use as its memory store; it is published as the most skeptic-proof number on offer, for readers who want zero asterisks. If you actually run Mnemos the way I run it, **99.15% R@5** is the number you will see.

> **Four full categories at 100% R@5. 313 questions in a row where Mnemos made zero retrieval mistakes.** Across `knowledge-update` (72/72), `multi-session` (121/121), `single-session-user` (64/64), and `single-session-assistant` (56/56), Mnemos's canonical configuration retrieves the right session every single time. The four total misses across all 470 non-abstention questions (one in single-session-preference, three in temporal-reasoning) give an **0.85% miss rate**. To my knowledge, no other memory system in any published comparison reports a result like this on LongMemEval, and the data is not being post-processed at evaluation time or scored through an LLM API. The score comes out of pure retrieval recall on the canonical configuration, every query handled locally on CPU, no generative model in the loop.

#### How the CML benchmark actually works (the part with the asterisk)

Storage requires an LLM call no matter what memory system you use, because something has to decide what is worth remembering and how to phrase it. That reasoning step happens in the agent (Claude Code, Cursor, or whichever AI is talking to the user) and is the same fixed cost for any AI memory system in the field. Mnemos does not double that cost. When the agent calls `memory_store()` with already-formed CML content, Mnemos receives the bytes, runs FTS+vector dedup, embeds via FastEmbed, and stores. **No second LLM round-trip inside Mnemos to process, format, or extract anything from what the agent already wrote.**

In the benchmark, LongMemEval provides raw verbatim conversation transcripts rather than pre-formatted CML notes. **Mnemos's own benchmark harness** (in `benchmarks/cml_convert.py`) calls Claude Haiku 4.5 once per session message via DO Gradient API to convert those transcripts into CML, caches the result on disk by SHA256 hash, then hands the CML output to the standard Mnemos storage pipeline. The LongMemEval benchmark itself does not call any LLM; Mnemos is the one doing it, as a one-time data-shape conversion that happens outside the retrieval pipeline being measured. This conversion is necessary to be able to measure how Mnemos actually works in CML mode, the benchmark dataset ships in raw conversation prose, so without converting it to CML at storage time there is no way to evaluate the canonical production configuration of the system. **This LLM call is a benchmark artifact**, present because LongMemEval data needs conversion in order to benchmark it the way Mnemos actually stores memories in real use. Real users never pay this cost: in production, the agent writing the memory does that formatting at no additional cost as part of its existing reasoning step, and any prose that slips through gets cemelified gradually by the Nyx cycle in the background.

**This is also exactly why the other three benchmark configurations exist**: the non-CML runs measure Mnemos against the raw LongMemEval data with no preprocessing at all, the lite mode measures it without the reranker, and the CML-without-reranker mode measures the worst-case combination. Together the four runs let you see exactly how each design choice contributes to the final number, decide for yourself which configuration best matches what you care about, or check whether I am full of shit.

In my own production setup the Nyx cycle uses **Qwen3-32B served via DigitalOcean Gradient** (their OpenAI-compatible inference endpoint) to cemelify anything that slipped through as prose. The public package has **no default model** and you set whichever one fits via `MNEMOS_LLM_MODEL` (and any OpenAI-compatible endpoint via `MNEMOS_LLM_API_URL`). If you do not configure an LLM at all, the Nyx cycle simply skips the LLM-driven phases and the rest of Mnemos still works. Most modern LLMs handle CML conversion equivalently because the task is constrained and mechanical: the grammar does the heavy lifting and the model is just a translator.

If you disagree with whether LLM-based CML preprocessing of the benchmark data counts as fair, **use the `--mode hybrid+rerank` table above instead**. That number (98.94% R@5) is the cleanest possible measurement and is at or above every verified retrieval-recall number I could compare it against. The CML number is a more accurate picture of how Mnemos actually performs in steady-state production use; the verbose number is a more conservative measurement of pure retrieval quality. Both are honest, both come from first-run no-tuning runs, both are in the same tier as every verified competitor retrieval-recall number at every comparable k, and both result files are in `benchmarks/` with timestamps you can verify against the file modification times.

### Configuration summary

| Configuration | R@5 | RAM | Latency | When to use |
|---|---|---|---|---|
| `hybrid` (BM25 + vector, lite) | 98.30% | ~1-1.2 GB | ~5-30 ms | Pi-class / 2 GB+ Pi 4 / embedded; or for `mnemos ingest`-indexed external prose content |
| `hybrid+rerank` (clean benchmark) | **98.94%** | ~1.5-1.7 GB | ~50-80 ms | The skeptic-proof published number; not what you would deploy by default, but what you cite when someone wants zero asterisks |
| `hybrid --cml` (CML, no rerank) | 98.09% | ~1-1.2 GB | ~5-30 ms | Honest ablation; bi-encoder is mildly OOD on CML, rerank rescues it below. R@5 looks fine but R@1 on `single-session-preference` collapses to **53.33%**, the worst cell in any configuration. Don't use this unless you hate the idea of a local CPU reranker but love CML. |
| **`hybrid+rerank --cml` (canonical, recommended default)** | **99.15%** | ~1.5-1.7 GB | ~50-80 ms | **The default for any deployment with RAM to spare.** What Mnemos was designed to run in. Steady-state of an actively-used Mnemos. |

All four are first-run, no-tuning measurements with their result JSON files committed to `benchmarks/`. **Mnemos lets you pick the hit rate you want against the resources you can spare**, on the same retrieval pipeline, by toggling two configuration flags. You are not locked in any direction. But if you ask me, you should run it in CML with the reranker on; that is the configuration the system was designed around and the one I use every day.

## LoCoMo retrieval recall (second public benchmark)

LongMemEval is the primary benchmark Mnemos was designed against, but publishing a single benchmark is brittle: any system can over-fit to one dataset. [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024) is a separate long-conversation memory benchmark: 10 conversations of 19-32 sessions each, 1,986 QA pairs across 5 categories. Median session length 2,652 chars (much longer than LongMemEval's typical sessions, which changes what configurations are practical).

The same Mnemos retrieval pipeline runs against LoCoMo with no parameter tuning. Methodology guardrails: top-K capped at 10 (the smallest LoCoMo conversation has 19 sessions; K ≥ 19 trivially returns every session and stops measuring retrieval), 446 adversarial-by-design questions in category 5 excluded from R@K (mathematically undefined; same convention LongMemEval applies to abstention).

| Mode | R@1 | R@3 | R@5 | R@10 | NDCG@5 |
|---|---|---|---|---|---|
| `hybrid` (BM25 + vector + RRF) | 57.9% | 77.0% | **84.7%** | **94.0%** | 77.1% |
| `hybrid --cml` | 49.0% | 69.9% | 79.4% | 91.0% | 70.1% |
| `hybrid+rerank --cml` | 60.9% | 79.7% | **86.1%** | 91.9% | 79.7% |

`hybrid+rerank` without CML is not in the table because it underperforms even plain `hybrid` on this benchmark: LoCoMo sessions are too long for the cross-encoder's attention window when scored at full length, and aggressive truncation loses the evidence the cross-encoder was supposed to read. CML preprocessing (sessions compressed to ~500 chars of dense facts) is the prerequisite for effective reranking on long-session data. Full methodology, per-category breakdown, and the conv-26 truncation-experiment data point are in [`benchmarks/README.md`](../benchmarks/README.md#3-locomo-retrieval-recall-locomo_benchpy).

The headline takeaway: **the same Mnemos pipeline that scores 99.15% on LongMemEval scores 86.1% on LoCoMo** without any retraining or per-dataset tuning. The gap is the dataset, not the system: LoCoMo conversations are denser and longer than LongMemEval's, with a larger unanswerable subset (446 adversarial questions in LoCoMo vs 30 abstention questions in LongMemEval, both excluded from R@K by the same convention since R@K is mathematically undefined when there is no gold session to retrieve). Both benchmarks evaluate retrieval recall the same way; both numbers are first-run, no-LLM-in-retrieval, fully air-gappable.

## End-to-end QA accuracy (retrieval + answerer + judge)

LongMemEval's primary metric in the original paper is end-to-end QA accuracy: retrieval returns top-K sessions, an answerer LLM produces an answer from those sessions, and a judge LLM scores the answer against the reference. The 500 questions include 30 abstention cases where the correct behavior is refusal. Published numbers in this metric class include [Mastra at 94.87%](https://mastra.ai/research/observational-memory) (binary QA accuracy on LongMemEval-S with gpt-5-mini as answerer, per their research page), Emergence at ~95%, and several other systems. Note: [Mem0](https://mem0.ai/research) has not published a LongMemEval number at the time of writing. Their 66.9% figure is overall accuracy on LoCoMo, a different benchmark entirely.

Mnemos's R@5 = 98.94% means retrieval is near-perfect. End-to-end QA accuracy is necessarily ≤ R@5, you can only answer correctly what retrieval found. The gap between R@5 and QA accuracy tells you how much the answerer LLM fumbles correct retrievals.

Two configurations tested, 500 questions each, all results JSON files in [`benchmarks/`](../benchmarks/):

**Mnemos hybrid retrieval + Claude Sonnet answerer + Claude Opus judge (raw session context, K=5):**

| Question Type | Accuracy | N |
|---|---|---|
| abstention (correct refusal) | 96.7% | 30 |
| single-session-assistant | 94.6% | 56 |
| single-session-user | 93.8% | 64 |
| knowledge-update | 81.9% | 72 |
| temporal-reasoning | 72.4% | 127 |
| multi-session | 66.1% | 121 |
| single-session-preference | 46.7% | 30 |
| **Overall** | **77.4%** | **500** |

**Mnemos hybrid retrieval + gpt-4o answerer + gpt-4o judge + per-session CML compression of retrieved context:**

| | Value |
|---|---|
| **Overall QA accuracy** | **58.6%** (293/500) |

The CML-compressed-context configuration performs worse on multi-session and preference question types because per-session CML compression *independently* destroys cross-session aggregation signals. When the same fact is mentioned across three sessions, raw text preserves three independent mentions (answerable by counting), while independent per-session CML compresses each session into a terse blob with no cross-session cardinality reference. This is a known architectural limitation of the query-time compression approach tested here, the production path (Nyx cycle consolidation before retrieval, documented below) addresses this by merging related memories into a single consolidated fact before retrieval runs, preserving aggregate signals. That configuration was not re-run against LongMemEval for this release because LongMemEval's haystack is constructed from topically-disjoint conversations and does not exercise the accumulation patterns consolidation is designed for. See the Consolidation Quality section below for the orthogonal measurement.

**Methodology disclosure for the QA numbers:**

- **Answerer:** Claude Sonnet 4.6 (raw-context run) / gpt-4o (CML-context run), temperature 0, K=5 retrieved sessions.
- **Judge:** Claude Opus 4.6 (raw run) / gpt-4o (CML run), temperature 0, per-question-type guidance prompts.
- **Same-family judge bias:** Sonnet-answerer-to-Opus-judge and gpt-4o-to-gpt-4o are both same-family evaluator pairs. Cross-family judge (e.g. gpt-4o judging Sonnet output) rerun is planned as a follow-up ablation; the numbers above may shift by up to ±3 points under a cross-family judge.
- **What the Sonnet+raw 77.4% number means:** Mnemos's retrieval paired with Claude Sonnet as answerer, using raw retrieved session text as answer context. This is what a Claude Code user pairing Mnemos with Sonnet would actually see. Not directly comparable to Mastra's 94.87% because their run uses gpt-5-mini as answerer on LongMemEval-S while ours uses Claude Sonnet on the same dataset. Answerer choice is a large factor in QA accuracy (gpt-5-mini scored 100% on our consolidation-quality bench with its reasoning tokens; different model families exhibit very different reading-comprehension behavior on this kind of retrieved-context QA). An apples-to-apples re-run against a standardized answerer is planned.
- **No tuning against the benchmark.** Same retrieval code, same hybrid mode, same K, no prompts tuned to LongMemEval's phrasing, no threshold sweeps. First runs.
- **Abstention is included.** The 30 abstention questions test whether the answerer correctly refuses when retrieved context lacks the answer. R@K evaluation necessarily skips them (no gold session to retrieve); QA evaluation must include them. Our 96.7% abstention accuracy demonstrates the answerer correctly refuses under retrieved-distractor pressure.

**Honest take on the QA numbers:**

Mnemos's 77.4% QA accuracy on the Sonnet+raw configuration sits in the upper-middle of the public field, above LongMemEval's paper baselines (60-65%), below Mastra's reported 94.87% (with gpt-5-mini as answerer), and below Emergence's reported ~95%. Under a cross-family judge rerun the number may move ±3 points. The retrieval layer (98.94% R@5) is the stronger story; the end-to-end number is bottlenecked by the answerer's reading comprehension on multi-session aggregation (66% on that type) and by judge strictness on fuzzy preference matches (47% on preferences). Neither is a retrieval problem. Mnemos's memory layer already found the right sessions in both cases. These are the limitations of treating LongMemEval's adversarial haystack of topically-disjoint conversations as a proxy for real accumulation-based memory use. See the Consolidation Quality section for where the accumulation-based improvements actually manifest.

## Consolidation quality (fact preservation in the Nyx cycle)

LongMemEval is constructed from one-shot conversation sessions and does not exercise the accumulate-merge-revisit cycle that consolidation is designed for. To measure consolidation quality on real data, Mnemos runs a separate benchmark against its own production memory database: 30 historical merge events (clusters of 2-8 memories that the Nyx cycle previously consolidated), exact ground truth via the `merged-into-<id>` tag lineage preserved on archived originals, unique-fact preservation measured by embedding-based semantic deduplication of extracted source facts followed by LLM-judged preservation check.

| Configuration | Unique-fact preservation | Cost/run | Notes |
|---|---|---|---|
| **Older MERGE_SYSTEM (Opus)** | **75.3%** | ~$6 | Historical merges already in the database from before the prompt rewrite. Fact loss concentrated on 6-8 memory clusters. |
| **Current MERGE_SYSTEM + Opus** | **89.0%** | ~$6 | Rewritten prompt ("co-location not compression" with self-audit rule) + hierarchical pairwise merge for clusters >2. |
| Current + Claude Sonnet | 90.4–94.5% | ~$1 | Two runs, ±4-point variance. 5× cheaper than Opus. |
| Current + Claude Haiku | 87.7% | ~$0.10 | Anthropic-family cheap tier. |
| Current + gpt-4o-mini | 91.8–97.3% | ~$0.05 | Two runs, ±5-point variance. 20× cheaper than Sonnet, comparable preservation. **Recommended default.** |
| Current + gpt-5-mini | **100%** (73/73 unique facts) | ~$0.50–1 | Slowest run (reasoning tokens), highest preservation of anything tested. |
| Current + Llama 3.3-70B | 84.9% | ~$0.20 | Open-weights, **self-hostable** via Ollama / vLLM. Reasonable for users without API access. |
| Current + Qwen3-32B | 61.6% | $0.07 | Over-compresses; drops too many specifics. Not recommended. |

**What this shows:** the preservation improvement under the current prompt is a genuine architectural fix, the older prompt's "output ≈ size of ONE input memory" instruction was dropping ~25% of unique facts on historical merges. The rewrite (in [`mnemos/consolidation/prompts.py`](../mnemos/consolidation/prompts.py)) removes that size cap and adds a self-audit rule. Hierarchical binary merging for clusters >2 (in [`mnemos/consolidation/phases.py`](../mnemos/consolidation/phases.py)) is the accompanying algorithmic fix.

**Metric caveat:** the preservation metric uses an LLM judge for the match step (Claude Sonnet), which introduces some noise, run-to-run variance is ~±4-5 points at 30 clusters. Absolute numbers should be read with that buffer; cross-configuration deltas remain meaningful because the same judge is applied to all runs.

**Nothing is ever lost even at 75%.** Mnemos's two-tier recall architecture means archived originals are always available; when retrieval asks for a specific fact that did not survive into the consolidated memory, `auto_widen` falls back to the archived originals. Higher merge preservation reduces tier-2 widening frequency (faster queries, cleaner contexts) but does not change what is recoverable. The 80-90% target is about fast-path availability, not information loss.
