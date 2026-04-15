# Mnemos Benchmarks

Reproducible benchmarks of Mnemos across multiple public datasets and Mnemos-specific quality measurements.

Five metric classes, clearly separated (see the [main README benchmark section](../README.md#benchmark) for the full discussion):

1. **LongMemEval retrieval recall**. R@K / NDCG@K on the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) dataset (ICLR 2025). Deterministic, no LLM in the measurement path.
2. **End-to-end QA accuracy**. LLM-judged answer correctness on all 500 LongMemEval questions including abstention. This is the metric most of the field publishes.
3. **LoCoMo retrieval recall**. R@K on the [LoCoMo](https://github.com/snap-research/locomo) dataset (ACL 2024). 1,540 evaluable QA pairs across 10 long conversations (19-32 sessions each), excluding 446 adversarial-by-design questions. Deterministic, no LLM in the measurement path. Top-K capped at 10 to avoid the "K >= session count" bypass.
4. **Consolidation quality**, unique-fact preservation rate across historical merge clusters. Mnemos-specific; measures how well the Nyx cycle merges without losing specifics.
5. **CML fidelity**, unique-fact preservation rate across a single prose → CML transformation step, on a small hand-curated corpus. Isolates the cemelification transform from the cluster-merge compression that the consolidation-quality bench conflates with it.

## Setup

```bash
# Download the dataset (~280MB)
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -O longmemeval_s.json
```

## 1. LongMemEval retrieval recall, `longmemeval_bench.py`

Runs the pure retrieval pipeline (no LLM, no answerer, no judge) and reports R@K + NDCG@K on the 470 non-abstention questions. Abstention questions are mathematically undefined for R@K (no gold session to retrieve) and are excluded per LongMemEval convention.

```bash
# All four canonical configurations
python longmemeval_bench.py --mode hybrid              # BM25 + vector, no rerank, no CML
python longmemeval_bench.py --mode hybrid+rerank       # + Jina cross-encoder
python longmemeval_bench.py --mode hybrid --cml        # + CML conversion at storage time
python longmemeval_bench.py --mode hybrid+rerank --cml # canonical Mnemos, all three

# Smoke test (5 questions, ~3 minutes)
python longmemeval_bench.py --mode hybrid+rerank --limit 5
```

Results from this repo:

| Configuration | R@1 | R@3 | R@5 | R@10 | Source |
|---|---|---|---|---|---|
| `hybrid` | 92.13% | 97.02% | 98.30% | 98.94% | [`results_hybrid_session.json`](results_hybrid_session.json) |
| `hybrid+rerank` | 94.3% | 98.1% | **98.94%** | 99.15% | [`results_hybrid+rerank_session.json`](results_hybrid+rerank_session.json) |
| `hybrid --cml` | 92.98% | 97.45% | 98.09% | 98.30% | [`results_hybrid_session_cml.json`](results_hybrid_session_cml.json) |
| `hybrid+rerank --cml` (canonical) | 95.74% | 98.72% | **99.15%** | 99.15% | [`results_hybrid+rerank_session_cml.json`](results_hybrid+rerank_session_cml.json) |

Per-question-type breakdowns in the JSON files. See [main README §Benchmark](../README.md#benchmark) for the full per-mode discussion and CML-methodology disclosure.

**Pipeline per question:** build corpus from `haystack_sessions` → SQLite FTS5 + sqlite-vec → FastEmbed e5-large → hybrid search (BM25 + vector + RRF + optional Jina rerank) → map doc IDs back to session IDs → compute R@K / NDCG@K against `answer_session_ids`.

## 2. End-to-end QA accuracy, `longmemeval_qa.py`

Retrieve top-K sessions → answer with an LLM → judge the answer against the reference with another LLM. This is the metric most of the field publishes (Mastra, Emergence, Mem0, Supermemory, etc.). Runs against all 500 LongMemEval questions including the 30 abstention cases (where the correct behavior is refusal).

```bash
# Default: hybrid retrieval + Claude Sonnet answerer + Claude Opus judge, K=5
python longmemeval_qa.py

# Paper-convention (gpt-4o as both answerer and judge)
python longmemeval_qa.py --answerer gpt4o --judge gpt4o

# With per-session CML compression of retrieved context
python longmemeval_qa.py --answerer gpt4o --judge gpt4o --cml-context

# Smoke test
python longmemeval_qa.py --limit 10
```

Results from this repo:

| Answerer | Context | Judge | Overall | Source |
|---|---|---|---|---|
| Sonnet | raw sessions | Opus | **77.4%** | [`qa_results_hybrid_k5_sonnet_opus.json`](qa_results_hybrid_k5_sonnet_opus.json) |
| gpt-4o | CML-compressed | gpt-4o | 58.6% | [`qa_results_hybrid_k5_gpt4o_gpt4o_cml.json`](qa_results_hybrid_k5_gpt4o_gpt4o_cml.json) |

Per-type breakdown for the Sonnet+raw baseline:

| Question Type | Accuracy | N |
|---|---|---|
| abstention (correct refusal) | 96.7% | 30 |
| single-session-assistant | 94.6% | 56 |
| single-session-user | 93.8% | 64 |
| knowledge-update | 81.9% | 72 |
| temporal-reasoning | 72.4% | 127 |
| multi-session | 66.1% | 121 |
| single-session-preference | 46.7% | 30 |

The CML-compressed-context configuration scores lower on multi-session and preference because per-session CML compression destroys cross-session cardinality signals. The production Nyx consolidation cycle addresses this by merging related memories into one consolidated fact before retrieval, see the consolidation benchmark below.

**Methodology:**
- Answerer receives top-K retrieved sessions as raw text (or CML-compressed with `--cml-context`) plus the question
- Judge receives the question, the reference answer, and the system's answer, with per-question-type guidance (from LongMemEval paper)
- Temperature 0 on both LLMs
- Same-family judge bias caveat: Sonnet+Opus and gpt-4o+gpt-4o pairs are same-family. Numbers may shift ±3 points under a cross-family judge

### Re-judging saved results, `longmemeval_rejudge.py`

Re-runs only the judge step on saved per-question records from a previous QA run. Useful for measuring judge bias or swapping in a different standardized judge without re-running the retrieval + answerer.

```bash
python longmemeval_rejudge.py qa_results_hybrid_k5_sonnet_opus.json --judge gpt4o
```

### Option-B consolidation QA, `longmemeval_consolidated.py`

Experimental: CML-compress each haystack session, extract fact-level signals, cluster same-subject facts, merge via LLM, retrieve at fact level. Was built to test whether pre-retrieval consolidation helps LongMemEval multi-session questions. Result: LongMemEval's haystacks are topically disjoint conversations, not accumulating user memory, so consolidation doesn't find merge candidates. Useful as a reference implementation for the consolidation approach but not a quality lift on this benchmark.

## 3. LoCoMo retrieval recall, `locomo_bench.py`

[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024) is a long-conversation memory benchmark: 10 conversations, 19-32 sessions each, 1,986 QA pairs across 5 categories. The mean session length is 2,843 characters (median 2,652, p90 4,090).

```bash
# Download the dataset (~3MB)
wget -O locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# Run a mode (CPU-only, no LLM in retrieval)
python locomo_bench.py --mode hybrid
python locomo_bench.py --mode hybrid+rerank --cml
python locomo_bench.py --mode hybrid --cml
```

**Methodology guardrails (honest baselines on LoCoMo are easy to fake; we deliberately do not):**

- **Adversarial questions excluded from R@K.** Category 5 (446 questions, ~22% of the dataset) is adversarial by design: the answer is NOT in the conversation, so R@K is mathematically undefined. The bench evaluates on the 1,540 answerable questions, same convention LongMemEval uses for abstention.
- **Top-K capped at 10.** The smallest LoCoMo conversation has 19 sessions. Top-K ≥ 19 returns every session and the retrieval stage stops doing real work; published R@K results in that regime measure how well the LLM reranker filters a complete dump, not retrieval recall. Mnemos caps K at 10 across the board, comfortably below every conversation's session count.
- **No LLM in the retrieval path.** The cross-encoder reranker (Jina v2) is a local discriminative ONNX scorer, not a generative model. Zero outbound API calls during retrieval.
- **Per-conversation session counts are published** alongside the result so anyone can verify K stayed below the bypass threshold.

Results from this repo:

| Mode | R@1 | R@3 | R@5 | R@10 | NDCG@5 | Source |
|---|---|---|---|---|---|---|
| `hybrid` (BM25 + vector + RRF, no LLM) | 57.9% | 77.0% | **84.7%** | **94.0%** | 77.1% | [`results_locomo_hybrid.json`](results_locomo_hybrid.json) |
| `hybrid --cml` | 49.0% | 69.9% | 79.4% | 91.0% | 70.1% | [`results_locomo_hybrid_cml.json`](results_locomo_hybrid_cml.json) |
| `hybrid+rerank --cml` (CML + Jina rerank) | 60.9% | 79.7% | **86.1%** | 91.9% | 79.7% | [`results_locomo_hybrid+rerank_cml.json`](results_locomo_hybrid+rerank_cml.json) |

Per-category breakdown (hybrid+rerank --cml):

| Category | R@1 | R@3 | R@5 | R@10 | N |
|---|---|---|---|---|---|
| single-hop | 59.6% | 79.4% | 85.8% | 94.0% | 282 |
| multi-hop | 68.5% | 82.9% | 88.2% | 93.5% | 321 |
| temporal-reasoning | 37.0% | 57.6% | 66.3% | 77.2% | 92 |
| open-domain | 61.1% | 81.0% | 87.5% | 92.3% | 841 |

**Why is `hybrid+rerank` (without CML) not in the table?** LoCoMo sessions are long: median 2,652 characters, p90 4,090. The Jina cross-encoder's attention window cannot see a whole session at once when scoring relevance, so we tested two configurations and neither was acceptable: (1) full-text rerank takes ~26 minutes per conversation on CPU (extrapolating to 4+ hours per benchmark run, infeasible); (2) truncating to the first 2,000 characters before rerank scored 78.0% R@5 on `conv-26`, net-negative against pure `hybrid`'s 86.0% on the same conversation, because the truncation cut off the very evidence the cross-encoder was supposed to score. CML preprocessing (which compresses each session to ~500 characters of dense facts) is therefore the prerequisite for effective cross-encoder reranking on long-session benchmarks. The published `hybrid+rerank --cml` row is the configuration that pairs the cross-encoder with text it can actually see in full.

**Reading the numbers:**

- The recommended deployment for LoCoMo-shape data (long, multi-session conversations) is **either** `hybrid` (highest R@10 at 94.0%, simplest, no LLM cost anywhere) **or** `hybrid+rerank --cml` (highest R@5 at 86.1%, with a CML compression cost amortized at index time). The choice is a small recall-position tradeoff against compression cost.
- `hybrid --cml` (CML at index but no rerank) underperforms plain `hybrid`. CML is mildly out-of-distribution for the e5-large bi-encoder at first-stage retrieval; on long-session inputs that gap is more visible than on the LongMemEval short-session inputs. Rerank recovers the loss and then some.
- Temporal-reasoning is the hardest category across all modes (66.3% R@5 even with CML+rerank). Multi-session temporal questions require composing facts across non-adjacent sessions, which retrieval alone does not solve; this is an answerer-side problem, not a retrieval problem.

## 4. Consolidation quality (fact preservation)

Separate benchmark that does not use LongMemEval. Instead, it measures unique-fact preservation on 30 historical merge events from an actual Mnemos production memory store, traceable via `merged-into-<id>` tags on archived originals.

Current data (this repository's production benchmark, run against a personal memory store):

| Configuration | Preservation | Cost/run |
|---|---|---|
| Older MERGE_SYSTEM (Opus) | 75.3% | ~$6 |
| Current MERGE_SYSTEM + hierarchical binary merge (Opus) | 89.0% | ~$6 |
| Current + Claude Sonnet | 90.4–94.5% | ~$1 |
| Current + Claude Haiku | 87.7% | ~$0.10 |
| Current + gpt-4o-mini | 91.8–97.3% | ~$0.05 |
| Current + gpt-5-mini | **100%** (73/73), slow, reasoning tokens | ~$0.50-1 |
| Current + Llama 3.3-70B | 84.9% | ~$0.20 |
| Current + Qwen3-32B | 61.6% | $0.07 |

See [main README §Consolidation quality](../README.md#consolidation-quality-fact-preservation-in-the-nyx-cycle) for methodology and the tier-2 recall framing.

## 5. CML fidelity (format-level content parity), `cml_fidelity_bench.py`

Answers the question: **does CML as a storage format preserve the same information as prose?** In normal use an agent writes memories directly in CML, so there is no transformation step at all; this bench asks whether, if you *had* written the same content as prose, anything would have been lost by choosing CML instead. The LongMemEval `--cml` runs already prove ranking parity (R@K is basically identical on prose vs CML haystacks); this benchmark proves *content* parity in the same direction.

The prose → CML transformation is used here as the measurement lens, rewrite a prose memory in CML using the same prompt the Nyx cycle uses, then check whether every atomic fact survived. If the transform preserves content, the format is sufficient to express the same information density. The Nyx merge step is a separate (harder) test covered in section 3.

**Corpus** (`cml_fidelity_corpus.json`): 20 hand-curated prose memories in two input styles so both ends of the compression range are actually measured, not just asserted.

| Subset | Entries | Ids | Style | Atomic facts | Prose length |
|---|---|---|---|---|---|
| Fact-dense | 15 | `cfg-*` `dec-*` `con-*` `pref-*` `fact-*` `warn-*` `learn-*` `runbook-*` | short, minimal filler, reads like a structured note a user or agent would log directly | 145 | 302–440 chars |
| Narrative | 5 | `narr-01…narr-05` | longer, rambling, full of connective tissue ("So I was trying to…", "After a few hours I realized…"); reads like someone dictating what happened | 64 | 804–1023 chars |
| **Total** | **20** | | | **209** | |

The fact-dense subset is the realistic ceiling for most agent-written memories. The narrative subset is the realistic ceiling for memories a user *dictates* or a raw conversation log that has not yet been structured. The two subsets together show the full compression range rather than the narrow band you would see from either alone. Both categories were authored by hand and then fact-extracted by a Sonnet-class judge; the extraction cache is committed in `cml_fidelity_cache/facts_*.json` so downstream measurements share one canonical ground truth per memory.

**Pipeline per memory:**
1. Fact-extract the original prose with an LLM
2. Cemelify the prose with the production `MERGE_SYSTEM` prompt (adapted for single-input)
3. Fact-extract the CML output
4. LLM-judge which originals survived via a PRESERVED/DROPPED format

Every LLM call goes through `mnemos.consolidation.llm.chat()`, so the configured `MNEMOS_LLM_MODEL` (and `MNEMOS_LLM_MODEL_MERGE` if set) drive the measurement. Fact extraction is cached by content hash; cemelification and matching are cached per (model, content) so swapping the model invalidates only the relevant cache entries.

```bash
# Canonical run (Sonnet 4.6 via DO Gradient in the repo's reference config)
MNEMOS_LLM_MODEL=anthropic-claude-4.6-sonnet python cml_fidelity_bench.py

# Smoke test
python cml_fidelity_bench.py --limit 3
```

Results from this repo (sorted by overall preservation). Each model row reports **dense / narrative / overall** so you can see how the input style changes both preservation and compression:

| Model | Dense (15 memories, 145 facts) | Narrative (5 memories, 64 facts) | Overall (20 memories, 209 facts) | Source |
|---|---|---|---|---|
| Claude Opus 4.6 | 100.0% @ 0.86× | 100.0% @ **0.59×** | **100.0%** @ 0.79× | [`cml_fidelity_opus.json`](cml_fidelity_opus.json) |
| Claude Sonnet 4.6 | 100.0% @ 0.78× | 93.8% @ **0.52×** | 98.1% @ 0.71× | [`cml_fidelity_sonnet.json`](cml_fidelity_sonnet.json) |
| Claude Haiku 4.5 | 97.9% @ 0.74× | 98.4% @ **0.55×** | 98.1% @ 0.69× | [`cml_fidelity_haiku.json`](cml_fidelity_haiku.json) |
| gpt-4o (full) | 98.6% @ 0.78× | 90.6% @ **0.39×** | 96.2% @ 0.68× | [`cml_fidelity_gpt4o.json`](cml_fidelity_gpt4o.json) |
| gpt-4o-mini | 96.6% @ 0.85× | 92.2% @ **0.48×** | 95.2% @ 0.75× | [`cml_fidelity_gpt4o_mini.json`](cml_fidelity_gpt4o_mini.json) |
| Llama 3.3-70B | 94.5% @ 0.85× | 75.0% @ 0.60× | 88.5% @ 0.79× | [`cml_fidelity_llama70b.json`](cml_fidelity_llama70b.json) |
| Qwen3-32B (partial 9/20) | 62.2% @ 0.57× | 96.2% @ 0.56× | 80.6% @ 0.57× | [`cml_fidelity_qwen3.json`](cml_fidelity_qwen3.json) |
| Minimax m2.5 (partial 19/20) | 59.4% @ 0.67× | 42.2% @ 0.56× | 54.0% @ 0.64× | [`cml_fidelity_minimax.json`](cml_fidelity_minimax.json) |

**Short version if you are on Claude:** every Claude tier lands at or above 98% overall preservation. Opus is a clean sweep (100% on every memory in both subsets); Sonnet and Haiku tie at 98.1% overall with different distributions (Sonnet stronger on dense, Haiku stronger on narrative). There is no accuracy reason to run a non-Claude model for the cemelification path on a Claude-driven deployment; pick on cost.

A few observations the numbers surface:

- **Narrative compression validates the 40–60% claim.** On narrative prose with lots of connective tissue, **gpt-4o lands at 0.39× (61% smaller)** at 90.6% preservation, gpt-4o-mini at 0.48× (52%), Sonnet at 0.52× (48%), Haiku at 0.55× (45%), Opus at 0.59× (41%). The spread is the honest "up to 60%" number the README quotes; the fact-dense subset lives in the 14–26% range as a separate regime.
- **gpt-4o inverts its dense ranking on narrative.** It scored similarly to gpt-4o-mini on dense (98.6% vs 96.6%) but its aggressive rewrite style becomes the single best narrative compression in the table (0.39×) once there is actual filler to cut. The pattern: on tight inputs the full model cuts just fine, on loose inputs it cuts harder than anything else.
- **Opus vs Sonnet on single-memory.** Opus scored 89.0% on the cluster-merge bench (section 3) but hits 100% overall here, every single fact in every single memory. The pattern fits the "Opus overthinks compression across multiple inputs" theory: with no cross-input trade-off to fret about, it just preserves everything. Sonnet sits at 98.1%, mostly losing specifics on narrative.
- **Haiku beats Sonnet on narrative.** 98.4% vs 93.8% at 0.55× vs 0.52× compression. Haiku's shorter-output bias (well-tuned for tight output) happens to line up with narrative prose's high redundancy, it can cut filler without cutting facts.
- **Llama 3.3-70B degrades more on narrative.** 94.5% dense → 75.0% narrative. The longer, more diffuse inputs stretch its preservation below the safe zone in a way that the Claude tiers and gpt-4o-mini resist. Still usable on fact-dense inputs, harder to trust on genuine rambling narrative.
- **Qwen3-32B and Minimax m2.5 are the clear outliers, not open-weights in general.** Llama 3.3-70B overall 88.5%, Qwen3 partial at 80.6%, Minimax m2.5 at 54.0% (partial 19/20 with one timeout). Both over-compress aggressively and drop specifics. The takeaway is "pick the right model", not "avoid open-weights"; Llama 3.3-70B is a reasonable open option, Qwen3-32B and Minimax m2.5 on this endpoint are not.
- **Compression varies by model even at the same preservation.** Haiku at 0.74× dense / 0.55× narrative and Llama at 0.85× / 0.60× both land in similar preservation bands on dense, but Haiku buys a 10-point compression win at a 3-point preservation cost. Which trade you want depends on whether tokens or fact density is the bottleneck for your deployment.

**What this bench answers:** when Mnemos converts a prose memory into CML, either as part of the Nyx cycle merge step, or when the user deliberately rewrites prose as CML, are the atomic facts preserved? Result: yes for every hosted frontier model on fact-dense prose, and yes for the Claude tier + gpt-4o-mini on narrative prose at 90–99% while saving 44–51% of the tokens. Output-to-input length ratio varies (0.36× to 1.28×), so if compression matters to your deployment as much as preservation, check both columns.

**Limitations:** the corpus is small (20 memories) and author-curated, so fact density and phrasing style may not match all real-world workloads. The fact-extraction and fact-matching judge is the same LLM class used for cemelification, so there is a mild same-family bias in the measurement path. This bench is complementary to section 3, not a replacement.

## Utilities

### `cml_session.py`

Haiku-based session → CML compressor used by `longmemeval_qa.py --cml-context` and by the `longmemeval_consolidated.py` pipeline. Caches outputs in a local SQLite by SHA256 of the session content so repeat runs are effectively free.

## What each script produces

Every script writes a JSON file with the full per-question (or per-cluster) result. These files are committed to the repo so the numbers in the main README and in the tables above can be verified against their source. If you re-run a benchmark, you'll overwrite the corresponding JSON with your result; `git diff` will show any deltas.
