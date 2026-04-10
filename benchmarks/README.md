# LongMemEval Benchmark for Mnemos

Reproducible benchmark of Mnemos against the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) dataset (ICLR 2025).

## Setup

```bash
# Download the dataset (~280MB)
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -O longmemeval_s.json
```

## Run

```bash
# Full benchmark (470 questions, ~6 hours)
python longmemeval_bench.py --mode hybrid

# Quick smoke test (5 questions, ~3 minutes)
python longmemeval_bench.py --mode hybrid --limit 5

# Other modes for comparison
python longmemeval_bench.py --mode fts            # BM25-only
python longmemeval_bench.py --mode vec            # vector-only
python longmemeval_bench.py --mode hybrid+rerank  # hybrid + Jina cross-encoder
```

## Results

Latest results from this repo (`results_hybrid_session.json`):

| Metric | Value |
|---|---|
| Recall@1 | 93.6% |
| Recall@3 | 97.4% |
| **Recall@5** | **98.1%** |
| Recall@10 | 98.7% |

Per question type breakdown is in the JSON file.

## How it works

For each question, the runner:

1. Builds a fresh corpus from the question's `haystack_sessions` (joining user turns per session)
2. Loads the corpus into a temporary SQLite DB with FTS5 + sqlite-vec
3. Embeds documents with FastEmbed e5-large
4. Runs the retrieval pipeline (FTS5 BM25 + vector search + RRF fusion, optionally + Jina rerank)
5. Maps document IDs back to session IDs
6. Computes Recall@K and NDCG@K against `answer_session_ids`

Abstention questions are excluded from retrieval metrics (per LongMemEval convention).
