#!/usr/bin/env python3
"""
CML fidelity benchmark for Mnemos.

Measures how much atomic-fact content survives the prose → CML transformation
that the Nyx cycle (or a user deliberately cemelifying a memory) applies.

This is distinct from the LongMemEval --cml retrieval-parity runs, which show
that R@K is the same on prose and CML haystacks. R@K parity means "CML-encoded
context retrieves equally well", but it does NOT by itself guarantee "every
fact in the prose survived the encoding". That is what this bench measures.

Methodology (per memory):
  1) Extract atomic facts from the original prose via a Sonnet-class LLM
  2) Cemelify the prose into CML using the production MERGE_SYSTEM prompt
     adapted for single-input (tells the model to preserve every specific,
     same self-audit rule as the cluster merge).
  3) Extract atomic facts from the CML output the same way
  4) LLM-judge which original facts survived (PRESERVED / DROPPED)
  5) Aggregate: (#PRESERVED) / (#original facts)

Uses mnemos.consolidation.llm.chat() for every LLM call, so MNEMOS_LLM_API_URL,
MNEMOS_LLM_API_KEY, and MNEMOS_LLM_MODEL (+ optional per-phase overrides)
drive the provider/model choice. Runs without Mnemos being installed as a
package, just importable from this repo's root.

Fact extraction and matching are cached by content hash in
`./cml_fidelity_cache/` so reruns are free.

Usage:
    python cml_fidelity_bench.py                    # full corpus
    python cml_fidelity_bench.py --limit 5          # smoke test
    python cml_fidelity_bench.py --output out.json  # save here
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

# Make the repo root importable so `mnemos.*` resolves when running from
# benchmarks/ without installing the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mnemos.consolidation.llm import chat as llm_chat, is_configured  # noqa: E402
from mnemos.consolidation.prompts import MERGE_SYSTEM  # noqa: E402


def _model_tag() -> str:
    """Shorten the configured MNEMOS_LLM_MODEL to a cache-safe suffix."""
    m = os.environ.get("MNEMOS_LLM_MODEL_MERGE") or os.environ.get("MNEMOS_LLM_MODEL", "unset")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", m).strip("_") or "unset"


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cml_fidelity_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "cml_fidelity_corpus.json")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "cml_fidelity_results.json")


FACT_EXTRACT_PROMPT = """Extract every atomic fact from this memory. A fact is a single concrete statement; preserve specifics verbatim.

INCLUDE:
- Names (people, projects, services, products, tools, places)
- Numbers, amounts, quantities, versions, ports, IPs
- Dates, times, durations, deadlines
- Paths, URLs, identifiers
- Decisions and their reasoning
- Preferences (stated or implied)
- Configuration values, thresholds, flags
- Contact details, emails, phone numbers
- Relationships between entities ("X uses Y", "A replaces B")
- Warnings / caveats / gotchas

EXCLUDE:
- Prose filler, transitions, formatting
- Duplicates within this same memory
- Vague generalities with no specifics

Output one fact per line. No numbering, no bullets, no preamble.

MEMORY:
{text}"""


FACT_MATCH_PROMPT = """You are checking whether facts from an ORIGINAL memory survived a transformation into a COMPRESSED form.

For each fact in ORIGINAL_FACTS, check whether it is preserved in COMPRESSED. A fact is preserved if the same concrete content is present, even if phrased differently or collapsed with other facts. A fact is dropped if it is absent, vague, or lost its specifics (e.g. "a contact" when the original said "alex@example.com").

ORIGINAL_FACTS:
{originals}

COMPRESSED:
{compressed}

For each original fact, output one line:
PRESERVED: <the fact text>
or
DROPPED: <the fact text>

Output nothing else."""


CEMELIFY_USER_TEMPLATE = """Cemelify this single memory into a dense CML block. The goal is compression via format, not truncation of content. Preserve every specific from the input, every name, number, date, path, amount, decision, preference, warning. Drop only prose filler (transitions, narrative glue) and duplicated phrasing.

MEMORY:
{prose}"""


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def extract_facts(text: str) -> list:
    """LLM-extract atomic facts. Cached by hash."""
    h = _hash(text)
    cache_file = _cache_path(f"facts_{h}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    result = llm_chat(
        [{"role": "user", "content": FACT_EXTRACT_PROMPT.format(text=text)}],
        max_tokens=800, temperature=0.0,
    )
    if not result:
        return []
    facts = [ln.strip() for ln in result.strip().splitlines()
             if ln.strip() and not ln.strip().startswith(("#", "//", "-"))]
    # Drop leading "- " bullets if the model added them anyway
    facts = [re.sub(r"^[-*]\s*", "", f) for f in facts]
    facts = [f for f in facts if f]
    with open(cache_file, "w") as f:
        json.dump(facts, f)
    return facts


def cemelify(prose: str) -> str:
    """Prose → CML using the production MERGE_SYSTEM prompt. Cached per model."""
    h = _hash(prose)
    cache_file = _cache_path(f"cml_{_model_tag()}_{h}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)["cml"]

    user = CEMELIFY_USER_TEMPLATE.format(prose=prose)
    result = llm_chat(
        [
            {"role": "system", "content": MERGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=max(512, int(len(prose) * 1.2)),
        temperature=0.2,
        phase="MERGE",
    )
    if not result:
        return ""
    # Strip any accidental code fencing
    result = re.sub(r"^```\w*\n?", "", result)
    result = re.sub(r"\n?```$", "", result).strip()
    with open(cache_file, "w") as f:
        json.dump({"cml": result}, f)
    return result


def match_facts(originals: list, compressed: str) -> tuple:
    """LLM-judge which originals survived in compressed. Cached."""
    key = _hash(compressed + "|" + "||".join(originals))
    cache_file = _cache_path(f"match_{key}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            d = json.load(f)
            return d["preserved"], d["dropped"]

    orig_str = "\n".join(f"- {o}" for o in originals)
    result = llm_chat(
        [{"role": "user", "content": FACT_MATCH_PROMPT.format(
            originals=orig_str, compressed=compressed,
        )}],
        max_tokens=2000, temperature=0.0,
    )
    if not result:
        return [], list(originals)
    preserved, dropped = [], []
    for line in result.strip().splitlines():
        line = line.strip()
        if line.startswith("PRESERVED:"):
            preserved.append(line[len("PRESERVED:"):].strip())
        elif line.startswith("DROPPED:"):
            dropped.append(line[len("DROPPED:"):].strip())
    with open(cache_file, "w") as f:
        json.dump({"preserved": preserved, "dropped": dropped}, f)
    return preserved, dropped


def run(corpus_path: str, limit: int, output_path: str) -> dict:
    if not is_configured():
        print("ERROR: MNEMOS_LLM_API_KEY and MNEMOS_LLM_MODEL must both be set.",
              file=sys.stderr)
        print("The bench calls an LLM for fact extraction, cemelification, and matching.",
              file=sys.stderr)
        sys.exit(1)

    with open(corpus_path) as f:
        corpus = json.load(f)
    memories = corpus["memories"]
    if limit:
        memories = memories[:limit]

    _log(f"Running CML fidelity bench on {len(memories)} memories")

    per_memory = []
    total_facts = 0
    total_preserved = 0
    compression_samples = []

    for i, entry in enumerate(memories, 1):
        mid = entry["id"]
        prose = entry["prose"]
        category = entry.get("category", "")
        _log(f"[{i}/{len(memories)}] {mid} ({category}, {len(prose)} chars)")

        # Step 1: ground-truth facts
        originals = extract_facts(prose)
        if not originals:
            _log(f"  ! fact extraction failed for {mid}, skipping")
            continue

        # Step 2: cemelify
        cml = cemelify(prose)
        if not cml:
            _log(f"  ! cemelification failed for {mid}, skipping")
            continue

        # Step 3: fact match
        preserved, dropped = match_facts(originals, cml)
        n_orig = len(originals)
        n_kept = len(preserved)
        rate = (n_kept / n_orig) if n_orig else 0.0

        total_facts += n_orig
        total_preserved += n_kept
        compression_samples.append(len(cml) / max(1, len(prose)))

        _log(f"  preservation: {n_kept}/{n_orig} = {rate:.1%}  "
             f"compression: {len(cml)}/{len(prose)} = {len(cml)/max(1,len(prose)):.2f}")

        per_memory.append({
            "id": mid,
            "category": category,
            "prose_len": len(prose),
            "cml_len": len(cml),
            "compression_ratio": round(len(cml) / max(1, len(prose)), 3),
            "n_original_facts": n_orig,
            "n_preserved": n_kept,
            "preservation_rate": round(rate, 4),
            "dropped_facts": dropped,
            "cml_output": cml,
        })

    overall = (total_preserved / total_facts) if total_facts else 0.0
    avg_compression = (sum(compression_samples) / len(compression_samples)
                       if compression_samples else 0.0)

    summary = {
        "corpus": os.path.basename(corpus_path),
        "n_memories": len(per_memory),
        "total_facts": total_facts,
        "total_preserved": total_preserved,
        "overall_preservation_rate": round(overall, 4),
        "average_compression_ratio": round(avg_compression, 3),
        "per_memory": per_memory,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    _log("")
    _log(f"DONE. {len(per_memory)} memories, {total_facts} facts.")
    _log(f"  Overall preservation: {total_preserved}/{total_facts} = {overall:.1%}")
    _log(f"  Average compression:  {avg_compression:.2f}x")
    _log(f"  Results saved to {output_path}")

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help=f"Corpus JSON path (default: {DEFAULT_CORPUS})")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only run on the first N memories (0 = all)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"Results JSON path (default: {DEFAULT_OUTPUT})")
    args = ap.parse_args()
    run(args.corpus, args.limit, args.output)


if __name__ == "__main__":
    main()
