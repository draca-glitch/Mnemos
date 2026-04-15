#!/usr/bin/env python3
"""
Re-judge saved LongMemEval QA results with a different judge model.

Takes a qa_results_*.json produced by longmemeval_qa.py and re-runs the
judge step only (no retrieval, no answerer). Useful for judge-bias
ablations and for adding the standardized gpt-4o-judge number to results
originally judged by Opus.

Usage:
    python longmemeval_rejudge.py <input.json> [--judge gpt4o|opus|sonnet]
                                               [--out rejudged.json]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/root/scripts")

from longmemeval_qa import JUDGE_PROMPT, JUDGE_GUIDANCE, JUDGE_DEFAULT, judge_verdict, DATA_PATH, JUDGES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="qa_results_*.json from longmemeval_qa.py")
    p.add_argument("--judge", choices=list(JUDGES), default="gpt4o")
    p.add_argument("--out", help="output path (default: <input>.rejudged_<judge>.json)")
    args = p.parse_args()

    with open(args.input) as f:
        orig = json.load(f)
    with open(DATA_PATH) as f:
        dataset = json.load(f)
    qid_to_question = {x["question_id"]: x["question"] for x in dataset}
    qid_to_qtype = {x["question_id"]: x["question_type"] for x in dataset}

    judge_fn = JUDGES[args.judge]
    records = [q for q in orig["per_question"] if q.get("system_answer") is not None]
    print(f"Re-judging {len(records)} records with {args.judge} (original judge: {orig.get('judge')})")

    correct_total = 0
    type_correct = defaultdict(int)
    type_count = defaultdict(int)
    flips = 0
    judge_fails = 0
    new_per_q = []

    for i, rec in enumerate(records):
        qid = rec["qid"]
        question = qid_to_question.get(qid, "")
        qtype = qid_to_qtype.get(qid, rec.get("type", ""))
        is_abs = "_abs" in qid
        type_key = rec.get("type") or ("abstention" if is_abs else qtype)

        verdict, raw = judge_verdict(
            judge_fn, question, rec["reference"], rec["system_answer"], qtype, is_abs
        )
        if verdict is None:
            judge_fails += 1
            new_per_q.append({**rec, "new_correct": None, "new_judge_raw": raw})
            continue
        type_count[type_key] += 1
        if verdict:
            correct_total += 1
            type_correct[type_key] += 1
        if rec.get("correct") is not None and rec["correct"] != verdict:
            flips += 1
        new_per_q.append({**rec, "new_correct": verdict})

        if (i + 1) % 25 == 0 or i == 0:
            evaluated = sum(type_count.values())
            acc = correct_total / max(1, evaluated)
            print(f"  [{i+1:3d}/{len(records)}] acc={acc:.3f} ({correct_total}/{evaluated}) | flips={flips}")

    evaluated = sum(type_count.values())
    overall = correct_total / max(1, evaluated)
    print("\n" + "=" * 72)
    print(f"Re-judged accuracy with {args.judge}: {overall:.1%} ({correct_total}/{evaluated})")
    print(f"Original ({orig.get('judge')}) accuracy: {orig.get('overall_accuracy'):.1%}")
    print(f"Flips: {flips} | judge_fails: {judge_fails}")
    print("\nPer-type (new judge):")
    for t in sorted(type_count):
        n = type_count[t]
        a = type_correct[t] / n if n else 0.0
        print(f"  {t:<28} {a:>6.1%} {n:>5}")

    out = {
        **{k: v for k, v in orig.items() if k != "per_question"},
        "original_judge": orig.get("judge"),
        "judge": args.judge,
        "overall_accuracy": overall,
        "per_type": {t: {"accuracy": type_correct[t] / type_count[t] if type_count[t] else 0.0,
                         "n": type_count[t]} for t in type_count},
        "judge_flips_vs_original": flips,
        "judge_fails": judge_fails,
        "per_question": new_per_q,
    }
    out_path = args.out or args.input.replace(".json", f".rejudged_{args.judge}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
