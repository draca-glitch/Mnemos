#!/usr/bin/env python3
"""Pass 2: line-level feed calibration for the NLI store decision layer.

Pass 1 (bench.py) showed blob-level NLI misses real contradictions buried in
long consolidated records. This pass scores STATEMENT pairs instead:

  contradiction: e5-embed all lines, preselect top-K most-similar cross line
    pairs, NLI both directions, aggregate max P(contra) over line pairs.
  dedup (comparison row): per-line coverage, entailment of each line by its
    best-matching counterpart, aggregated as min of the two directions' means.

Arms: native and english (translations.json is line-aligned by construction).
Models: deberta-v3-mnli-fever-anli (english), mdeberta-xnli (native).
Includes the acid-test assertion that p011's conflicting diagnosis lines are
among the preselected pairs.

Output: results_pass2.json + report_pass2.md.
Usage: /root/venvs/ai/bin/python bench_lines.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/work/mnemos")

from bench import NliScorer, sha, metrics, CONTRA_POS, DUP_POS  # noqa: E402
from mnemos.embed import embed  # noqa: E402

TOP_K = 8
MAX_LINES = 30
MIN_LINE_CHARS = 10
NLI_MAX_LEN = 256

MODELS = {
    "deberta-en": ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", ("english",)),
    "mdeberta": ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                 ("native",)),
}


def lines_of(text):
    out = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= MIN_LINE_CHARS]
    return out[:MAX_LINES] or [text.strip()[:500]]


def load_pairs():
    pairs = []
    for fn in ("pairs.jsonl", "synth_pairs.jsonl"):
        for line in open(fn, encoding="utf-8"):
            pairs.append(json.loads(line))
    labels = json.load(open("labels.json"))
    out = []
    for p in pairs:
        lab = p.get("label") or labels.get(p["pair_id"], {}).get("label")
        if lab:
            p["label"] = lab
            out.append(p)
    return out


def main():
    pairs = load_pairs()
    tr = json.load(open("translations.json"))

    def eng(t):
        return tr.get(sha(t), t)

    # --- 1. split into lines per arm and embed every unique line, batched ---
    per_pair = {}
    all_lines = {}
    for p in pairs:
        for arm in ("native", "english"):
            get = (lambda t: t) if arm == "native" else eng
            la, lb = lines_of(get(p["a_content"])), lines_of(get(p["b_content"]))
            per_pair[(p["pair_id"], arm)] = (la, lb)
            for ln in la + lb:
                all_lines[ln] = None
    uniq = list(all_lines)
    print(f"{len(pairs)} pairs; {len(uniq)} unique lines to embed", file=sys.stderr)
    t0 = time.time()
    vecs = embed(uniq, prefix="passage")
    mat = np.array(vecs, dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    line_vec = {ln: mat[i] for i, ln in enumerate(uniq)}
    print(f"embedded in {time.time() - t0:.0f}s", file=sys.stderr)

    def preselect(la, lb, k=TOP_K):
        va = np.stack([line_vec[l] for l in la])
        vb = np.stack([line_vec[l] for l in lb])
        sims = va @ vb.T
        flat = [(float(sims[i, j]), i, j)
                for i in range(len(la)) for j in range(len(lb))]
        flat.sort(reverse=True)
        return flat[:k]

    # --- acid check: p011 diagnosis lines must be preselected ---
    la, lb = per_pair[("p011", "native")]
    sel = preselect(la, lb)
    hit = any(("renal cell carcinoma" in la[i] or "njurcancer" in la[i].lower())
              and ("yeloprolif" in lb[j] or "yeloid" in lb[j])
              for _, i, j in sel)
    print(f"ACID preselect check p011 (native): diagnosis pair in top-{TOP_K}: {hit}",
          file=sys.stderr)

    # --- 2. NLI over preselected line pairs ---
    results = {p["pair_id"]: {"label": p["label"]} for p in pairs}
    latency = {}
    for name, (model_id, arms) in MODELS.items():
        print(f"loading {name}...", file=sys.stderr)
        scorer = NliScorer(model_id)

        def nli(a, b):
            import torch
            with torch.no_grad():
                enc = scorer.tok(a, b, return_tensors="pt",
                                 truncation=True, max_length=NLI_MAX_LEN)
                probs = torch.softmax(scorer.model(**enc).logits[0], dim=-1)
            return (float(probs[scorer.idx["entail"]]),
                    float(probs[scorer.idx["contra"]]))

        for arm in arms:
            t0 = time.time()
            for p in pairs:
                la, lb = per_pair[(p["pair_id"], arm)]
                sel = preselect(la, lb)
                # contradiction: max P(contra) over selected line pairs, both dirs
                best_contra = 0.0
                ent_ab = {}
                ent_ba = {}
                for _, i, j in sel:
                    e1, c1 = nli(la[i], lb[j])
                    e2, c2 = nli(lb[j], la[i])
                    best_contra = max(best_contra, c1, c2)
                    ent_ab[i] = max(ent_ab.get(i, 0.0), e1)
                    ent_ba[j] = max(ent_ba.get(j, 0.0), e2)
                # dedup coverage: mean best-entailment per line, worst direction.
                # Lines never preselected count as uncovered (0), so a long
                # memory with unique lines cannot look like a duplicate.
                cov_a = sum(ent_ab.get(i, 0.0) for i in range(len(la))) / len(la)
                cov_b = sum(ent_ba.get(j, 0.0) for j in range(len(lb))) / len(lb)
                r = results[p["pair_id"]]
                r[f"{name}:contra:{arm}"] = round(best_contra, 4)
                r[f"{name}:cover:{arm}"] = round(min(cov_a, cov_b), 4)
            latency[f"{name}:{arm}"] = (time.time() - t0) / len(pairs)
        del scorer

    json.dump({"results": results, "latency_s_per_pair": latency,
               "acid_preselect_p011": hit},
              open("results_pass2.json", "w"), indent=1)

    # --- 3. metrics + acid test on the three real contradictions ---
    labels = [r["label"] for r in results.values()]
    y_contra = [l in CONTRA_POS for l in labels]
    y_dup = [l in DUP_POS for l in labels]

    lines = ["# NLI bench pass 2 (line-level feed)", "",
             f"pairs: {len(pairs)}; preselect top-{TOP_K} line pairs by e5 cosine; "
             f"NLI max_len {NLI_MAX_LEN}", "",
             f"acid preselect check (p011 diagnosis pair in top-{TOP_K}): {hit}", ""]
    for task, y, kind in (("contradiction (pos=contradicts+evolves)", y_contra, "contra"),
                          ("dedup coverage (pos=duplicate)", y_dup, "cover")):
        lines += [f"## {task}", "",
                  "| scorer | arm | AUC | bestF1 | prec | rec | thr | fp | fn |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key in sorted({k for r in results.values() for k in r if f":{kind}:" in k}):
            s = [results[p["pair_id"]].get(key) for p in pairs]
            if any(v is None for v in s):
                continue
            m = metrics(y, s)
            scorer_name, arm = key.rsplit(":", 1)
            lines.append(
                f"| {scorer_name} | {arm} | {m.get('auc')} | {m.get('f1')} | "
                f"{m.get('precision')} | {m.get('recall')} | {m.get('threshold')} | "
                f"{m.get('fp')} | {m.get('fn')} |")
        lines.append("")
    lines += ["## real-contradiction acid test (blob-level missed all three)", ""]
    for pid in ("p010", "p011", "p018"):
        r = results[pid]
        vals = {k: v for k, v in r.items() if ":contra:" in k}
        lines.append(f"- {pid} [{r['label']}]: " +
                     ", ".join(f"{k}={v}" for k, v in sorted(vals.items())))
    lines += ["", "## latency (s/pair incl. preselect NLI calls, CPU)", ""] + [
        f"- {k}: {v:.3f}" for k, v in sorted(latency.items())]
    open("report_pass2.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
