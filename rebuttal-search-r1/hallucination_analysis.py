"""
Post-hoc hallucination + KB-faithfulness analysis on hotpotqa-dev eval generations.

NOTE: src/eval_multihop.py:382 has a misalignment bug — phase1_info uses batch-
local idx against a global list, so only the FIRST batch (~64 examples) gets
correctly-aligned phase1.triplets per example. We work around this by filtering
to examples whose triplets are >=50% grounded in their listed context (this
naturally captures the aligned batch + any other coincidentally-aligned cases).

Compares replicate (f1 only) vs grounding vs hallucount on the SAME data.
Investigates: (a) num_triplets / num_hallu / hallu_rate per run on aligned subset,
(b) length-bias hypothesis — do hallucinations cluster at the end of generation?
"""
import json, os, re
from collections import defaultdict, Counter
from datasets import load_dataset

print("Loading hotpotqa validation ...", flush=True)
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                  trust_remote_code=True)
id_to_ctx = {}
for ex in ds:
    parts = []
    for t, ss in zip(ex["context"]["title"], ex["context"]["sentences"]):
        parts.append(t)
        parts.append(" ".join(ss))
    id_to_ctx[ex["id"]] = " ".join(parts).lower()
print(f"  Built {len(id_to_ctx)} contexts.", flush=True)

def grounded(triplet, ctx_lower):
    if not isinstance(triplet, (list, tuple)) or len(triplet) < 3:
        return (False, False, False)
    s = str(triplet[0]).strip().lower()
    o = str(triplet[2]).strip().lower()
    if not s or not o:
        return (False, s in ctx_lower if s else False, o in ctx_lower if o else False)
    return ((s in ctx_lower) and (o in ctx_lower), s in ctx_lower, o in ctx_lower)

RUNS = {
    "replicate": (
        "Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4"
        "-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak"
        "-ckpt500", "_0529_rep"),
    "grounding": (
        "Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4"
        "-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_grounding-2ph-prsft-wcount"
        "-th0.6-topk4-lam0.5-nak-ckpt500", "_0330"),
    "hallucount": (
        "Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4"
        "-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_hallucount-2ph-prsft-wcount"
        "-th0.6-topk4-lam0.5-nak-ckpt500", "_0330"),
}
BASE = "/home/lz586/icl/Multi-Hop-Reasoning/output/main_tables/two_phase/hotpotqa"

def find_gen_file(model_dir, sv):
    d = f"{BASE}/{model_dir}/generations{sv}"
    if not os.path.isdir(d):
        return None
    for f in os.listdir(d):
        if f.startswith("eval_hotpotqa_dev_") and f.endswith(".json"):
            return os.path.join(d, f)
    return None

def analyze(model_dir, sv, alignment_threshold=0.5):
    path = find_gen_file(model_dir, sv)
    if path is None:
        return None
    d = json.load(open(path))
    res = d["results"]
    aligned_examples = []        # list of (ex_id, n_triplets, n_hallu, list_of_(idx, is_hallu, reason))
    misaligned_count = 0
    # First pass: figure out which examples are aligned
    for ex_id, r in res.items():
        ctx = id_to_ctx.get(ex_id, "")
        if not ctx:
            continue
        triplets = r["phase1"]["triplets"]
        n = len(triplets)
        if n == 0:
            misaligned_count += 1
            continue
        per_triplet = []
        for i, t in enumerate(triplets):
            g, s_in, o_in = grounded(t, ctx)
            if g:
                reason = "grounded"
            elif s_in and not o_in:
                reason = "obj_missing"
            elif o_in and not s_in:
                reason = "subj_missing"
            else:
                reason = "both_missing"
            per_triplet.append((i, not g, reason))
        n_grounded = sum(1 for (_, h, _) in per_triplet if not h)
        if n_grounded / n >= alignment_threshold:
            aligned_examples.append((ex_id, n, n - n_grounded, per_triplet))
        else:
            misaligned_count += 1
    # Aggregate over aligned subset
    n_ex = len(aligned_examples)
    if n_ex == 0:
        return {"n_aligned": 0}
    nt = [e[1] for e in aligned_examples]
    nh = [e[2] for e in aligned_examples]
    # 10-bin relative-position hallu rate
    bin_h, bin_t = [0]*10, [0]*10
    # Reason breakdown
    rsn = Counter()
    # Tail-repetition: how many examples have last-20% triplets where one tuple repeats 5+ times
    tail_loop = 0
    # Per-decile *count* (not rate) of hallu — to see absolute distribution
    bin_count_h = [0]*10
    for ex_id, n, nh_i, per_t in aligned_examples:
        # Get triplets again to check tail loops
        triplets = res[ex_id]["phase1"]["triplets"]
        for i, is_hallu, reason in per_t:
            rel = min(int(10 * i / n), 9)
            bin_t[rel] += 1
            if is_hallu:
                bin_h[rel] += 1
                bin_count_h[rel] += 1
                rsn[reason] += 1
            else:
                rsn["grounded"] += 1
        # tail loop
        tail = triplets[-max(20, n//5):]
        ctr = Counter(tuple(map(str, t)) for t in tail)
        if ctr.most_common(1)[0][1] >= 5:
            tail_loop += 1
    return {
        "n_aligned": n_ex,
        "n_misaligned": misaligned_count,
        "mean_num_triplets": sum(nt)/n_ex,
        "median_num_triplets": sorted(nt)[n_ex//2],
        "mean_num_hallucinated": sum(nh)/n_ex,
        "overall_hallu_rate": sum(nh)/sum(nt) if sum(nt) else 0,
        "total_triplets": sum(nt),
        "total_hallu": sum(nh),
        "tail_loop_count": tail_loop,
        "tail_loop_frac": tail_loop/n_ex,
        "hallu_rate_by_relpos": [bin_h[i]/bin_t[i] if bin_t[i] else None for i in range(10)],
        "hallu_count_by_relpos": bin_count_h,  # absolute counts
        "total_by_relpos": bin_t,
        "reasons": dict(rsn),
    }

results = {}
for name, (mdir, sv) in RUNS.items():
    print(f"Analyzing {name} ...", flush=True)
    results[name] = analyze(mdir, sv)

print("\n" + "=" * 90)
print("ANALYSIS  (hotpotqa-dev, filtered to grounded_frac >= 0.5 per example)")
print("=" * 90)
keys = ["n_aligned","n_misaligned","mean_num_triplets","median_num_triplets",
        "mean_num_hallucinated","overall_hallu_rate","tail_loop_count","tail_loop_frac"]
hdr = "metric".ljust(28) + "".join(f"{r:>14}" for r in results)
print(hdr); print("-" * len(hdr))
for k in keys:
    cells = []
    for r in results:
        v = results[r].get(k, "—")
        if isinstance(v, float):
            cells.append(f"{v:>14.4f}")
        else:
            cells.append(f"{v:>14}")
    print(k.ljust(28) + "".join(cells))

print("\nHallu reason breakdown (% of ALL triplets in aligned subset):")
for r in results:
    rsn = results[r]["reasons"]
    tot = sum(rsn.values())
    print(f"  {r:<12}: grounded={rsn.get('grounded',0)/tot:.3%}  "
          f"obj_missing={rsn.get('obj_missing',0)/tot:.3%}  "
          f"subj_missing={rsn.get('subj_missing',0)/tot:.3%}  "
          f"both_missing={rsn.get('both_missing',0)/tot:.3%}")

print("\nHallu RATE by relative position (10 bins, start->end of triplet list):")
print(f"  {'run':<12}  " + "  ".join(f"bin{i}" for i in range(10)))
for r in results:
    vs = results[r]["hallu_rate_by_relpos"]
    print(f"  {r:<12}  " + "  ".join(f"{v:.3f}" if v else " --- " for v in vs))

print("\nHallu COUNT by relative position (absolute # of hallucinated triplets per bin):")
print(f"  {'run':<12}  " + "  ".join(f"bin{i}" for i in range(10)))
for r in results:
    cs = results[r]["hallu_count_by_relpos"]
    print(f"  {r:<12}  " + "  ".join(f"{c:>5}" for c in cs))

# Save raw
out_path = "/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1/hallucination_analysis.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved raw stats to {out_path}")
