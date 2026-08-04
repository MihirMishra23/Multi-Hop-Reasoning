"""
Richer KB/hallucination analysis on the aligned subset (~64 examples per run),
with proper deduplication + entity-level + per-part-position breakdowns.

Adds, vs the v1 analysis:
  - Dedup at the (subj, relation, object) level AND at (subj, object) level
  - Unique vs duplicate per example
  - Repetition distribution (how many copies of each triplet)
  - Length-bias of DUPLICATION (is repetition concentrated at the tail?)
  - Entity collapse: dominant subject/object distribution
  - Grounded vs hallucinated after dedup
"""
import json, os
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
        return False
    s = str(triplet[0]).strip().lower()
    o = str(triplet[2]).strip().lower()
    if not s or not o:
        return False
    return (s in ctx_lower) and (o in ctx_lower)

RUNS = {
    "replicate": (
        "Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4"
        "-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak"
        "-ckpt500", "_0529_phase1fix"),
    "grounding": (
        "Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4"
        "-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_grounding-2ph-prsft-wcount"
        "-th0.6-topk4-lam0.5-nak-ckpt500", "_0529_phase1fix"),
    "hallucount": (
        "Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4"
        "-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_hallucount-2ph-prsft-wcount"
        "-th0.6-topk4-lam0.5-nak-ckpt500", "_0529_phase1fix"),
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

def norm_triplet(t):
    """Normalize a triplet for hashing. lower-strip on each part."""
    if not isinstance(t, (list, tuple)) or len(t) < 3:
        return None
    return (str(t[0]).strip().lower(),
            str(t[1]).strip().lower(),
            str(t[2]).strip().lower())

def analyze(model_dir, sv, alignment_threshold=0.5):
    path = find_gen_file(model_dir, sv)
    if path is None:
        return None
    d = json.load(open(path))
    res = d["results"]
    per_ex = []        # records for aligned examples
    n_misaligned = 0
    for ex_id, r in res.items():
        ctx = id_to_ctx.get(ex_id, "")
        if not ctx: continue
        triplets = r["phase1"]["triplets"]
        n = len(triplets)
        if n == 0: continue
        n_g = sum(1 for t in triplets if grounded(t, ctx))
        if n_g / n < alignment_threshold:
            n_misaligned += 1
            continue
        per_ex.append((ex_id, triplets, ctx, n_g))
    # ------- AGGREGATE -------
    out = {
        "n_aligned": len(per_ex),
        "n_misaligned": n_misaligned,
    }
    # Per-example raw + dedup counts
    raw_total = 0
    raw_grounded_total = 0
    uniq_spo_total = 0       # unique (subj, rel, obj)
    uniq_spo_grounded_total = 0
    uniq_so_total = 0        # unique (subj, obj) — collapses relation paraphrases
    uniq_so_grounded_total = 0
    # Repetition stats
    rep_freq_distrib = Counter()  # repetition count -> # of unique triplets with that count
    # Position-of-duplicates
    bin_dup, bin_total = [0]*10, [0]*10
    # Entity collapse: distribution of subject frequency per example
    top_subject_share = []
    top1_subject_grounded = []  # is the most-frequent subject grounded?
    # Tail-repetition: any single (s,r,o) repeated >=5 times in last 20% of triplets
    tail_loop = 0
    # length-bias of HALLU on UNIQUE triplets only
    uniq_bin_h, uniq_bin_t = [0]*10, [0]*10
    for ex_id, triplets, ctx, _ng in per_ex:
        n = len(triplets)
        raw_total += n
        spo = Counter()
        so  = Counter()
        first_pos = {}  # first position of each unique triplet
        for i, t in enumerate(triplets):
            spo_k = norm_triplet(t)
            if spo_k is None: continue
            so_k = (spo_k[0], spo_k[2])
            spo[spo_k] += 1
            so[so_k] += 1
            first_pos.setdefault(spo_k, i)
        # Count grounded for raw
        raw_grounded_total += sum(1 for t in triplets if grounded(t, ctx))
        # Count grounded for unique-spo
        for k in spo:
            uniq_spo_total += 1
            if (k[0] in ctx) and (k[2] in ctx):
                uniq_spo_grounded_total += 1
        # Count grounded for unique-so
        for k in so:
            uniq_so_total += 1
            if (k[0] in ctx) and (k[1] in ctx):
                uniq_so_grounded_total += 1
        # Repetition distribution
        for k, c in spo.items():
            rep_freq_distrib[c] += 1
        # Position of duplicates: for each triplet at position i, is it a repeat (i.e., same key seen earlier)?
        seen = set()
        for i, t in enumerate(triplets):
            spo_k = norm_triplet(t)
            rel = min(int(10 * i / n), 9)
            bin_total[rel] += 1
            if spo_k in seen:
                bin_dup[rel] += 1
            seen.add(spo_k)
        # Unique-only positional hallu rate: for each unique triplet take its first_pos,
        # determine grounded, bin
        for k, p in first_pos.items():
            rel = min(int(10 * p / n), 9)
            uniq_bin_t[rel] += 1
            if not ((k[0] in ctx) and (k[2] in ctx)):
                uniq_bin_h[rel] += 1
        # Entity collapse — top subject share
        subj_counter = Counter(k[0] for k in spo.keys())
        if subj_counter:
            top_subj, top_subj_count = subj_counter.most_common(1)[0]
            top_subject_share.append(top_subj_count / len(spo))
            # is the top subject in the context?
            top1_subject_grounded.append(top_subj in ctx)
        # Tail-loop
        tail = triplets[-max(20, n//5):]
        ctr = Counter(tuple(map(str, t)) for t in tail)
        if ctr.most_common(1)[0][1] >= 5:
            tail_loop += 1

    n_ex = out["n_aligned"]
    out["raw_total_triplets"] = raw_total
    out["raw_grounded_total"] = raw_grounded_total
    out["raw_grounded_rate"] = raw_grounded_total / raw_total
    out["mean_raw_per_example"] = raw_total / n_ex

    out["unique_spo_total"] = uniq_spo_total
    out["unique_spo_grounded"] = uniq_spo_grounded_total
    out["unique_spo_grounded_rate"] = uniq_spo_grounded_total / uniq_spo_total
    out["mean_unique_spo_per_example"] = uniq_spo_total / n_ex
    out["dedup_ratio_spo"] = uniq_spo_total / raw_total
    out["expansion_factor_spo"] = raw_total / uniq_spo_total   # how many copies on average

    out["unique_so_total"] = uniq_so_total
    out["unique_so_grounded"] = uniq_so_grounded_total
    out["unique_so_grounded_rate"] = uniq_so_grounded_total / uniq_so_total
    out["mean_unique_so_per_example"] = uniq_so_total / n_ex

    out["rep_freq_distrib"] = dict(rep_freq_distrib.most_common(15))
    out["dup_rate_by_relpos"] = [bin_dup[i]/bin_total[i] if bin_total[i] else None for i in range(10)]
    out["uniq_hallu_rate_by_relpos"] = [uniq_bin_h[i]/uniq_bin_t[i] if uniq_bin_t[i] else None for i in range(10)]
    out["mean_top_subject_share"] = sum(top_subject_share)/len(top_subject_share) if top_subject_share else None
    out["frac_top_subject_grounded"] = sum(top1_subject_grounded)/len(top1_subject_grounded) if top1_subject_grounded else None
    out["tail_loop_count"] = tail_loop
    out["tail_loop_frac"] = tail_loop / n_ex
    return out

results = {}
for name, (mdir, sv) in RUNS.items():
    print(f"Analyzing {name} ...", flush=True)
    results[name] = analyze(mdir, sv)

# ============================
# Print comparison
# ============================
def fmt(v, w=14):
    if isinstance(v, float):
        return f"{v:>{w}.4f}"
    return f"{v:>{w}}"

print("\n" + "=" * 100)
print("KB DEDUP + HALLUCINATION ANALYSIS  (aligned hotpotqa-dev, ckpt500, all 3 runs)")
print("=" * 100)

sections = [
    ("Raw (no dedup)", [
        "n_aligned",
        "mean_raw_per_example",
        "raw_total_triplets",
        "raw_grounded_rate",
    ]),
    ("Deduplicated to unique (subj, relation, object)", [
        "mean_unique_spo_per_example",
        "unique_spo_total",
        "unique_spo_grounded_rate",
        "dedup_ratio_spo",
        "expansion_factor_spo",
    ]),
    ("Deduplicated to unique (subj, object) — collapses relation paraphrases", [
        "mean_unique_so_per_example",
        "unique_so_total",
        "unique_so_grounded_rate",
    ]),
    ("Entity collapse / repetition", [
        "mean_top_subject_share",
        "frac_top_subject_grounded",
        "tail_loop_count",
        "tail_loop_frac",
    ]),
]

for title, keys in sections:
    print(f"\n--- {title} ---")
    hdr = "metric".ljust(35) + "".join(f"{n:>14}" for n in results)
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        cells = "".join(fmt(results[r].get(k, "—")) for r in results)
        print(k.ljust(35) + cells)

print("\n--- Repetition frequency distribution (# unique triplets that appear N times) ---")
print(f"  {'run':<11}  " + "  ".join(f"{c:>3}x" for c in [1,2,3,4,5,6,7,8,9,10]))
for r in results:
    rf = results[r]["rep_freq_distrib"]
    print(f"  {r:<11}  " + "  ".join(f"{rf.get(c,0):>4}" for c in [1,2,3,4,5,6,7,8,9,10]))

print("\n--- DUPLICATION rate by relative position (10 bins, start→end) ---")
print(f"  {'run':<12}  " + "  ".join(f"bin{i}" for i in range(10)))
for r in results:
    vs = results[r]["dup_rate_by_relpos"]
    print(f"  {r:<12}  " + "  ".join(f"{v:.3f}" if v else " --- " for v in vs))

print("\n--- HALLU rate by relative position OF UNIQUE TRIPLETS (using first-occurrence) ---")
print(f"  {'run':<12}  " + "  ".join(f"bin{i}" for i in range(10)))
for r in results:
    vs = results[r]["uniq_hallu_rate_by_relpos"]
    print(f"  {r:<12}  " + "  ".join(f"{v:.3f}" if v else " --- " for v in vs))

out_path = "/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1/hallucination_analysis_dedup.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved raw stats to {out_path}")
