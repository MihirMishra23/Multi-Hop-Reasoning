"""
Cross-analysis: For each reasoning chain (Phase 2), identify the KB triplets actually
used via <|db_entity|>X<|db_relationship|>Y<|db_return|>Z<|db_end|> lookups,
match them back to the *unified DB* (concat of all 1000 rows' generated_db_0),
look up their faithful/hallucinated labels (from Phase 1, which labeled rows 0-99),
and aggregate per-chain % faithful within each grounding bucket.

The inference used --concat-all-db so a lookup in chain k can retrieve a triplet
from ANY row's local KB. Therefore we must match against the global KB.

Usage:
    /share/j_sun/mx253/envs/lmlm_mx/bin/python menghan-scripts/grounding_x_faithfulness.py
"""

import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/share/j_sun/mx253/Multi-Hop-Reasoning")
PHASE1_DIR = ROOT / "reward_hacking_evaluate/phase1"
PHASE2_DIR = ROOT / "reward_hacking_evaluate/phase2"
CSV_DIR = ROOT / "KG_results"

MODELS = {
    "4B GRPO":   ("phase1_results_ckpt500_hotpotqa_dev_n1000_all_concat_trainparams.json",
                  "phase2_results_ckpt500_hotpotqa_dev_n1000_all_concat_trainparams.json",
                  "ckpt500_hotpotqa_dev_n1000_all_concat_trainparams.csv"),
    "4B SFT":    ("phase1_results_sft_hotpotqa_dev_n1000_all_concat_trainparams.json",
                  "phase2_results_sft_hotpotqa_dev_n1000_all_concat_trainparams.json",
                  "sft_hotpotqa_dev_n1000_all_concat_trainparams.csv"),
    "1.7B GRPO": ("phase1_results_grpo_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.json",
                  "phase2_results_grpo_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.json",
                  "grpo_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.csv"),
    "1.7B SFT":  ("phase1_results_new_sft_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.json",
                  "phase2_results_new_sft_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.json",
                  "new_sft_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.csv"),
}

LOOKUP_RE = re.compile(
    r"<\|db_entity\|>(.*?)<\|db_relationship\|>(.*?)<\|db_return\|>(.*?)<\|db_end\|>",
    re.DOTALL,
)
BUCKETS = ["fully_grounded", "partially_grounded", "ungrounded", "no_answer"]
TOP_K = 4  # retrieval top_k used at inference


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def build_global_kb_and_label_map(csv_path: Path, phase1_path: Path):
    """Return:
        value_index: {value_norm: [(row_idx, tidx, e_norm, r_norm)]}  -- across ALL 1000 rows
        label_map: {(row_idx, tidx): 'faithful'/'hallucinated'}        -- labeled rows only
    """
    df = pd.read_csv(csv_path)
    value_index = defaultdict(list)
    for row_idx in range(len(df)):
        try:
            triplets = ast.literal_eval(df.iloc[row_idx]["generated_db_0"])
        except Exception:
            continue
        for tidx, t in enumerate(triplets):
            if not (isinstance(t, (list, tuple)) and len(t) == 3):
                continue
            e, r, v = (str(x) for x in t)
            vn = norm(v)
            if vn:
                value_index[vn].append((row_idx, tidx, norm(e), norm(r)))

    with open(phase1_path) as f:
        phase1 = json.load(f)
    label_map = {}
    for p1r in phase1:
        ridx = p1r["row_index"]
        for ev in p1r["db_results"][0]["evaluation"]["triplet_evaluations"]:
            label_map[(ridx, ev["triplet_index"])] = ev["faithfulness"]

    return value_index, label_map, set(p["row_index"] for p in phase1), df


def match_lookup_to_triplets(X, Y, Z, value_index):
    """Given a single <|db_entity|>X|db_relationship|>Y|db_return|>Z|db_end|> lookup,
    find up to TOP_K matching (row, tidx) tuples from the unified KB.

    Strategy:
      1. Candidate triplets = all triplets in the global value_index whose value v
         either equals Zn or is a substring of Zn (handles multi-value returns).
      2. Score by entity/relation similarity to (X, Y).
      3. Return top-TOP_K by score.
    """
    Xn, Yn, Zn = norm(X), norm(Y), norm(Z)
    if not Zn:
        return []

    candidates = []  # (row_idx, tidx, score, v_norm)
    for v_norm, entries in value_index.items():
        if v_norm == Zn or v_norm in Zn:
            for (row_idx, tidx, en, rn) in entries:
                score = 0
                if en == Xn:
                    score += 4
                elif Xn and (Xn in en or en in Xn):
                    score += 2
                if rn == Yn:
                    score += 3
                elif Yn and (Yn in rn or rn in Yn):
                    score += 1
                # Slight bonus for exact value match (single-value return)
                if v_norm == Zn:
                    score += 1
                candidates.append((row_idx, tidx, score))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[2], reverse=True)
    return [(r, t) for (r, t, _s) in candidates[:TOP_K]]


def analyze_model(label, csv_file, p1_file, p2_file):
    csv_path = CSV_DIR / csv_file
    p1_path = PHASE1_DIR / p1_file
    p2_path = PHASE2_DIR / p2_file

    print(f"[{label}] building global KB index from {csv_file} ...")
    value_index, label_map, labeled_rows, _df = build_global_kb_and_label_map(csv_path, p1_path)
    print(f"   global KB: {sum(len(v) for v in value_index.values())} triplets "
          f"across {len(set(r for entries in value_index.values() for (r,_,_,_) in entries))} rows")
    print(f"   labeled rows: {len(labeled_rows)}, labeled triplets: {len(label_map)}")

    with open(p2_path) as f:
        phase2 = json.load(f)

    # Per-chain bookkeeping
    bucket_pct_faithful = defaultdict(list)
    bucket_pct_hallu = defaultdict(list)
    bucket_n_used = defaultdict(list)
    bucket_n_labeled = defaultdict(list)
    bucket_total_chains = defaultdict(int)

    for p2r in phase2:
        comp = p2r["db_results"][0]["completions"][0]
        grounding = comp["grounding"]
        bucket_total_chains[grounding] += 1
        comp_text = comp["completion_text"]

        # Collect all matched triplets (with multiplicity per lookup)
        matched = []
        for X, Y, Z in LOOKUP_RE.findall(comp_text):
            matched.extend(match_lookup_to_triplets(X, Y, Z, value_index))

        bucket_n_used[grounding].append(len(matched))
        if not matched:
            continue

        # Check labels
        n_faithful = 0
        n_hallu = 0
        n_labeled = 0
        for (rrow, ttidx) in matched:
            lab = label_map.get((rrow, ttidx))
            if lab is None:
                continue
            n_labeled += 1
            if lab == "faithful":
                n_faithful += 1
            elif lab == "hallucinated":
                n_hallu += 1

        bucket_n_labeled[grounding].append(n_labeled)
        if n_labeled == 0:
            continue  # no labeled lookups for this chain, can't compute %

        bucket_pct_faithful[grounding].append(n_faithful / n_labeled)
        bucket_pct_hallu[grounding].append(n_hallu / n_labeled)

    rows = []
    for b in BUCKETS:
        n_total = bucket_total_chains[b]
        n_with_pct = len(bucket_pct_faithful[b])
        avg_f = (sum(bucket_pct_faithful[b]) / n_with_pct * 100) if n_with_pct else float("nan")
        avg_h = (sum(bucket_pct_hallu[b]) / n_with_pct * 100) if n_with_pct else float("nan")
        avg_used = (sum(bucket_n_used[b]) / len(bucket_n_used[b])) if bucket_n_used[b] else 0
        avg_labeled = (sum(bucket_n_labeled[b]) / len(bucket_n_labeled[b])) if bucket_n_labeled[b] else 0
        rows.append({
            "model": label,
            "grounding": b,
            "n_chains": n_total,
            "n_chains_w_labeled_lookup": n_with_pct,
            "avg_total_matched_per_chain": round(avg_used, 1),
            "avg_labeled_matched_per_chain": round(avg_labeled, 1),
            "avg_pct_faithful": round(avg_f, 1) if avg_f == avg_f else None,
            "avg_pct_hallucinated": round(avg_h, 1) if avg_h == avg_h else None,
        })
    return rows


def main():
    all_rows = []
    for label, (p1f, p2f, csvf) in MODELS.items():
        if not ((PHASE1_DIR / p1f).exists() and (PHASE2_DIR / p2f).exists() and (CSV_DIR / csvf).exists()):
            print(f"[skip] {label}: missing files")
            continue
        all_rows.extend(analyze_model(label, csvf, p1f, p2f))

    df = pd.DataFrame(all_rows)
    out = ROOT / "reward_hacking_evaluate/grounding_x_faithfulness.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
