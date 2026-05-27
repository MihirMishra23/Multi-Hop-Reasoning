"""Table D: Compare original vs filtered KB Phase 2 grounding/reasoning results.

For each model, reports both original and filtered metrics side by side:
    - KB triplet count
    - Hallucination rate
    - Grounding distribution (fully/partially/ungrounded/no_answer)
    - Reasoning distribution (correct/minor_error/major_error)
    - Mean EM

Usage:
    /share/j_sun/mx253/envs/lmlm_mx/bin/python menghan-scripts/table_d_compare.py \\
        --models qwen3_1.7b_sft_n10_sanity \\
        --tag sanity
"""

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2")

GROUNDING_BUCKETS = ["fully_grounded", "partially_grounded", "ungrounded", "no_answer"]
REASONING_BUCKETS = ["correct", "minor_error", "major_error"]


def load_phase2(path: Path) -> dict[int, dict]:
    with open(path) as f:
        p2 = json.load(f)
    out = {}
    for r in p2:
        c = r["db_results"][0]["completions"][0]
        out[r["row_index"]] = {"grounding": c["grounding"], "reasoning": c["reasoning"]}
    return out


def kb_stats_from_csv(csv_path: Path) -> tuple[int, int]:
    """Total triplets + total rows from CSV's generated_db_0 column."""
    total = 0
    n = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            n += 1
            try:
                total += len(ast.literal_eval(row["generated_db_0"]))
            except Exception:
                pass
    return n, total


def hallu_rate_from_phase1(p1_path: Path) -> tuple[int, int, float]:
    """Returns (total_triplets, hallu_count, hallu_rate_pct)."""
    with open(p1_path) as f:
        p1 = json.load(f)
    total = sum(r["db_results"][0]["num_triplets"] for r in p1)
    n_hallu = sum(
        1
        for r in p1
        for e in r["db_results"][0]["evaluation"]["triplet_evaluations"]
        if e["faithfulness"] == "hallucinated"
    )
    rate = n_hallu / total * 100 if total else 0.0
    return total, n_hallu, rate


def em_from_csv(csv_path: Path) -> dict[int, int]:
    out = {}
    with open(csv_path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            try:
                out[i] = int(row.get("em_accuracy_0_0") or 0)
            except Exception:
                out[i] = 0
    return out


def dist_with_em(p2_labels, em_by_row):
    """Returns dict with grounding/reasoning counts/% and mean EM."""
    n = len(p2_labels)
    g_counts = Counter(v["grounding"] for v in p2_labels.values())
    r_counts = Counter(v["reasoning"] for v in p2_labels.values())
    out = {"n_chains": n}
    for b in GROUNDING_BUCKETS:
        c = g_counts.get(b, 0)
        out[f"{b}_n"] = c
        out[f"{b}_pct"] = round(c / n * 100, 2) if n else None
    for b in REASONING_BUCKETS:
        c = r_counts.get(b, 0)
        out[f"reasoning_{b}_n"] = c
        out[f"reasoning_{b}_pct"] = round(c / n * 100, 2) if n else None
    em_vals = [em_by_row.get(r, 0) for r in p2_labels.keys()]
    out["n_em_correct"] = sum(em_vals)
    out["mean_em"] = round(sum(em_vals) / n, 3) if n else None
    return out


def process_model(model_name: str) -> list[dict]:
    csv_orig = ROOT / "inference" / f"{model_name}.csv"
    csv_filt = ROOT / "inference" / f"{model_name}_filtered.csv"
    p1_json = ROOT / "phase1_judge" / f"phase1_{model_name}.json"
    p2_orig = ROOT / "phase2_judge" / f"phase2_{model_name}.json"
    p2_filt = ROOT / "phase2_judge" / f"phase2_{model_name}_filtered.json"
    for p in [csv_orig, csv_filt, p1_json, p2_orig, p2_filt]:
        if not p.exists():
            raise FileNotFoundError(f"Missing for {model_name}: {p}")

    n_rows_orig, total_orig = kb_stats_from_csv(csv_orig)
    _, total_filt = kb_stats_from_csv(csv_filt)
    p1_total, p1_hallu, p1_rate = hallu_rate_from_phase1(p1_json)

    em_orig = em_from_csv(csv_orig)
    em_filt = em_from_csv(csv_filt)
    p2_lab_orig = load_phase2(p2_orig)
    p2_lab_filt = load_phase2(p2_filt)

    s_orig = dist_with_em(p2_lab_orig, em_orig)
    s_filt = dist_with_em(p2_lab_filt, em_filt)

    rows = []
    rows.append({
        "kb_version": "original",
        "kb_triplet_count": total_orig,
        "labeled_triplet_count": p1_total,
        "hallu_count": p1_hallu,
        "hallu_rate_pct": round(p1_rate, 2),
        **s_orig,
    })
    rows.append({
        "kb_version": "filtered",
        "kb_triplet_count": total_filt,
        "labeled_triplet_count": p1_total - p1_hallu,
        "hallu_count": 0,
        "hallu_rate_pct": 0.0,
        **s_filt,
    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tag", default="results")
    args = ap.parse_args()

    out_dir = ROOT / "cross_analysis"
    out_dir.mkdir(exist_ok=True)

    all_rows = []
    for m in args.models:
        print(f"\n== Processing {m} ==")
        for row in process_model(m):
            all_rows.append({"model": m, **row})

    # Write CSV
    if all_rows:
        all_keys = []
        seen = set()
        for r in all_rows:
            for k in r:
                if k not in seen:
                    seen.add(k); all_keys.append(k)
        out_csv = out_dir / f"table_d_posthoc_filtering_{args.tag}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nSaved: {out_csv}")

    # Console summary
    print("\n== Table D: Post-hoc Filtering ==")
    print(f"{'Model':<35s} {'KB ver':<10s} {'KB triplets':>13s} {'Hallu rate':>11s} "
          f"{'FG%':>6s} {'PG%':>6s} {'UG%':>6s} {'NA%':>6s} "
          f"{'Rcorrect%':>10s} {'Rminor%':>8s} {'Rmajor%':>8s} {'EM':>8s}")
    print("-" * 145)
    for r in all_rows:
        em_str = f"{r['n_em_correct']}/{r['n_chains']}={r['mean_em']!s}"
        print(f"{r['model']:<35s} {r['kb_version']:<10s} "
              f"{r['kb_triplet_count']:>13,} {r['hallu_rate_pct']!s:>10s}%  "
              f"{r['fully_grounded_pct']!s:>5}% {r['partially_grounded_pct']!s:>5}% "
              f"{r['ungrounded_pct']!s:>5}% {r['no_answer_pct']!s:>5}%  "
              f"{r['reasoning_correct_pct']!s:>8}% {r['reasoning_minor_error_pct']!s:>6}% "
              f"{r['reasoning_major_error_pct']!s:>6}%  {em_str:>10s}")


if __name__ == "__main__":
    main()
