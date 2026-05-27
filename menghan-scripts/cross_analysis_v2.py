"""Cross-analysis pipeline producing Tables A / B / C from the v2 pipeline outputs.

Inputs (per model):
    * JSONL retrieval log (from run_inference_csv.py --log-jsonl)
        - per chain: phase2.lookups[*] with `retrieved` (source_row + source_tidx + cosine_sim)
          and `selected_value` / `selected_retrieved_idx`
    * Phase 1 LLM-judge JSON (faithful / hallucinated per (row, triplet_idx))
    * Phase 2 LLM-judge JSON (grounding / reasoning per chain)
    * Inference CSV (for EM per chain — column em_accuracy_0_0)

Outputs (in KG_results_v2/cross_analysis/):
    * table_a_used_vs_unused_faithfulness.csv
    * table_b_returned_vs_selected_faithfulness.csv
    * table_c_faithfulness_conditioned_grounding.csv
    * cross_analysis_per_chain.jsonl   (intermediate per-chain detail, for debugging)

Usage:
    /share/j_sun/mx253/envs/lmlm_mx/bin/python menghan-scripts/cross_analysis_v2.py \\
        --models qwen3_1.7b_sft_n10_sanity \\
        --tag sanity

For production:
    --models qwen3_1.7b_sft_n1000_prod qwen3_1.7b_grpo_ckpt500_n1000_prod \\
              qwen3_4b_sft_n1000_prod qwen3_4b_grpo_ckpt500_n1000_prod \\
        --tag prod
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2")

GROUNDING_BUCKETS = ["fully_grounded", "partially_grounded", "ungrounded", "no_answer"]
SOUNDNESS_ISSUES = ["ambiguous_entity_value", "trivial", "non_specific", "malformed", "reversed_roles"]


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def load_phase1_labels(path: Path) -> dict[tuple[int, int], dict]:
    """Return {(row_idx, triplet_idx): {'faithfulness': str, 'soundness_issues': list}}."""
    with open(path) as f:
        p1 = json.load(f)
    labels = {}
    for r in p1:
        ridx = r["row_index"]
        evs = r["db_results"][0]["evaluation"]["triplet_evaluations"]
        for ev in evs:
            labels[(ridx, ev["triplet_index"])] = {
                "faithfulness": ev["faithfulness"],
                "soundness_issues": ev.get("soundness_issues", []),
            }
    return labels


def load_phase2_labels(path: Path) -> dict[int, dict]:
    """Return {row_idx: {'grounding': str, 'reasoning': str}}."""
    with open(path) as f:
        p2 = json.load(f)
    out = {}
    for r in p2:
        c = r["db_results"][0]["completions"][0]
        out[r["row_index"]] = {"grounding": c["grounding"], "reasoning": c["reasoning"]}
    return out


def load_em(csv_path: Path) -> dict[int, int]:
    """Return {row_idx: em_accuracy_0_0}.

    CSV's `step` field is per-batch not per-row, so row_idx is just the CSV position.
    """
    out = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                out[i] = int(row.get("em_accuracy_0_0") or 0)
            except (TypeError, ValueError):
                out[i] = 0
    return out


# ── Table 0: Phase 1 KB summary (paper Table 8 style) ───────────────────────
def compute_table_0_kb_summary(phase1_path: Path) -> list[dict]:
    """Per-model Phase 1 KB quality summary: count + % for valid/hallucinated/
    soundness, broken down by each soundness sub-issue.

    Output rows (1 row per metric):
        - Total triplets
        - Valid (faithful + no soundness issues)
        - Hallucinated
        - Soundness Issues (any)
        - (i)–(v) per-issue breakdown
    """
    with open(phase1_path) as f:
        p1 = json.load(f)

    total = n_faithful = n_hallu = n_with_issues = 0
    issue_counts = Counter()

    for r in p1:
        for e in r["db_results"][0]["evaluation"]["triplet_evaluations"]:
            total += 1
            if e["faithfulness"] == "faithful":
                n_faithful += 1
                issues = e.get("soundness_issues") or []
                if issues:
                    n_with_issues += 1
                    for issue in issues:
                        issue_counts[issue] += 1
            elif e["faithfulness"] == "hallucinated":
                n_hallu += 1

    def pct(c):
        return round(c / total * 100, 2) if total else None

    rows = [
        {"metric": "total_triplets",          "count": total, "pct": 100.0},
        {"metric": "valid_clean",              "count": n_faithful - n_with_issues, "pct": pct(n_faithful - n_with_issues)},
        {"metric": "hallucinated",             "count": n_hallu, "pct": pct(n_hallu)},
        {"metric": "faithful_any",             "count": n_faithful, "pct": pct(n_faithful)},
        {"metric": "soundness_issues_any",     "count": n_with_issues, "pct": pct(n_with_issues)},
    ]
    for issue in SOUNDNESS_ISSUES:
        c = issue_counts.get(issue, 0)
        rows.append({"metric": f"soundness_{issue}", "count": c, "pct": pct(c)})
    return rows


# ── Table A ──────────────────────────────────────────────────────────────────
def compute_table_a(chains: list[dict], p1_labels: dict) -> list[dict]:
    """Used vs Unused Faithfulness.
    Universe = labeled triplets (across all labeled rows).
    Used = labeled triplets retrieved by ANY chain (any lookup) at any point.
    Unused = labeled triplets not retrieved by anyone.
    Reports: triplet count, faithful%, hallucinated%, coverage% (used/universe).
    """
    # Aggregate: which (row, tidx) was retrieved by ANY chain?
    used_keys = set()
    for ch in chains:
        for lk in ch["phase2"]["lookups"]:
            for r in lk.get("retrieved", []):
                src_row = r.get("source_row")
                src_tidx = r.get("source_tidx")
                if src_row is None or src_tidx is None:
                    continue
                used_keys.add((src_row, src_tidx))

    universe_keys = set(p1_labels.keys())
    used_set = universe_keys & used_keys
    unused_set = universe_keys - used_keys
    n_universe = len(universe_keys)

    def bucket_stats(keys):
        n_f = sum(1 for k in keys if p1_labels[k]["faithfulness"] == "faithful")
        n_h = sum(1 for k in keys if p1_labels[k]["faithfulness"] == "hallucinated")
        return n_f, n_h

    rows = []
    for label, keys in [("used", used_set), ("unused", unused_set)]:
        n_total = len(keys)
        n_f, n_h = bucket_stats(keys)
        n_labeled = n_f + n_h
        rows.append({
            "usage": label,
            "n_total": n_total,
            "n_faithful": n_f,
            "n_hallucinated": n_h,
            "faithful_pct": round(n_f / n_labeled * 100, 2) if n_labeled else None,
            "hallucinated_pct": round(n_h / n_labeled * 100, 2) if n_labeled else None,
            "n_universe": n_universe,
            "coverage_pct": round(n_total / n_universe * 100, 2) if n_universe else None,
        })
    return rows


# ── Table B ──────────────────────────────────────────────────────────────────
def compute_table_b(chains: list[dict], p1_labels: dict) -> list[dict]:
    """Per-lookup-level faithfulness:
        - returned_faithful_pct: among all (lookup, returned) pairs that have a label,
          % faithful  (treats each top-k retrieved triplet as a separate observation)
        - selected_faithful_pct: among lookups where the selected_value resolves to a labeled
          triplet, % faithful
    """
    # Per-chain ratios (chain-weighted mean), per user spec:
    #   - selected_ratio_per_chain = (#faithful selected) / (#labeled selected)
    #   - returned_ratio_per_chain = (#faithful returned) / (#labeled returned)
    # Chains with 0 labeled in their bucket are excluded from the mean.
    selected_chain_ratios = []
    selected_chains_excluded = 0  # had lookups but 0 labeled
    selected_chains_no_lookups = 0

    returned_chain_ratios = []
    returned_chains_excluded = 0
    returned_chains_no_lookups = 0

    # Also keep pooled (triplet/lookup-weighted) counts for completeness.
    pool_sel_faithful = pool_sel_hallu = pool_sel_labeled = pool_sel_total = pool_sel_none = pool_sel_unlabeled = 0
    pool_ret_faithful = pool_ret_hallu = pool_ret_labeled = pool_ret_total = 0

    for ch in chains:
        lookups = ch["phase2"]["lookups"]
        if not lookups:
            selected_chains_no_lookups += 1
            returned_chains_no_lookups += 1
            continue

        chain_sel_f = chain_sel_h = chain_sel_labeled = 0
        chain_ret_f = chain_ret_h = chain_ret_labeled = 0

        for lk in lookups:
            retrieved = lk.get("retrieved", [])
            for r in retrieved:
                pool_ret_total += 1
                key = (r.get("source_row"), r.get("source_tidx"))
                if key in p1_labels:
                    chain_ret_labeled += 1
                    pool_ret_labeled += 1
                    if p1_labels[key]["faithfulness"] == "faithful":
                        chain_ret_f += 1
                        pool_ret_faithful += 1
                    elif p1_labels[key]["faithfulness"] == "hallucinated":
                        chain_ret_h += 1
                        pool_ret_hallu += 1

            pool_sel_total += 1
            sel_idx = lk.get("selected_retrieved_idx")
            if sel_idx is None or not (0 <= sel_idx < len(retrieved)):
                pool_sel_none += 1
                continue
            sel_triplet = retrieved[sel_idx]
            sel_key = (sel_triplet.get("source_row"), sel_triplet.get("source_tidx"))
            if sel_key in p1_labels:
                chain_sel_labeled += 1
                pool_sel_labeled += 1
                if p1_labels[sel_key]["faithfulness"] == "faithful":
                    chain_sel_f += 1
                    pool_sel_faithful += 1
                elif p1_labels[sel_key]["faithfulness"] == "hallucinated":
                    chain_sel_h += 1
                    pool_sel_hallu += 1
            else:
                pool_sel_unlabeled += 1

        # Per-chain ratio
        if chain_sel_labeled > 0:
            selected_chain_ratios.append(chain_sel_f / chain_sel_labeled)
        else:
            selected_chains_excluded += 1

        if chain_ret_labeled > 0:
            returned_chain_ratios.append(chain_ret_f / chain_ret_labeled)
        else:
            returned_chains_excluded += 1

    def mean_pct(lst):
        return round(sum(lst) / len(lst) * 100, 2) if lst else None

    rows = [
        {
            "metric": "mean_selected_faithful_lookup_ratio",
            "n_chains_included": len(selected_chain_ratios),
            "n_chains_excluded_no_labeled": selected_chains_excluded,
            "n_chains_no_lookups": selected_chains_no_lookups,
            "mean_ratio_pct": mean_pct(selected_chain_ratios),
            "pool_n_labeled": pool_sel_labeled,
            "pool_n_faithful": pool_sel_faithful,
            "pool_n_hallucinated": pool_sel_hallu,
            "pool_faithful_pct": round(pool_sel_faithful / pool_sel_labeled * 100, 2) if pool_sel_labeled else None,
            "pool_n_lookups_total": pool_sel_total,
            "pool_n_selected_none": pool_sel_none,
            "pool_n_selected_unlabeled": pool_sel_unlabeled,
        },
        {
            "metric": "mean_returned_faithful_ratio",
            "n_chains_included": len(returned_chain_ratios),
            "n_chains_excluded_no_labeled": returned_chains_excluded,
            "n_chains_no_lookups": returned_chains_no_lookups,
            "mean_ratio_pct": mean_pct(returned_chain_ratios),
            "pool_n_labeled": pool_ret_labeled,
            "pool_n_faithful": pool_ret_faithful,
            "pool_n_hallucinated": pool_ret_hallu,
            "pool_faithful_pct": round(pool_ret_faithful / pool_ret_labeled * 100, 2) if pool_ret_labeled else None,
            "pool_n_retrieved_total": pool_ret_total,
        },
    ]
    return rows


# ── Table C ──────────────────────────────────────────────────────────────────
def _bucket_stats(chs: list[tuple]) -> dict:
    n = len(chs)
    out = {"n_chains": n}
    for b in GROUNDING_BUCKETS:
        cnt = sum(1 for (_, g, _) in chs if g == b)
        out[f"{b}_n"] = cnt
        out[f"{b}_pct"] = round(cnt / n * 100, 2) if n else None
    n_em_correct = sum(em for (_, _, em) in chs)
    out["n_em_correct"] = n_em_correct
    out["mean_em"] = round(n_em_correct / n, 3) if n else None
    return out


def _classify_by_selected(chain: dict, p1_labels: dict) -> str:
    """A: all labeled selected faithful; B: any hallu; C: 0 labeled selected; D: 0 lookups."""
    lookups = chain["phase2"]["lookups"]
    if not lookups:
        return "D_no_lookup"
    labels = []
    for lk in lookups:
        sel_idx = lk.get("selected_retrieved_idx")
        retrieved = lk.get("retrieved", [])
        if sel_idx is None or not (0 <= sel_idx < len(retrieved)):
            labels.append(None)
            continue
        sel_key = (retrieved[sel_idx].get("source_row"), retrieved[sel_idx].get("source_tidx"))
        labels.append(p1_labels.get(sel_key, {}).get("faithfulness"))
    labeled = [l for l in labels if l is not None]
    if not labeled:
        return "C_unknown_only"
    if any(l == "hallucinated" for l in labeled):
        return "B_has_hallucinated"
    return "A_all_faithful"


def _classify_by_returned(chain: dict, p1_labels: dict) -> str:
    """A': all labeled returned faithful; B': any hallu returned;
       C': 0 labeled returned (all unlabeled); D': 0 lookups."""
    lookups = chain["phase2"]["lookups"]
    if not lookups:
        return "D_no_lookup"
    labels = []
    for lk in lookups:
        for r in lk.get("retrieved", []):
            key = (r.get("source_row"), r.get("source_tidx"))
            labels.append(p1_labels.get(key, {}).get("faithfulness"))
    labeled = [l for l in labels if l is not None]
    if not labeled:
        return "C_unknown_only"
    if any(l == "hallucinated" for l in labeled):
        return "B_has_hallucinated"
    return "A_all_faithful"


def compute_table_c(chains: list[dict], p1_labels: dict, p2_labels: dict, em_by_row: dict) -> list[dict]:
    """Two grouping methods (per user spec):
        Method 1 (by selected): A/B/C/D
        Method 2 (by all returned): A'/B' (+ C'/D' for completeness)
    Each group: grounding distribution + mean EM (count + %).
    Output rows tagged with `grouping_method` column.
    """
    methods = [
        ("by_selected", _classify_by_selected),
        ("by_returned", _classify_by_returned),
    ]
    group_order = ["A_all_faithful", "B_has_hallucinated", "C_unknown_only", "D_no_lookup"]

    rows = []
    for method_name, classify_fn in methods:
        group_chains = defaultdict(list)
        for ch in chains:
            row_idx = ch["row_idx"]
            if row_idx not in p2_labels:
                continue
            grounding = p2_labels[row_idx]["grounding"]
            em = em_by_row.get(row_idx, 0)
            group = classify_fn(ch, p1_labels)
            group_chains[group].append((row_idx, grounding, em))

        for group in group_order:
            chs = group_chains[group]
            rows.append({"grouping_method": method_name, "group": group, **_bucket_stats(chs)})
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────
def process_model(model_name: str) -> dict:
    """Returns dict with table_a, table_b, table_c, per_chain (for one model)."""
    inf_csv = ROOT / "inference" / f"{model_name}.csv"
    inf_jsonl = ROOT / "inference" / f"{model_name}.jsonl"
    p1_json = ROOT / "phase1_judge" / f"phase1_{model_name}.json"
    p2_json = ROOT / "phase2_judge" / f"phase2_{model_name}.json"
    for p in [inf_csv, inf_jsonl, p1_json, p2_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input for {model_name}: {p}")

    chains = load_jsonl(inf_jsonl)
    p1 = load_phase1_labels(p1_json)
    p2 = load_phase2_labels(p2_json)
    em = load_em(inf_csv)

    table_0 = compute_table_0_kb_summary(p1_json)
    table_a = compute_table_a(chains, p1)
    table_b = compute_table_b(chains, p1)
    table_c = compute_table_c(chains, p1, p2, em)

    return {
        "model": model_name,
        "n_chains": len(chains),
        "n_labeled_triplets": len(p1),
        "n_phase2_labels": len(p2),
        "table_0": table_0,
        "table_a": table_a,
        "table_b": table_b,
        "table_c": table_c,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model file stems (e.g. qwen3_1.7b_sft_n10_sanity)")
    ap.add_argument("--tag", default="results", help="Output filename suffix")
    args = ap.parse_args()

    out_dir = ROOT / "cross_analysis"
    out_dir.mkdir(exist_ok=True)

    all_table_0, all_table_a, all_table_b, all_table_c = [], [], [], []
    for m in args.models:
        print(f"\n== Processing {m} ==")
        res = process_model(m)
        print(f"   chains={res['n_chains']}, labeled_triplets={res['n_labeled_triplets']}, "
              f"phase2_chains={res['n_phase2_labels']}")
        for row in res["table_0"]:
            all_table_0.append({"model": m, **row})
        for row in res["table_a"]:
            all_table_a.append({"model": m, **row})
        for row in res["table_b"]:
            all_table_b.append({"model": m, **row})
        for row in res["table_c"]:
            all_table_c.append({"model": m, **row})

    def write_csv(path, rows):
        if not rows:
            print(f"   [warn] no rows for {path.name}")
            return
        # union of all keys to handle table_b's heterogeneous rows
        all_keys = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k); all_keys.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"   saved {path}")

    print("\n== Writing tables ==")
    write_csv(out_dir / f"table_0_kb_summary_{args.tag}.csv", all_table_0)
    write_csv(out_dir / f"table_a_used_vs_unused_{args.tag}.csv", all_table_a)
    write_csv(out_dir / f"table_b_returned_vs_selected_{args.tag}.csv", all_table_b)
    write_csv(out_dir / f"table_c_faithful_conditioned_grounding_{args.tag}.csv", all_table_c)

    # Print summaries to console
    print("\n== Table 0: KB summary (paper Table 8 style) ==")
    for r in all_table_0:
        pct_str = f"{r['pct']:.2f}%" if r['pct'] is not None else "n/a"
        print(f"  {r['model']:35s} {r['metric']:30s} count={r['count']:>7,}  pct={pct_str:>8s}")

    print("\n== Table A: Used vs Unused Faithfulness (with reward_signal_coverage) ==")
    for r in all_table_a:
        print(f"  {r['model']:35s} {r['usage']:7s} n_total={r['n_total']:>6,} "
              f"faithful={r['n_faithful']:>5,} ({r['faithful_pct']!s:>6}%)  "
              f"hallu={r['n_hallucinated']:>4,} ({r['hallucinated_pct']!s:>6}%)  "
              f"coverage={r['coverage_pct']!s:>5}%")

    print("\n== Table B: Per-Chain Faithful Lookup Ratio (chain-weighted means) ==")
    for r in all_table_b:
        print(f"  {r['model']:35s} {r['metric']:40s} "
              f"chain_mean={r['mean_ratio_pct']!s:>6}%  "
              f"(n_chains_used={r['n_chains_included']}, excluded_nolabel={r['n_chains_excluded_no_labeled']}, "
              f"nolookup={r['n_chains_no_lookups']})  "
              f"[pool: faithful={r['pool_n_faithful']}/{r['pool_n_labeled']}={r['pool_faithful_pct']!s}%]")

    print("\n== Table C: Faithfulness-Conditioned Grounding ==")
    for r in all_table_c:
        em_str = f"{r['n_em_correct']}/{r['n_chains']}={r['mean_em']}" if r['n_chains'] else "n/a"
        bucket_str = " ".join(
            f"{b[:2].upper()}={r[f'{b}_n']!s}({r[f'{b}_pct']!s}%)" for b in GROUNDING_BUCKETS
        )
        print(f"  {r['model']:35s} [{r['grouping_method']}] {r['group']:23s} n={r['n_chains']:>3}  EM={em_str:>12s}  {bucket_str}")


if __name__ == "__main__":
    main()
