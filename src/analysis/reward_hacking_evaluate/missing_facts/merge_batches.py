"""
Recomputes macro (mean per-passage) and micro (pooled counts) P/R/F1.

Usage:
    python merge_batches.py
"""
import csv
from pathlib import Path

csv.field_size_limit(10 ** 7)
HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# key -> (display, list of per-row CSV basenames to union)
MODELS = [
    ("1.7b_sft",  "SFT (1.7B)",        ["missing_facts_1.7b_sft.csv",  "missing_facts_1.7b_sft_b2.csv"]),
    ("1.7b_grpo", "GRPO (1.7B)",       ["missing_facts_1.7b_grpo.csv", "missing_facts_1.7b_grpo_b2.csv"]),
    ("4b_sft",    "SFT (4B)",          ["missing_facts_4b_sft.csv",    "missing_facts_4b_sft_b2.csv"]),
    ("4b_grpo",   "GRPO (4B)",         ["missing_facts_4b_grpo.csv",   "missing_facts_4b_grpo_b2.csv"]),
    ("atlas",     "AutoSchemaKG (8B)", ["missing_facts_atlas.csv"]),
]


def load(basenames):
    """Union rows by row_index across the given CSVs (first occurrence wins)."""
    by_idx = {}
    for b in basenames:
        p = RES / b
        if not p.exists():
            print(f"  [WARN] missing file: {p}")
            continue
        for r in csv.DictReader(open(p)):
            idx = int(r["row_index"])
            by_idx.setdefault(idx, r)
    return list(by_idx.values())


def metrics(rows):
    n = len(rows)
    P = [float(r["precision"]) for r in rows]
    R = [float(r["recall"]) for r in rows]
    F = [float(r["f1"]) for r in rows]
    macro = (sum(P) / n, sum(R) / n, sum(F) / n)
    sN = sum(int(r["N"]) for r in rows)
    sFa = sum(int(r["faithful"]) for r in rows)
    sM = sum(int(r["missing"]) for r in rows)
    mp = sFa / sN if sN else 0.0
    mr = sFa / (sFa + sM) if (sFa + sM) else 0.0
    mf = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
    return n, macro, (mp, mr, mf)


def main():
    summary = []
    for key, display, files in MODELS:
        rows = load(files)
        if not rows:
            continue
        # write combined per-row CSV
        out = RES / f"missing_facts_{key}_combined.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_index", "N", "faithful", "missing", "precision", "recall", "f1", "missing_text"])
            for r in sorted(rows, key=lambda x: int(x["row_index"])):
                w.writerow([r["row_index"], r["N"], r["faithful"], r["missing"],
                            r["precision"], r["recall"], r["f1"], r["missing_text"]])
        n, (Pm, Rm, Fm), (Pu, Ru, Fu) = metrics(rows)
        summary.append((display, n, Pm, Rm, Fm, Pu, Ru, Fu))
        print(f"{display:<20} n={n:<4} macro P={Pm:.3f} R={Rm:.3f} F1={Fm:.3f}  | micro P={Pu:.3f} R={Ru:.3f} F1={Fu:.3f}")

    sp = RES / "missing_facts_summary_combined.csv"
    with open(sp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_passages", "precision", "recall", "f1",
                    "micro_precision", "micro_recall", "micro_f1"])
        for d, n, Pm, Rm, Fm, Pu, Ru, Fu in summary:
            w.writerow([d, n, f"{Pm:.4f}", f"{Rm:.4f}", f"{Fm:.4f}",
                        f"{Pu:.4f}", f"{Ru:.4f}", f"{Fu:.4f}"])

    print(f"\n{'='*64}\nCOMBINED SUMMARY (macro per-passage)\n{'='*64}")
    print(f"{'Model':<20}{'n':>5}{'Precision':>11}{'Recall':>10}{'F1':>10}")
    for d, n, Pm, Rm, Fm, *_ in summary:
        print(f"{d:<20}{n:>5}{Pm:>11.3f}{Rm:>10.3f}{Fm:>10.3f}")
    print(f"\nSaved {sp}")


if __name__ == "__main__":
    main()
