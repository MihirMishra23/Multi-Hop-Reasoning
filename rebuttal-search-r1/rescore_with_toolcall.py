"""Re-score existing eval result files under the patched answer extractor.

The original eval pipeline only extracted answers from <answer>...</answer>
tags. Late-training checkpoints emit answers in a hallucinated tool-call
format <tool_call>{"name":"answer","arguments":{"answer":"..."}}</tool_call>
which the legacy extractor silently misses.

This script re-extracts answers from each per_question entry's
raw_output_head + raw_output_tail (the head+tail combo covers the answer
in practice because the model always emits the final answer near the end).

Non-destructive: writes a sibling .rescored.json file for each input.
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rewards"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.metrics import exact_match_score  # noqa: E402
from hotpotqa_f1 import compute_score as searchr1_compute_score  # noqa: E402


def extract_answer(text: str) -> str:
    """Same logic as the patched eval_search_r1.extract_answer."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    last = ""
    for tc in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            data = json.loads(tc.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("name") == "answer":
            ans = data.get("arguments", {}).get("answer", "")
            if isinstance(ans, str) and ans.strip():
                last = ans.strip()
    return last


def rescore_file(path: Path) -> dict:
    d = json.loads(path.read_text())
    qs = d.get("per_question", [])
    n = len(qs)

    f1_total = 0.0
    em_total = 0.0
    n_answered = 0
    n_changed = 0  # how many entries flipped from blank ans -> tool-call ans

    new_per_q = []
    for q in qs:
        blob = (q.get("raw_output_head") or "") + (q.get("raw_output_tail") or "")
        new_ans = extract_answer(blob)
        old_ans = q.get("answer_text", "")
        if not old_ans and new_ans:
            n_changed += 1

        # Recompute EM
        em = 0.0
        for g in q["gold"]:
            if exact_match_score(new_ans, g):
                em = 1.0
                break

        # Recompute F1 via the same SearchR1 compute_score, passing a synthetic
        # <answer>-wrapped solution so the legacy extractor inside compute_score
        # sees our new extraction.
        sr1 = searchr1_compute_score(
            data_source="searchR1_hotpotqa",
            solution_str=f"<answer>{new_ans}</answer>" if new_ans else "",
            ground_truth={"target": q["gold"]},
        )

        if new_ans:
            n_answered += 1
        f1_total += sr1
        em_total += em

        nq = dict(q)
        nq["answer_text_rescored"] = new_ans
        nq["score_em_rescored"] = em
        nq["score_f1_rescored"] = sr1
        new_per_q.append(nq)

    rescored_summary = {
        "n": n,
        "token_f1_mean": f1_total / max(1, n),
        "em_mean": em_total / max(1, n),
        "n_with_answer": n_answered,
        "n_recovered_via_toolcall": n_changed,
    }

    return {
        "original_summary": d.get("summary"),
        "rescored_summary": rescored_summary,
        "metadata": d.get("metadata"),
        "per_question": new_per_q,
    }


def main():
    project_dir = Path("/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1")
    results_dir = project_dir / "eval_results"
    targets = sorted(results_dir.glob("icl3hop_tr1024_step*_*.json"))
    # Exclude any prior rescored files
    targets = [t for t in targets if ".rescored." not in t.name]

    print(f"Found {len(targets)} files to re-score")
    print()
    print(f'{"file":<70}  {"orig EM":>8}  {"new EM":>8}  {"Δ":>6}  {"recovered":>9}')
    print("-" * 110)

    table_rows = []
    for t in targets:
        out = rescore_file(t)
        orig = out["original_summary"]
        new = out["rescored_summary"]
        em_d = new["em_mean"] - orig["em_mean"]
        print(
            f'{t.name[:68]:<70}  {orig["em_mean"]:>8.3f}  {new["em_mean"]:>8.3f}  '
            f'{em_d:>+6.3f}  {new["n_recovered_via_toolcall"]:>9d}'
        )

        # Save sibling file (non-destructive)
        out_path = t.with_suffix(".rescored.json")
        out_path.write_text(json.dumps(out, indent=2))

        # Parse step/benchmark for the final pivot table
        m = re.match(r"icl3hop_tr1024_step(\d+)_(.+?)_n\d+_", t.name)
        if m:
            step = int(m.group(1))
            bench = m.group(2)
            table_rows.append((step, bench, orig["em_mean"], new["em_mean"], new["token_f1_mean"], new["n_with_answer"]))

    # Pretty pivot
    print()
    print("=== Pivot: EM (rescored) ===")
    steps = sorted({r[0] for r in table_rows})
    benches = ["hotpotqa_train_val1k", "hotpotqa_dev", "musique_dev", "2wiki_dev", "popqa_dev"]
    header = "dataset                 " + "  ".join(f"step {s:<4}" for s in steps)
    print(header)
    print("-" * len(header))
    by_key = {(r[0], r[1]): r for r in table_rows}
    for b in benches:
        cells = []
        for s in steps:
            r = by_key.get((s, b))
            cells.append(f"{r[3]:>8.3f}" if r else f'{"":>8}')
        print(f"{b:<22}  " + "  ".join(cells))

    print()
    print("=== Pivot: Δ EM (rescored - original) ===")
    print(header)
    print("-" * len(header))
    for b in benches:
        cells = []
        for s in steps:
            r = by_key.get((s, b))
            cells.append(f"{r[3]-r[2]:>+8.3f}" if r else f'{"":>8}')
        print(f"{b:<22}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
