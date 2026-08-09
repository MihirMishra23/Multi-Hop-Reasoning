"""
Missing-Facts (Recall) Judge for extracted knowledge databases.

For each sampled passage we already know, from the phase-1 faithfulness judge:
    N        = total_triplets        (triplets the model extracted)
    faithful = faithful_count        (triplets grounded in the passage)

This script adds the missing piece for recall: it asks Gemini to count how many
salient facts in the passage are NOT covered by the extracted triplets.

    missing  = # facts in passage but absent from the DB   (judged here)

Per-passage metrics (treat faithful triplets as true positives, hallucinated
ones as false positives, missing facts as false negatives):

    precision = faithful / N
    recall    = faithful / (faithful + missing)
    f1        = 2 * P * R / (P + R)

We macro-average over the sampled passages (and also report a micro average).

Usage:
    python evaluate_missing_facts.py --model 1.7b_sft --num_rows 100 --concurrency 15
    python evaluate_missing_facts.py --model all      --num_rows 100 --concurrency 15
"""

import argparse
import asyncio
import csv
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Paths / config ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

HERE = Path(__file__).resolve().parent
SYSTEM_PROMPT = (HERE / "system_prompt.txt").read_text()
USER_PROMPT_TEMPLATE = (HERE / "user_prompt.txt").read_text()

JUDGE_DIR = REPO_ROOT / "KG_results_v2" / "phase1_judge"

# key -> (display name, phase1 judge json filename)
MODELS = {
    "1.7b_sft":  ("SFT (1.7B)",          "phase1_qwen3_1.7b_sft_n1000_prod.json"),
    "1.7b_grpo": ("GRPO (1.7B)",         "phase1_qwen3_1.7b_grpo_ckpt500_n1000_prod.json"),
    "4b_sft":    ("SFT (4B)",            "phase1_qwen3_4b_sft_n1000_prod.json"),
    "4b_grpo":   ("GRPO (4B)",           "phase1_qwen3_4b_grpo_ckpt500_n1000_prod.json"),
    "atlas":     ("AutoSchemaKG (8B)",   "phase1_atlas_hotpotqa_n300.json"),
    "edc":       ("EDC (Mistral-7B)",    "phase1_edc_hotpotqa_n100.json"),
}


# ── Output parsing ─────────────────────────────────────────────────────────
def count_missing(text: str) -> int:
    """Count non-empty lines in the judge output. 'All covered!' -> 0."""
    if not text:
        return 0
    norm = text.strip().lower().rstrip("!. ")
    if norm == "all covered":
        return 0
    count = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # strip common list markers so they don't inflate/deflate the count
        stripped = line.lstrip("-*0123456789.)# ").strip()
        if not stripped:
            continue
        # a stray "all covered!" mixed into otherwise-empty output -> 0
        if stripped.lower().rstrip("!. ") == "all covered":
            continue
        count += 1
    return count


def format_triplets(triplet_evals: list[dict]) -> str:
    """Build a numbered triplet list from phase-1 triplet_evaluations entries."""
    lines = []
    for i, te in enumerate(triplet_evals):
        lines.append(f"{i}: {te.get('triplet', '')}")
    return "\n".join(lines)


# ── Async Gemini call ────────────────────────────────────────────────────
async def call_gemini(
    client: genai.Client,
    semaphore: asyncio.Semaphore,
    context: str,
    triplets_str: str,
    label: str,
    max_retries: int = 3,
) -> str | None:
    """Return the raw text listing missing facts (or 'All covered!'), or None."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        database_triplets=triplets_str,
    )
    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=8192,
                        thinking_config=types.ThinkingConfig(thinking_budget=4096),
                    ),
                )
            except Exception as e:
                print(f"  [ERROR] API failed for {label} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
        if response.text:
            return response.text
        print(f"  [WARN] empty response for {label} (attempt {attempt + 1})")
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
    return None


async def judge_row(client, semaphore, row, label):
    """Compute P/R/F1 for one passage. Returns a dict row or None to skip."""
    context = row.get("context", "")
    db_results = row.get("db_results", [])
    if not context.strip() or not db_results:
        return None
    db = db_results[0]
    scores = db.get("scores", {})
    N = scores.get("total_triplets", 0)
    faithful = scores.get("faithful_count", 0)
    if N <= 0:
        return None

    triplet_evals = db.get("evaluation", {}).get("triplet_evaluations", [])
    triplets_str = format_triplets(triplet_evals)

    text = await call_gemini(client, semaphore, context, triplets_str, label)
    if text is None:
        return None
    missing = count_missing(text)

    precision = faithful / N if N > 0 else 0.0
    recall = faithful / (faithful + missing) if (faithful + missing) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "row_index": row.get("row_index"),
        "N": N,
        "faithful": faithful,
        "missing": missing,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing_text": text.strip(),
    }


def load_done_indices(out_dir, key, tag=""):
    """row_index values already judged in a prior batch (to avoid overlap)."""
    done = set()
    for t in {"", tag}:
        p = out_dir / f"missing_facts_{key}{t}.csv"
        if p.exists():
            for r in csv.DictReader(open(p)):
                try:
                    done.add(int(r["row_index"]))
                except (KeyError, ValueError):
                    pass
    return done


# ── Per-model driver ───────────────────────────────────────────────────────
async def run_model(client, semaphore, key, num_rows, seed, out_dir, tag="", exclude_done=False):
    display, fname = MODELS[key]
    path = JUDGE_DIR / fname
    print(f"\n=== {display}  ({fname}) ===")
    data = json.load(open(path))

    pool = data
    if exclude_done:
        done = load_done_indices(out_dir, key, tag)
        pool = [row for row in data if row.get("row_index") not in done]
        print(f"Excluding {len(done)} already-judged passages -> pool of {len(pool)}")

    if num_rows is not None and num_rows < len(pool):
        rng = random.Random(seed)
        sampled = rng.sample(pool, num_rows)
        print(f"Sampled {num_rows} of {len(pool)} passages (seed={seed})")
    else:
        sampled = pool
        print(f"Using all {len(pool)} passages")

    tasks = [
        judge_row(client, semaphore, row, f"{key} row {row.get('row_index')}")
        for row in sampled
    ]
    results = [r for r in await asyncio.gather(*tasks) if r is not None]

    # per-row CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"missing_facts_{key}{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "N", "faithful", "missing", "precision", "recall", "f1", "missing_text"])
        for r in results:
            w.writerow([r["row_index"], r["N"], r["faithful"], r["missing"],
                        f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                        r["missing_text"].replace("\n", " | ")])
    print(f"  wrote {csv_path} ({len(results)} passages)")

    if not results:
        return None
    n = len(results)
    macro_p = sum(r["precision"] for r in results) / n
    macro_r = sum(r["recall"] for r in results) / n
    macro_f1 = sum(r["f1"] for r in results) / n
    # micro (pooled counts)
    sN = sum(r["N"] for r in results)
    sF = sum(r["faithful"] for r in results)
    sM = sum(r["missing"] for r in results)
    micro_p = sF / sN if sN else 0.0
    micro_r = sF / (sF + sM) if (sF + sM) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    print(f"  macro  P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}")
    print(f"  micro  P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}")
    return {
        "key": key, "model": display, "n_passages": n,
        "precision": macro_p, "recall": macro_r, "f1": macro_f1,
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
    }


async def async_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    help=f"'all' or comma-separated subset of {list(MODELS)}")
    ap.add_argument("--num_rows", type=int, default=100,
                    help="passages to sample per model (default 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--out_dir", default=str(HERE / "results"))
    ap.add_argument("--tag", default="",
                    help="suffix for output files, e.g. '_b2' for a second batch")
    ap.add_argument("--exclude_done", action="store_true",
                    help="skip row_index values already judged in prior batch CSVs (no overlap)")
    args = ap.parse_args()

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env or environment.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    semaphore = asyncio.Semaphore(args.concurrency)
    out_dir = Path(args.out_dir)

    keys = list(MODELS) if args.model == "all" else [k.strip() for k in args.model.split(",") if k.strip()]
    for k in keys:
        if k not in MODELS:
            raise ValueError(f"unknown model '{k}'. choices: {list(MODELS)} or 'all'")

    summary = []
    for k in keys:
        s = await run_model(client, semaphore, k, args.num_rows, args.seed, out_dir,
                            tag=args.tag, exclude_done=args.exclude_done)
        if s:
            summary.append(s)

    # summary table
    summary_path = out_dir / f"missing_facts_summary{args.tag}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_passages", "precision", "recall", "f1",
                    "micro_precision", "micro_recall", "micro_f1"])
        for s in summary:
            w.writerow([s["model"], s["n_passages"],
                        f"{s['precision']:.4f}", f"{s['recall']:.4f}", f"{s['f1']:.4f}",
                        f"{s['micro_precision']:.4f}", f"{s['micro_recall']:.4f}", f"{s['micro_f1']:.4f}"])

    print(f"\n{'='*70}\nSUMMARY (macro-averaged per passage)\n{'='*70}")
    print(f"{'Model':<22}{'Precision':>11}{'Recall':>10}{'F1':>10}")
    for s in summary:
        print(f"{s['model']:<22}{s['precision']:>11.3f}{s['recall']:>10.3f}{s['f1']:>10.3f}")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(async_main())
