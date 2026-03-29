"""
Phase 1: Knowledge Database Quality Evaluation (Async + Pydantic Structured Output)
Usage:
    python phase1/evaluate.py --num_rows 10
    python phase1/evaluate.py --num_rows 10 --concurrency 10
    python phase1/evaluate.py                   # evaluate all rows
    python phase1/evaluate.py --num_rows 5 --seed 123
"""

import argparse
import ast
import asyncio
import csv
import json
import random
import os
from enum import Enum
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ── Load .env ───────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── Config ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "final_v2.2_0.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "phase1_results_flash_final_v2.2_0.json")

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text()
USER_PROMPT_TEMPLATE = (Path(__file__).parent / "user_prompt.txt").read_text()


# ── Pydantic Schema ────────────────────────────────────────────────────
class TripletEvaluation(BaseModel):
    triplet_index: int
    triplet: str
    reasoning: str
    faithfulness: Literal["faithful", "hallucinated"]
    quality_issues: list[Literal[
        "ambiguous_entity_value",
        "trivial",
        "non_specific",
        "malformed",
        "reversed_roles",
    ]]


class EvaluationResult(BaseModel):
    triplet_evaluations: list[TripletEvaluation]


# ── Helpers ─────────────────────────────────────────────────────────────
def parse_db_column(raw: str) -> list[tuple[str, str, str]]:
    """Parse a generated_db column (Python list-of-tuples string) into triplets."""
    try:
        triplets = ast.literal_eval(raw)
        return [(str(e), str(r), str(v)) for e, r, v in triplets]
    except Exception as e:
        print(f"  [WARN] Failed to parse db column: {e}")
        return []


def format_triplets(triplets: list[tuple[str, str, str]]) -> str:
    """Format triplets as numbered list for the prompt."""
    lines = []
    for i, (e, r, v) in enumerate(triplets):
        lines.append(f"{i}: ({e}, {r}, {v})")
    return "\n".join(lines)


def get_db_columns(header: list[str]) -> list[str]:
    """Find all generated_db_* columns, sorted by index."""
    db_cols = [c for c in header if c.startswith("generated_db_")]
    db_cols.sort(key=lambda c: int(c.split("_")[-1]))
    return db_cols


def compute_score(evals: list[TripletEvaluation], total_triplets: int) -> dict:
    """Compute scores from judge output."""
    faithful = [e for e in evals if e.faithfulness == "faithful"]
    clean = [e for e in faithful if len(e.quality_issues) == 0]
    hallucinated = [e for e in evals if e.faithfulness == "hallucinated"]

    return {
        "total_triplets": total_triplets,
        "faithful_count": len(faithful),
        "hallucinated_count": len(hallucinated),
        "clean_count": len(clean),
        "faithfulness_score": len(faithful) / total_triplets if total_triplets > 0 else 0,
        "quality_score": len(clean) / len(faithful) if len(faithful) > 0 else 0,
        "overall_score": len(clean) / total_triplets if total_triplets > 0 else 0,
    }


# ── Async API call ──────────────────────────────────────────────────────
async def call_gemini(
    client: genai.Client,
    semaphore: asyncio.Semaphore,
    context: str,
    triplets_str: str,
    label: str,
    max_retries: int = 3,
) -> EvaluationResult | None:
    """Call Gemini async with Pydantic structured output."""
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
                        response_mime_type="application/json",
                        response_schema=EvaluationResult,
                        max_output_tokens=65536,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=4096,
                        ),
                    ),
                )
            except Exception as e:
                print(f"  [ERROR] API call failed for {label} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None

        # Parse with Pydantic — response.parsed should work directly
        try:
            if response.parsed:
                return response.parsed
        except Exception:
            pass

        # Fallback: parse from text
        try:
            if response.text:
                data = json.loads(response.text)
                return EvaluationResult.model_validate(data)
        except Exception as e:
            print(f"  [WARN] Failed to parse for {label} (attempt {attempt + 1}): {e}")

            # Save debug info
            debug_dir = Path(__file__).parent / "debug"
            debug_dir.mkdir(exist_ok=True)
            safe_label = label.replace("/", "-").replace(" ", "_")
            debug_file = debug_dir / f"failed_{safe_label}_attempt{attempt + 1}.txt"
            with open(debug_file, "w") as f:
                f.write(f"=== Error: {e} ===\n")
                if response.candidates:
                    candidate = response.candidates[0]
                    f.write(f"finish_reason: {getattr(candidate, 'finish_reason', 'unknown')}\n")
                    f.write(f"token_count: {getattr(response, 'usage_metadata', 'unknown')}\n")
                f.write(f"\n=== Raw text ===\n")
                f.write(response.text or "(empty)")
            print(f"           Saved debug output to {debug_file}")

        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)

    return None


# ── Evaluate one row ────────────────────────────────────────────────────
async def evaluate_row(
    client: genai.Client,
    semaphore: asyncio.Semaphore,
    row_idx: int,
    row: dict,
    db_cols: list[str],
    total_rows: int,
) -> dict:
    """Evaluate all DBs in one row concurrently."""
    context = row.get("phase1_context_0", "")

    row_result = {
        "row_index": row_idx,
        "context": context,
        "db_results": [],
        "row_score": 0.0,
    }

    if not context.strip():
        print(f"  Row {row_idx + 1}/{total_rows}: [SKIP] Empty context")
        return row_result

    # Build tasks for all DBs in this row
    tasks = []
    task_meta = []

    for db_col in db_cols:
        raw = row.get(db_col, "")
        if not raw.strip():
            continue

        triplets = parse_db_column(raw)
        if not triplets:
            continue

        triplets_str = format_triplets(triplets)
        label = f"Row {row_idx + 1}/{total_rows} - {db_col}"
        tasks.append(call_gemini(client, semaphore, context, triplets_str, label))
        task_meta.append((db_col, triplets))

    if not tasks:
        return row_result

    # Run all DB evaluations for this row concurrently
    results = await asyncio.gather(*tasks)

    db_scores = []
    for (db_col, triplets), evaluation in zip(task_meta, results):
        if evaluation is None:
            print(f"  Row {row_idx + 1} - {db_col}: [ERROR]")
            continue

        scores = compute_score(evaluation.triplet_evaluations, len(triplets))

        # Serialize Pydantic model to dict for JSON output
        row_result["db_results"].append({
            "db_column": db_col,
            "num_triplets": len(triplets),
            "evaluation": evaluation.model_dump(),
            "scores": scores,
        })
        db_scores.append(scores["overall_score"])

    if db_scores:
        row_result["row_score"] = sum(db_scores) / len(db_scores)

    print(f"  Row {row_idx + 1}/{total_rows}: score={row_result['row_score']:.2%} "
          f"({len(db_scores)} DBs evaluated)")

    return row_result


# ── Main ────────────────────────────────────────────────────────────────
async def async_main():
    parser = argparse.ArgumentParser(description="Phase 1: Evaluate generated knowledge databases")
    parser.add_argument("--num_rows", type=int, default=None,
                        help="Number of rows to randomly sample. If not set, evaluate all rows.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent API calls (default: 5)")
    parser.add_argument("--csv", type=str, default=CSV_PATH,
                        help="Path to CSV file")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help="Path to output JSON file")
    args = parser.parse_args()

    # ── Init client ──
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Set it in .env or as an environment variable.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    semaphore = asyncio.Semaphore(args.concurrency)

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        db_cols = get_db_columns(header)
        print(f"Found {len(db_cols)} database columns: {db_cols}")
        rows = list(reader)

    # ── Random sampling ──
    if args.num_rows is not None and args.num_rows < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.num_rows)
        print(f"Randomly sampled {args.num_rows} rows (seed={args.seed})")
    else:
        print(f"Evaluating all {len(rows)} rows")

    print(f"Concurrency: {args.concurrency}")

    # ── Run all rows concurrently ──
    tasks = [
        evaluate_row(client, semaphore, row_idx, row, db_cols, len(rows))
        for row_idx, row in enumerate(rows)
    ]
    all_results = await asyncio.gather(*tasks)

    # ── Save results ──
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # ── Final score ──
    valid_rows = [r for r in all_results if r["row_score"] > 0]
    if valid_rows:
        final_score = sum(r["row_score"] for r in valid_rows) / len(valid_rows)
        print(f"\n{'='*60}")
        print(f"Rows evaluated: {len(valid_rows)}")
        print(f"Final score: {final_score:.2%}")
        print(f"(1.0 = perfect, 0.0 = worst)")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()