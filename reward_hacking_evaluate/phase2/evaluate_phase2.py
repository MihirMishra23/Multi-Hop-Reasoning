"""
Phase 2: Completion Quality Evaluation (Async + Pydantic Structured Output)
Usage:
    python phase2/evaluate.py --num_rows 10
    python phase2/evaluate.py --num_rows 10 --concurrency 5
    python phase2/evaluate.py                   # evaluate all rows
"""

import argparse
import ast
import asyncio
import csv
import json
import random
import re
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

# ── Load .env ───────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── Config ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "final_v2.2_0.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "phase2_results_final_v2.2_0.json")

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text()
USER_PROMPT_TEMPLATE = (Path(__file__).parent / "user_prompt.txt").read_text()


# ── Pydantic Schema ────────────────────────────────────────────────────
class CompletionEvaluation(BaseModel):
    grounding_reasoning: str
    grounding: Literal["fully_grounded", "partially_grounded", "ungrounded", "no_answer"]
    reasoning_reasoning: str
    reasoning: Literal["correct", "minor_error", "major_error"]


# ── Preprocessing ───────────────────────────────────────────────────────
def clean_completion(text: str) -> str:
    """Remove endoftext tokens and clean up whitespace."""
    text = re.sub(r'<\|endoftext\|>', '', text)
    text = text.strip()
    return text


# ── CSV Column Discovery ───────────────────────────────────────────────
def get_db_columns(header: list[str]) -> list[str]:
    """Find all generated_db_* columns, sorted by index."""
    db_cols = [c for c in header if c.startswith("generated_db_")]
    db_cols.sort(key=lambda c: int(c.split("_")[-1]))
    return db_cols


def get_completion_columns(header: list[str], db_idx: int) -> list[str]:
    """Find all phase2_completion_{db_idx}_* columns, sorted by index."""
    prefix = f"phase2_completion_{db_idx}_"
    comp_cols = [c for c in header if c.startswith(prefix)]
    comp_cols.sort(key=lambda c: int(c.split("_")[-1]))
    return comp_cols


def format_triplets(raw: str) -> str:
    """Parse and format DB triplets for the prompt."""
    try:
        triplets = ast.literal_eval(raw)
        lines = []
        for i, (e, r, v) in enumerate(triplets):
            lines.append(f"{i}: ({e}, {r}, {v})")
        return "\n".join(lines)
    except Exception:
        return raw


# ── Async API call ──────────────────────────────────────────────────────
async def call_gemini(
    client: genai.Client,
    semaphore: asyncio.Semaphore,
    question: str,
    db_triplets_str: str,
    completion: str,
    label: str,
    max_retries: int = 3,
) -> CompletionEvaluation | None:
    """Call Gemini async with Pydantic structured output."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        database_triplets=db_triplets_str,
        completion=completion,
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
                        response_schema=CompletionEvaluation,
                        max_output_tokens=4096,
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

        # Parse with Pydantic
        try:
            if response.parsed:
                return response.parsed
        except Exception:
            pass

        # Fallback: parse from text
        try:
            if response.text:
                data = json.loads(response.text)
                return CompletionEvaluation.model_validate(data)
        except Exception as e:
            print(f"  [WARN] Failed to parse for {label} (attempt {attempt + 1}): {e}")

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
    header: list[str],
    total_rows: int,
) -> dict:
    """Evaluate all completions in one row."""
    question = row.get("phase2_prompt", "")

    row_result = {
        "row_index": row_idx,
        "question": question,
        "db_results": [],
    }

    if not question.strip():
        print(f"  Row {row_idx + 1}/{total_rows}: [SKIP] Empty question")
        return row_result

    # Build tasks for all completions across all DBs
    tasks = []
    task_meta = []  # (db_col, comp_col) parallel to tasks

    for db_col in db_cols:
        raw_db = row.get(db_col, "")
        if not raw_db.strip():
            continue

        db_idx = int(db_col.split("_")[-1])
        comp_cols = get_completion_columns(header, db_idx)

        for comp_col in comp_cols:
            raw_comp = row.get(comp_col, "")
            if not raw_comp.strip():
                continue

            label = f"Row {row_idx + 1}/{total_rows} - {comp_col}"
            db_triplets_str = format_triplets(raw_db)
            cleaned_comp = clean_completion(raw_comp)

            tasks.append(call_gemini(
                client, semaphore, question, db_triplets_str, cleaned_comp, label
            ))
            task_meta.append((db_col, comp_col, cleaned_comp, raw_db))

    if not tasks:
        print(f"  Row {row_idx + 1}/{total_rows}: [SKIP] No completions")
        return row_result

    # Run all evaluations concurrently
    results = await asyncio.gather(*tasks)

    # Group results by DB
    db_groups = {}  # db_col -> list of completion results
    for (db_col, comp_col, cleaned_comp, raw_db), evaluation in zip(task_meta, results):
        if db_col not in db_groups:
            db_groups[db_col] = {
                "db_column": db_col,
                "db_triplets": format_triplets(raw_db),
                "completions": [],
            }

        comp_result = {
            "completion_column": comp_col,
            "completion_text": cleaned_comp,
            "evaluation": evaluation.model_dump() if evaluation else None,
            "grounding": evaluation.grounding if evaluation else None,
            "reasoning": evaluation.reasoning if evaluation else None,
        }

        if evaluation is None:
            print(f"  Row {row_idx + 1} - {comp_col}: [ERROR]")

        db_groups[db_col]["completions"].append(comp_result)

    row_result["db_results"] = list(db_groups.values())

    # Print summary
    total_comps = sum(len(db["completions"]) for db in row_result["db_results"])
    evaluated = sum(
        1 for db in row_result["db_results"]
        for c in db["completions"] if c["evaluation"] is not None
    )
    print(f"  Row {row_idx + 1}/{total_rows}: {evaluated}/{total_comps} completions evaluated")

    return row_result


# ── Main ────────────────────────────────────────────────────────────────
async def async_main():
    parser = argparse.ArgumentParser(description="Phase 2: Evaluate completion quality")
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
        evaluate_row(client, semaphore, row_idx, row, db_cols, header, len(rows))
        for row_idx, row in enumerate(rows)
    ]
    all_results = await asyncio.gather(*tasks)

    # ── Save results ──
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()