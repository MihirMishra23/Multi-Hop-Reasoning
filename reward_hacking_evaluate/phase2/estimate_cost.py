"""
Estimate Phase 2 API costs before running.
Usage: python phase2/estimate_cost.py
"""

import ast
import csv
import re
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "test_rollouts_22316.csv")
SYSTEM_PROMPT = (open(os.path.join(os.path.dirname(__file__), "system_prompt.txt")).read())

# Gemini 2.5 Flash pricing (paid tier)
INPUT_PRICE_PER_M = 0.30   # $/1M input tokens
OUTPUT_PRICE_PER_M = 2.50  # $/1M output tokens (includes thinking tokens)

# Rough estimate: 1 token ≈ 4 chars
CHARS_PER_TOKEN = 4


def get_db_columns(header):
    db_cols = [c for c in header if c.startswith("generated_db_")]
    db_cols.sort(key=lambda c: int(c.split("_")[-1]))
    return db_cols


def get_completion_columns(header, db_idx):
    prefix = f"phase2_completion_{db_idx}_"
    comp_cols = [c for c in header if c.startswith(prefix)]
    comp_cols.sort(key=lambda c: int(c.split("_")[-1]))
    return comp_cols


def estimate_tokens(text):
    """Rough token estimate based on character count."""
    clean = re.sub(r'<\|endoftext\|>', '', text)
    return len(clean) / CHARS_PER_TOKEN


def main():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        db_cols = get_db_columns(header)
        rows = list(reader)

    system_tokens = estimate_tokens(SYSTEM_PROMPT)

    total_completions = 0
    total_input_tokens = 0
    completion_lengths = []

    for row in rows:
        question = row.get("phase2_prompt", "")
        question_tokens = estimate_tokens(question)

        for db_col in db_cols:
            raw_db = row.get(db_col, "")
            if not raw_db.strip():
                continue
            db_tokens = estimate_tokens(raw_db)

            db_idx = int(db_col.split("_")[-1])
            comp_cols = get_completion_columns(header, db_idx)

            for comp_col in comp_cols:
                raw_comp = row.get(comp_col, "")
                if not raw_comp.strip():
                    continue

                comp_tokens = estimate_tokens(raw_comp)
                input_tokens = system_tokens + question_tokens + db_tokens + comp_tokens

                total_completions += 1
                total_input_tokens += input_tokens
                completion_lengths.append(comp_tokens)

    # Output estimate: Pydantic schema ~200 tokens + thinking budget per call
    thinking_per_call = 4096  # your budget setting
    output_per_call = 200
    total_output_tokens = total_completions * (output_per_call + thinking_per_call)

    # Costs
    input_cost = (total_input_tokens / 1_000_000) * INPUT_PRICE_PER_M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
    total_cost = input_cost + output_cost

    avg_comp = sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0
    max_comp = max(completion_lengths) if completion_lengths else 0
    min_comp = min(completion_lengths) if completion_lengths else 0

    print(f"{'='*55}")
    print(f"  Phase 2 Cost Estimate (Gemini 2.5 Flash)")
    print(f"{'='*55}")
    print(f"  Total completions:       {total_completions}")
    print(f"  Avg completion tokens:   {avg_comp:,.0f}")
    print(f"  Min completion tokens:   {min_comp:,.0f}")
    print(f"  Max completion tokens:   {max_comp:,.0f}")
    print(f"\n  Total input tokens:      {total_input_tokens:,.0f}")
    print(f"  Total output tokens:     {total_output_tokens:,.0f}")
    print(f"    (includes thinking budget of {thinking_per_call} per call)")
    print(f"\n{'='*55}")
    print(f"  Cost Breakdown (paid tier)")
    print(f"{'='*55}")
    print(f"  Input  ($0.30/1M):       ${input_cost:.2f}")
    print(f"  Output ($2.50/1M):       ${output_cost:.2f}")
    print(f"  ─────────────────────────────────")
    print(f"  Total (upper bound):     ${total_cost:.2f}")
    print(f"  Note: Thinking tokens usually < budget,")
    print(f"        real cost likely 30-50% less.")
    print(f"\n  Per --num_rows 10:       ~${total_cost / len(rows) * 10:.2f}")
    print(f"  Per --num_rows 100:      ~${total_cost / len(rows) * 100:.2f}")
    print(f"\n  Free tier: $0 but data used to improve Google products")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()