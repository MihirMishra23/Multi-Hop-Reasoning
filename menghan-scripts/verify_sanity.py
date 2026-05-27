"""Sanity-check the new --log-jsonl logging by comparing JSONL retrieved values
against the <|db_return|> contents that appear in the CSV's phase2_completion_0_0.

Usage:
    /share/j_sun/mx253/envs/lmlm_mx/bin/python menghan-scripts/verify_sanity.py
"""

import ast
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path("/share/j_sun/mx253/Multi-Hop-Reasoning")
CSV_PATH = ROOT / "KG_results_v2/sanity_1.7b_sft_n10.csv"
JSONL_PATH = ROOT / "KG_results_v2/sanity_1.7b_sft_n10.jsonl"

EXPECTED_CSV_COLUMNS = [
    "step", "answer", "phase1_prompt", "phase2_prompt",
    "phase1_completion_0", "phase1_context_0", "generated_db_0",
    "phase1_advantage_0", "db_size_threshold_0",
    "phase2_completion_0_0", "phase2_advantage_0_0", "em_accuracy_0_0",
]

EXPECTED_JSONL_FIELDS = {
    "row_idx", "question", "gold_answer", "em_accuracy", "phase1", "phase2",
}
EXPECTED_PHASE1_FIELDS = {"raw_completion", "parsed_triplets", "num_triplets"}
EXPECTED_PHASE2_FIELDS = {"raw_completion", "final_answer", "stop_reason", "n_turns", "lookups"}
EXPECTED_LOOKUP_FIELDS = {
    "lookup_idx", "query_entity", "query_relationship",
    "retrieved", "returned_values_concat", "retrieval_status",
    "text_after_db_end", "selected_value", "selected_retrieved_idx", "selection_method",
}
EXPECTED_RETRIEVED_FIELDS = {"source_row", "source_tidx", "entity", "relation", "value", "cosine_sim"}

LOOKUP_RE = re.compile(
    r"<\|db_entity\|>(.*?)<\|db_relationship\|>(.*?)<\|db_return\|>(.*?)<\|db_end\|>",
    re.DOTALL,
)

passed = 0
failed = 0

def check(cond, name, msg=""):
    global passed, failed
    if cond:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}{(' — ' + msg) if msg else ''}")
        failed += 1


# ========== Check 1: CSV format unchanged ==========
print("=" * 60)
print("CHECK 1: CSV format unchanged (existing consumers can read it)")
print("=" * 60)
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    rows = list(reader)

check(cols == EXPECTED_CSV_COLUMNS, "CSV columns identical to original",
      f"got {cols}")
check(len(rows) == 10, f"10 rows in CSV (got {len(rows)})")
check(all("generated_db_0" in r and r["generated_db_0"] for r in rows),
      "all rows have non-empty generated_db_0")
check(all("phase2_completion_0_0" in r for r in rows),
      "all rows have phase2_completion_0_0")


# ========== Check 2: JSONL structure ==========
print("\n" + "=" * 60)
print("CHECK 2: JSONL structure (all expected fields present)")
print("=" * 60)
with open(JSONL_PATH) as f:
    jsonl_lines = [json.loads(line) for line in f if line.strip()]

check(len(jsonl_lines) == 10, f"10 lines in JSONL (got {len(jsonl_lines)})")

if jsonl_lines:
    sample = jsonl_lines[0]
    missing_top = EXPECTED_JSONL_FIELDS - set(sample)
    check(not missing_top, "JSONL top-level fields present",
          f"missing: {missing_top}")

    missing_p1 = EXPECTED_PHASE1_FIELDS - set(sample.get("phase1", {}))
    check(not missing_p1, "phase1 fields present", f"missing: {missing_p1}")

    missing_p2 = EXPECTED_PHASE2_FIELDS - set(sample.get("phase2", {}))
    check(not missing_p2, "phase2 fields present", f"missing: {missing_p2}")

    # check at least one chain with at least one lookup, and that lookup is well-formed
    chain_with_lookup = None
    for c in jsonl_lines:
        if c["phase2"]["lookups"]:
            chain_with_lookup = c
            break

    if chain_with_lookup:
        lk = chain_with_lookup["phase2"]["lookups"][0]
        missing_lk = EXPECTED_LOOKUP_FIELDS - set(lk)
        check(not missing_lk, "lookup fields present", f"missing: {missing_lk}")

        if lk.get("retrieved"):
            r = lk["retrieved"][0]
            missing_r = EXPECTED_RETRIEVED_FIELDS - set(r)
            check(not missing_r, "retrieved-triplet fields present",
                  f"missing: {missing_r}")
            check(isinstance(r["source_row"], int) or r["source_row"] is None,
                  "retrieved.source_row is int")
            check(isinstance(r["cosine_sim"], float),
                  "retrieved.cosine_sim is float")
            check(0 <= r["cosine_sim"] <= 1.001,
                  "retrieved.cosine_sim in [0, 1]",
                  f"got {r['cosine_sim']}")
    else:
        print("  [WARN] no chain has any lookup — can't verify lookup fields")


# ========== Check 3: JSONL retrieved values match CSV completion's <|db_return|> ==========
print("\n" + "=" * 60)
print("CHECK 3: JSONL retrieved values match CSV completion <|db_return|>")
print("=" * 60)

mismatches = []
all_match_count = 0
total_lookups_checked = 0

for j, row in zip(jsonl_lines, rows):
    # CSV's "step" field is per-batch (global_step), not per-row, so don't compare directly.
    # JSONL and CSV are both written in row order, so zip() alignment is correct by construction.
    completion = row["phase2_completion_0_0"]
    csv_lookups = LOOKUP_RE.findall(completion)
    jsonl_lookups = j["phase2"]["lookups"]

    if len(csv_lookups) != len(jsonl_lookups):
        mismatches.append({
            "row_idx": j["row_idx"],
            "issue": f"lookup count mismatch: CSV has {len(csv_lookups)}, JSONL has {len(jsonl_lookups)}"
        })
        continue

    for li, ((cX, cY, cZ), jlk) in enumerate(zip(csv_lookups, jsonl_lookups)):
        total_lookups_checked += 1
        csv_returned = cZ.strip()
        jsonl_returned = jlk["returned_values_concat"]
        status = jlk.get("retrieval_status", "ok")
        # OK case: CSV and JSONL match exactly when retrieval succeeded.
        if csv_returned == jsonl_returned and status == "ok":
            all_match_count += 1
        # Failure case: CSV gets the placeholder "unknown" injected when retrieval failed.
        # JSONL records empty returned_values_concat and non-"ok" retrieval_status.
        elif csv_returned == "unknown" and status != "ok" and jsonl_returned == "":
            all_match_count += 1
        else:
            mismatches.append({
                "row_idx": j["row_idx"],
                "lookup_idx": li,
                "csv_return": csv_returned[:80],
                "jsonl_return": jsonl_returned[:80],
                "status": status,
            })

check(len(mismatches) == 0,
      f"all {total_lookups_checked} CSV lookups match JSONL returned_values_concat",
      f"{len(mismatches)} mismatches (showing first 3): {mismatches[:3]}")


# ========== Check 4: source_row / source_tidx valid ==========
print("\n" + "=" * 60)
print("CHECK 4: source_row + source_tidx point into a valid triplet in unified KB")
print("=" * 60)

# Build global KB index from CSV (row_idx -> list of triplets).
# CSV's "step" field is per-batch, NOT per-row. Use the CSV position (i) as the row_idx,
# which matches the convention used by run_inference_csv.py when writing JSONL.
row_kbs = {}
for i, row in enumerate(rows):
    try:
        triplets = ast.literal_eval(row["generated_db_0"])
        row_kbs[i] = list(triplets)
    except Exception as e:
        print(f"  [WARN] couldn't parse row {i} KB: {e}")
        row_kbs[i] = []

bad_sources = 0
total_retrieved = 0
for j in jsonl_lines:
    for lk in j["phase2"]["lookups"]:
        for r in lk.get("retrieved", []):
            total_retrieved += 1
            src_row = r["source_row"]
            src_tidx = r["source_tidx"]
            if src_row not in row_kbs:
                # source from row 0-9 in sanity is only valid; in real 1000-row run, ANY row 0-999 is valid
                bad_sources += 1
                continue
            kb = row_kbs[src_row]
            if not (0 <= src_tidx < len(kb)):
                bad_sources += 1
                continue
            # check that the triplet value matches what was logged
            try:
                src_triplet = kb[src_tidx]
                if str(src_triplet[2]).strip() != str(r["value"]).strip():
                    bad_sources += 1
            except Exception:
                bad_sources += 1

check(bad_sources == 0,
      f"all {total_retrieved} retrieved triplets resolve to valid (source_row, source_tidx)",
      f"{bad_sources} bad ones")


# ========== Check 5: selected_value parsing ==========
print("\n" + "=" * 60)
print("CHECK 5: selected_value sanity (when set, must be a substring of text_after_db_end)")
print("=" * 60)

bad_select = 0
total_with_select = 0
for j in jsonl_lines:
    for lk in j["phase2"]["lookups"]:
        sv = lk.get("selected_value")
        if sv is None:
            continue
        total_with_select += 1
        if sv.lower() not in lk.get("text_after_db_end", "").lower():
            bad_select += 1

check(bad_select == 0,
      f"all {total_with_select} selected_values are substrings of text_after_db_end",
      f"{bad_select} bad ones")


# ========== Summary ==========
print("\n" + "=" * 60)
print(f"SUMMARY: {passed} passed, {failed} failed")
print("=" * 60)
if failed:
    sys.exit(1)
