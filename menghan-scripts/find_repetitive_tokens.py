"""
Scan SFT data for completions with bad patterns:
1. Contains <flush>, </flush>, <hr> tags
2. </thinking><thinking> or </answer><answer> appearing 2+ times (re-opening closed tags)
3. Any repetitive token loops at the end
"""
import json
import re

INPUT_PATH = "/share/j_sun/lmlm_multihop/sft_data/gemini_2phase_rollouts_hotpotqa_6k_db_train_end_context_fifths_1203_ex_6k_qa_hotpot_rollouts_classic_retrieval_train_from_start.json"
OUTPUT_PATH = "/share/j_sun/lmlm_multihop/checkpoints/debug/repetitive_token_examples.json"

MIN_REPEATS = 3
TAIL_LENGTH = 500


def check_bad_tags(text):
    """Check for forbidden tags like <flush>, <hr>."""
    bad_tags = ["<flush>", "</flush>", "<hr>", "<hr/>", "<hr />"]
    found = []
    for tag in bad_tags:
        if tag in text:
            found.append(tag)
    return found


def check_reopened_tags(text):
    """Check for </thinking><thinking> or </answer><answer> appearing 2+ times."""
    issues = []
    patterns = [
        ("</thinking><thinking>", re.compile(r'</thinking>\s*<thinking>')),
        ("</thinking> </thinking>", re.compile(r'</thinking>\s*</thinking>')),
        ("</answer><answer>", re.compile(r'</answer>\s*<answer>')),
        ("</answer> </answer>", re.compile(r'</answer>\s*</answer>')),
        ("</thinking><answer></thinking><answer>", re.compile(r'</thinking>\s*<answer>\s*</thinking>\s*<answer>')),
        ("</thinking></answer> loop", re.compile(r'(</thinking>\s*</answer>\s*){2,}')),
        ("<thinking><answer> loop", re.compile(r'(<thinking>\s*<answer>\s*){2,}')),
    ]
    for name, pattern in patterns:
        matches = pattern.findall(text)
        if len(matches) >= 2:
            issues.append((name, len(matches)))
        # For loop patterns, even 1 match is bad
        if "loop" in name and len(matches) >= 1:
            issues.append((name, len(matches)))
    return issues


def detect_tail_repetition(text):
    """Check if the tail of text contains repetitive patterns."""
    tail = text[-TAIL_LENGTH:] if len(text) > TAIL_LENGTH else text

    for pattern_len in range(2, min(101, len(tail) // MIN_REPEATS + 1)):
        candidate = tail[-pattern_len:]
        if not candidate.strip():
            continue
        count = 0
        pos = len(tail)
        while pos >= pattern_len:
            if tail[pos - pattern_len:pos] == candidate:
                count += 1
                pos -= pattern_len
            else:
                break
        if count >= MIN_REPEATS:
            return True, count, candidate

    return False, 0, ""


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    examples = data["examples"]
    print(f"Loaded {len(examples)} examples")

    flagged = []
    for i, ex in enumerate(examples):
        for field in ("annotated_text", "full_response", "triplets"):
            text = ex.get(field, "")
            if not text:
                continue

            reasons = []

            # Check 1: bad tags
            bad_tags = check_bad_tags(text)
            if bad_tags:
                reasons.append(f"bad_tags: {bad_tags}")

            # Check 2: reopened/looping tags
            reopened = check_reopened_tags(text)
            if reopened:
                reasons.append(f"reopened_tags: {reopened}")

            # Check 3: tail repetition
            is_rep, count, pattern = detect_tail_repetition(text)
            if is_rep:
                reasons.append(f"tail_repeat: {count}x '{pattern[:80]}'")

            if reasons:
                flagged.append({
                    "index": i,
                    "field": field,
                    "question": ex.get("question", ""),
                    "reasons": reasons,
                    "tail_500": text[-500:],
                    "full_text_length": len(text),
                })

    print(f"\nFlagged {len(flagged)} / {len(examples)} examples")

    if flagged:
        # Summary by reason type
        from collections import Counter
        reason_counts = Counter()
        for ex in flagged:
            for r in ex["reasons"]:
                reason_counts[r.split(":")[0]] += 1
        print("\nBreakdown:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")

        print("\nExamples:")
        for ex in flagged[:5]:
            print(f"\n--- Index {ex['index']} (field: {ex['field']}) ---")
            print(f"Q: {ex['question']}")
            print(f"Reasons: {ex['reasons']}")
            print(f"Tail: ...{ex['tail_500'][-300:]}")

        with open(OUTPUT_PATH, "w") as f:
            json.dump(flagged, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(flagged)} flagged examples to {OUTPUT_PATH}")
    else:
        print("No bad patterns found!")


if __name__ == "__main__":
    main()
