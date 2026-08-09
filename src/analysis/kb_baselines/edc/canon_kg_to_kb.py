"""Convert EDC official-pipeline output (canon_kg.txt) → KBevo KB JSON.

EDC's canon_kg.txt has one line per input passage, each line being a
Python-literal list of [subj, rel, obj] triplets (post-canonicalization).

The KBevo KB JSON schema (also used by atlas/gemini KBs) is:
    {entities, relationships, return_values, triplets, _meta}

with triplets as a flat deduped list of [subj, rel, obj] across all passages.

Usage:
    python kb_baselines/edc/canon_kg_to_kb.py \\
        --canon-kg /share/j_sun/mx253/edc/output/2wiki_dev_n1000/iter0/canon_kg.txt \\
        --dataset 2wiki \\
        --output kb_baselines/edc/output/kbs/edc_official_mistral7b_2wiki_dev_n1000_seed42_kb.json
"""

import argparse
import ast
import json
import os
from pathlib import Path


def _entity_norm(s: str) -> str:
    """EDC OIE (with webnlg-style few-shot) emits entities with underscores
    (e.g. 'Robin_Barton'). Replacing with spaces matches the natural-language
    entity convention used in ATLAS/Gemini KBs, which:
      (a) makes substring matching work for reachability analysis, and
      (b) gives the reasoner LLM cleaner context (no underscored noise) at
          retrieval time.
    Applied to subject + object; relation labels stay as-is.
    """
    return s.replace("_", " ").strip()


def parse_canon_kg(path: str):
    """Read canon_kg.txt → list of per-passage triplet lists."""
    per_passage = []
    with open(path) as f:
        for line_idx, raw in enumerate(f):
            raw = raw.rstrip("\n")
            if not raw.strip():
                per_passage.append([])
                continue
            try:
                lst = ast.literal_eval(raw)
            except (SyntaxError, ValueError) as e:
                print(f"WARN: line {line_idx} not parseable, treating as empty: {e}")
                per_passage.append([])
                continue
            # Sanitize: keep only valid 3-tuples of strings; underscore→space on s/o.
            kept = []
            for t in lst or []:
                if (isinstance(t, (list, tuple)) and len(t) >= 3
                        and all(isinstance(x, str) for x in t[:3])):
                    kept.append([_entity_norm(t[0]), t[1].strip(), _entity_norm(t[2])])
            per_passage.append(kept)
    return per_passage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canon-kg", required=True, help="path to EDC canon_kg.txt")
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wiki"])
    p.add_argument("--n-passages-expected", type=int, default=None,
                   help="sanity-check: expected number of lines (passages)")
    p.add_argument("--output", required=True, help="output KB JSON path")
    args = p.parse_args()

    print(f"Reading EDC canon_kg from {args.canon_kg} ...")
    per_passage = parse_canon_kg(args.canon_kg)
    print(f"  {len(per_passage)} passages")
    if args.n_passages_expected and len(per_passage) != args.n_passages_expected:
        print(f"  WARN: expected {args.n_passages_expected}, got {len(per_passage)}")

    raw_count = sum(len(p) for p in per_passage)
    print(f"  raw triplets: {raw_count}")

    # Dedup triplets across passages while preserving order of first appearance
    seen = set()
    triplets = []
    for plist in per_passage:
        for t in plist:
            key = (t[0], t[1], t[2])
            if key in seen:
                continue
            seen.add(key)
            triplets.append(t)
    print(f"  unique triplets: {len(triplets)}")

    # Build entity / relationship / return_value lists (order of first appearance)
    entities, relationships, return_values = [], [], []
    seen_e, seen_r, seen_v = set(), set(), set()
    for s, r, o in triplets:
        if s not in seen_e:
            seen_e.add(s); entities.append(s)
        if r not in seen_r:
            seen_r.add(r); relationships.append(r)
        if o not in seen_v:
            seen_v.add(o); return_values.append(o)
    print(f"  unique entities: {len(entities)}, relationships: {len(relationships)}, "
          f"return_values: {len(return_values)}")

    n_passages_with_triplets = sum(1 for p in per_passage if p)
    n_empty = sum(1 for p in per_passage if not p)
    print(f"  passages with ≥1 triplet: {n_passages_with_triplets} / {len(per_passage)}  "
          f"(empty: {n_empty})")

    out = {
        "entities": entities,
        "relationships": relationships,
        "return_values": return_values,
        "triplets": triplets,
        "_meta": {
            "source": "edc-official-mistral7b",
            "edc_repo": "/share/j_sun/mx253/edc",
            "edc_canon_kg": args.canon_kg,
            "dataset": args.dataset,
            "extractor": "mistralai/Mistral-7B-Instruct-v0.2",
            "schema_canonicalization_embedder": "intfloat/e5-mistral-7b-instruct",
            "schema_canonicalization_verifier": "mistralai/Mistral-7B-Instruct-v0.2",
            "mode": "self_canonicalization (--enrich_schema)",
            "refinement_iterations": 0,
            "n_passages": len(per_passage),
            "n_passages_with_triplets": n_passages_with_triplets,
            "n_raw_triplets": raw_count,
            "n_unique_triplets": len(triplets),
            "n_unique_entities": len(entities),
            "n_unique_relations": len(relationships),
            "n_unique_return_values": len(return_values),
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote KB JSON → {args.output}  ({os.path.getsize(args.output)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()