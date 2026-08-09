"""Convert atlas-rag KG extraction output -> KBevo unified KB JSON.

atlas-rag dumps each example as JSONL with three stage outputs:
    Stage 1 entity_relation_dict:       [{"Head","Relation","Tail"}, ...]
    Stage 2 event_entity_relation_dict: [{"Event","Entity":[..]}, ...]   (no Relation field!)
    Stage 3 event_relation_dict:        [{"Head","Relation","Tail"}, ...] (h,t are sentences)

Flattening into the KBevo unified-KB schema:
    Stage 1: passes through as (Head, Relation, Tail)
    Stage 2: each (Event, [E1,E2,...]) -> N triplets (Ei, '<rel>', Event)
             default rel = "involved_in" — entity-as-head matches KBevo retrieval style
             (KBevo encodes f"{entity} {rel}" into FAISS, so short head is preferable)
    Stage 3: passes through as (Head, Relation, Tail) — h/t are event sentences

Per the W2 rebuttal plan: NO inverse triplets added here. Whether inverses get
expanded at retrieval time is decided by DatabaseManager.load_database's
use_inverses flag.

Usage:
    PYTHONPATH=src python kb_baselines/autoschemakg/atlas_to_kbevo_kb.py \
        --atlas-output-dir kb_baselines/autoschemakg/output/extractions/hotpotqa_dev_n5_seed42 \
        --output kb_baselines/autoschemakg/output/kbs/hotpotqa_dev_n5_seed42_kb.json \
        --stages 1,2,3
"""

import argparse
import json
import os
from pathlib import Path


def parse_stage_with_hrt(raw_list):
    """Yield (h, r, t) tuples from a list of {Head, Relation, Tail} dicts (stages 1 and 3).

    Tolerates capitalization variants.
    """
    if not isinstance(raw_list, list):
        return
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        h = item.get("Head") or item.get("head") or item.get("subject")
        r = item.get("Relation") or item.get("relation") or item.get("predicate")
        t = item.get("Tail") or item.get("tail") or item.get("object")
        if h is None or r is None or t is None:
            continue
        h, r, t = str(h).strip(), str(r).strip(), str(t).strip()
        if not h or not r or not t:
            continue
        yield (h, r, t)


def parse_stage2_event_entity(raw_list, stage2_relation="involved_in"):
    """Yield (entity, '<rel>', event_sentence) tuples from stage 2 records.

    Each {Event: str, Entity: [str, ...]} fans out into one triplet per entity.
    Entity-as-head keeps FAISS retrieval keys short (entity + relation) and only
    inflates the value side (event sentence) which is what the LLM consumes.
    """
    if not isinstance(raw_list, list):
        return
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        event = item.get("Event") or item.get("event")
        entities = item.get("Entity") or item.get("entity") or item.get("Entities") or item.get("entities")
        if event is None or not entities:
            continue
        event_str = str(event).strip()
        if not event_str:
            continue
        if isinstance(entities, str):
            entities = [entities]
        for ent in entities:
            ent_str = str(ent).strip()
            if not ent_str:
                continue
            yield (ent_str, stage2_relation, event_str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-output-dir", required=True,
                   help="Directory containing kg_extraction/*.json from atlas-rag")
    p.add_argument("--output", required=True, help="Path to write KBevo unified KB JSON")
    p.add_argument("--stages", default="1,2,3",
                   help="Which atlas-rag stages to flatten into triplets. Comma-separated subset of {1,2,3}.")
    p.add_argument("--stage2-relation", default="involved_in",
                   help="Synthetic relation phrase to use when flattening Stage 2 event-entity links")
    args = p.parse_args()

    stages = set(s.strip() for s in args.stages.split(","))
    valid_stages = {"1", "2", "3"}
    bad = stages - valid_stages
    if bad:
        raise ValueError(f"--stages contains invalid values: {bad}. Allowed: {valid_stages}")

    kg_dir = Path(args.atlas_output_dir) / "kg_extraction"
    if not kg_dir.exists():
        raise FileNotFoundError(f"{kg_dir} not found")

    files = sorted(kg_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No JSON files in {kg_dir}")
    print(f"Reading {len(files)} atlas-rag output files from {kg_dir}")
    print(f"Including stages: {sorted(stages)}")

    triplets_set = set()
    per_stage_counts = {"1": 0, "2": 0, "3": 0}
    per_qid_counts = {}
    n_records = 0
    n_records_with_any = 0

    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_records += 1

                rec_triplets = []
                if "1" in stages:
                    s1 = list(parse_stage_with_hrt(rec.get("entity_relation_dict", [])))
                    per_stage_counts["1"] += len(s1)
                    rec_triplets.extend(s1)
                if "2" in stages:
                    # atlas-rag key renamed across versions: older used
                    # 'event_entity_relation_dict', newer uses 'event_entity_dict'.
                    # Accept either (older alias as fallback).
                    stage2_raw = rec.get("event_entity_dict")
                    if stage2_raw is None:
                        stage2_raw = rec.get("event_entity_relation_dict", [])
                    s2 = list(parse_stage2_event_entity(
                        stage2_raw,
                        stage2_relation=args.stage2_relation,
                    ))
                    per_stage_counts["2"] += len(s2)
                    rec_triplets.extend(s2)
                if "3" in stages:
                    s3 = list(parse_stage_with_hrt(rec.get("event_relation_dict", [])))
                    per_stage_counts["3"] += len(s3)
                    rec_triplets.extend(s3)

                if rec_triplets:
                    n_records_with_any += 1
                for t in rec_triplets:
                    triplets_set.add(t)

                rid = rec.get("id", "")
                # Strip ONLY the trailing "__{para_idx}" — MuSiQue qids natively
                # contain "__" (e.g. "2hop__44815_494136"), so split() collapses them.
                qid = rid.rsplit("__", 1)[0] if "__" in rid else rid
                per_qid_counts[qid] = per_qid_counts.get(qid, 0) + len(rec_triplets)

    entities = set()
    relationships = set()
    return_values = set()
    for h, r, t in triplets_set:
        entities.add(h)
        relationships.add(r)
        return_values.add(t)

    kb = {
        "entities": sorted(entities),
        "relationships": sorted(relationships),
        "return_values": sorted(return_values),
        "triplets": [list(t) for t in sorted(triplets_set)],
        "_meta": {
            "source": "atlas-rag",
            "atlas_output_dir": str(Path(args.atlas_output_dir).resolve()),
            "atlas_files": [f.name for f in files],
            "stages_included": sorted(stages),
            "stage2_relation": args.stage2_relation if "2" in stages else None,
            "per_stage_raw_counts": per_stage_counts,
            "n_atlas_records": n_records,
            "n_records_with_any_triplet": n_records_with_any,
            "n_unique_triplets": len(triplets_set),
            "n_unique_entities": len(entities),
            "n_unique_relations": len(relationships),
            "n_questions_covered": len(per_qid_counts),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    print(f"\n[atlas->KBevo] Saved unified KB JSON: {args.output}")
    print(f"  records:                {n_records}")
    print(f"  records w/ ≥1 triplet:  {n_records_with_any} ({n_records_with_any/max(n_records,1)*100:.1f}%)")
    print(f"  per-stage raw counts:")
    for k in ("1", "2", "3"):
        marker = " (included)" if k in stages else " (skipped)"
        print(f"    stage {k}: {per_stage_counts[k]:>6d}{marker}")
    print(f"  unique triplets:        {len(triplets_set)}")
    print(f"  unique entities:        {len(entities)}")
    print(f"  unique relations:       {len(relationships)}")
    print(f"  questions covered:      {len(per_qid_counts)}")
    if per_qid_counts:
        avg = sum(per_qid_counts.values()) / len(per_qid_counts)
        print(f"  avg trip/qid:           {avg:.1f}")


if __name__ == "__main__":
    main()
