"""
DB Structure Analysis — aligned with Wikontic (EACL 2026) metrics.
Usage:
    python analysis/structure_analysis.py
    python analysis/structure_analysis.py --csv path/to/csv
"""

import argparse
import ast
import csv
import os
from collections import Counter, defaultdict

try:
    import networkx as nx
except ImportError:
    print("Please install networkx: pip install networkx")
    exit(1)


CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "KG_results", "ckpt500_hotpotqa_dev_n1000_all_concat_trainparams.csv")


def get_db_columns(header):
    db_cols = [c for c in header if c.startswith("generated_db_")]
    db_cols.sort(key=lambda c: int(c.split("_")[-1]))
    return db_cols


def parse_db_column(raw):
    try:
        triplets = ast.literal_eval(raw)
        return [(str(e).strip(), str(r).strip(), str(v).strip()) for e, r, v in triplets]
    except Exception:
        return []


def analyze_triplets(all_triplets, label=""):
    """
    Compute structure statistics directly from triplets.
    Metrics aligned with Wikontic (EACL 2026) Table 2.
    """
    if not all_triplets:
        return None

    # ── |E|: unique entities (both subject and object positions) ──
    entities = set()
    for e, r, v in all_triplets:
        entities.add(e)
        entities.add(v)
    n_entities = len(entities)

    # ── |R|: unique relations ──
    relations = set(r for _, r, _ in all_triplets)
    n_relations = len(relations)

    # ── Total triplets (includes duplicates if any) ──
    n_triplets = len(all_triplets)
    unique_triplets = len(set(all_triplets))

    # ── Avg entity degree ──
    # Degree = how many triplets an entity appears in (as subject OR object)
    # Use MultiGraph to count all edges including duplicates
    degree_count = Counter()
    for e, r, v in all_triplets:
        degree_count[e] += 1
        degree_count[v] += 1
    avg_degree = sum(degree_count.values()) / len(degree_count) if degree_count else 0

    # ── Unique entities per relation ──
    # For each relation, count how many unique entities appear with it
    entities_per_relation = defaultdict(set)
    for e, r, v in all_triplets:
        entities_per_relation[r].add(e)
        entities_per_relation[r].add(v)
    avg_unique_e_per_r = (
        sum(len(ents) for ents in entities_per_relation.values()) / len(entities_per_relation)
        if entities_per_relation else 0
    )

    # ── Relation diversity per entity pair ──
    # For each (entity, value) pair, how many different relations connect them?
    pair_relations = defaultdict(set)
    for e, r, v in all_triplets:
        pair = tuple(sorted([e, v]))  # undirected
        pair_relations[pair].add(r)
    avg_r_diversity = (
        sum(len(rels) for rels in pair_relations.values()) / len(pair_relations)
        if pair_relations else 0
    )

    # ── Connectivity (using networkx) ──
    G = nx.Graph()
    for e, r, v in all_triplets:
        G.add_edge(e, v)
    components = list(nx.connected_components(G))
    n_components = len(components)
    largest = max(len(c) for c in components) if components else 0
    giant_pct = largest / n_entities * 100 if n_entities > 0 else 0
    isolated = sum(1 for _, d in G.degree() if d == 0)

    return {
        "label": label,
        "n_entities": n_entities,
        "n_relations": n_relations,
        "n_triplets": n_triplets,
        "unique_triplets": unique_triplets,
        "duplicate_triplets": n_triplets - unique_triplets,
        "avg_degree": avg_degree,
        "avg_unique_e_per_r": avg_unique_e_per_r,
        "avg_r_diversity": avg_r_diversity,
        "n_components": n_components,
        "giant_component_nodes": largest,
        "giant_component_pct": giant_pct,
        "isolated_nodes": isolated,
    }


def print_stats(stats):
    print(f"\n{'='*60}")
    print(f"  {stats['label']}")
    print(f"{'='*60}")
    print(f"  |E| (unique entities):       {stats['n_entities']:,}")
    print(f"  |R| (unique relations):       {stats['n_relations']:,}")
    print(f"  Total triplets:               {stats['n_triplets']:,}")
    print(f"  Unique triplets:              {stats['unique_triplets']:,}")
    print(f"  Duplicate triplets:           {stats['duplicate_triplets']:,}")
    print(f"  Avg entity degree:            {stats['avg_degree']:.2f}")
    print(f"  Unique entities per relation: {stats['avg_unique_e_per_r']:.2f}")
    print(f"  Relation diversity per pair:  {stats['avg_r_diversity']:.2f}")
    print(f"  Connected components:         {stats['n_components']:,}")
    print(f"  Giant component:              {stats['giant_component_nodes']:,} ({stats['giant_component_pct']:.1f}%)")
    print(f"  Isolated nodes:               {stats['isolated_nodes']}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="DB Structure Analysis")
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    args = parser.parse_args()

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        db_cols = get_db_columns(header)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows, {len(db_cols)} DB columns")

    # ── Per-sample stats: compute for each unique context, then average ──
    per_sample_stats = []
    seen_contexts = set()

    for row in rows:
        ctx = row.get("phase1_context_0", "").strip()
        if ctx in seen_contexts:
            continue
        seen_contexts.add(ctx)

        for db_col in db_cols:
            raw = row.get(db_col, "")
            if not raw.strip():
                continue
            triplets = parse_db_column(raw)
            if triplets:
                stats = analyze_triplets(triplets)
                if stats:
                    per_sample_stats.append(stats)
                break

    print(f"Unique contexts: {len(seen_contexts)}")
    print(f"Samples with valid DB: {len(per_sample_stats)}")

    if per_sample_stats:
        avg = lambda key: sum(s[key] for s in per_sample_stats) / len(per_sample_stats)

        print(f"\n{'='*60}")
        print(f"  Per-Sample Average (across {len(per_sample_stats)} samples)")
        print(f"{'='*60}")
        print(f"  Avg |E| (unique entities):       {avg('n_entities'):.1f}")
        print(f"  Avg |R| (unique relations):       {avg('n_relations'):.1f}")
        print(f"  Avg total triplets:               {avg('n_triplets'):.1f}")
        print(f"  Avg duplicate triplets:           {avg('duplicate_triplets'):.2f}")
        print(f"  Avg entity degree:                {avg('avg_degree'):.2f}")
        print(f"  Avg unique entities per relation: {avg('avg_unique_e_per_r'):.2f}")
        print(f"  Avg relation diversity per pair:  {avg('avg_r_diversity'):.2f}")
        print(f"  Avg connected components:         {avg('n_components'):.1f}")
        print(f"  Avg giant component %%:            {avg('giant_component_pct'):.1f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()