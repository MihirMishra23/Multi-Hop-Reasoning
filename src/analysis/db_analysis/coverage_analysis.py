"""
Coverage Analysis — aligned with Wikontic (EACL 2026) Table 3.
Measures: answer coverage (full graph, 1-10 hop neighborhoods).

Usage:
    python analysis/coverage_analysis.py
    python analysis/coverage_analysis.py --csv path/to/csv_with_gt
"""

import argparse
import ast
import csv
import re
import os

try:
    import networkx as nx
except ImportError:
    print("Please install networkx: pip install networkx")
    exit(1)


CSV_PATH = os.path.join(os.path.dirname(__file__), "final_v2.2_375_with_gt.csv")

MAX_HOPS = 10


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


def extract_question(prompt):
    prompt = prompt.strip()
    if prompt.startswith("Question:"):
        prompt = prompt[len("Question:"):].strip()
    prompt = re.sub(r'\s*Answer:\s*$', '', prompt).strip()
    return prompt


def normalize(text):
    """Lowercase, remove punctuation, collapse whitespace — same as Wikontic."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def find_matching_nodes(target, graph_nodes):
    """
    Find graph nodes matching target via substring match (Wikontic method).
    Two entities match if one is a substring of the other.
    """
    target_norm = normalize(target)
    if not target_norm or len(target_norm) < 2:
        return []
    matched = []
    for node in graph_nodes:
        node_norm = normalize(node)
        if not node_norm:
            continue
        if target_norm in node_norm or node_norm in target_norm:
            matched.append(node)
    return matched


def get_k_hop_neighborhood(G, source_nodes, k):
    """Get all nodes within k hops of any source node."""
    neighborhood = set(source_nodes)
    for node in source_nodes:
        if node not in G:
            continue
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=k)
        neighborhood.update(lengths.keys())
    return neighborhood


def main():
    parser = argparse.ArgumentParser(description="Coverage Analysis")
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    args = parser.parse_args()

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        db_cols = get_db_columns(header)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows")

    # Deduplicate by context
    seen_contexts = set()
    unique_rows = []
    for row in rows:
        ctx = row.get("phase1_context_0", "").strip()
        if ctx in seen_contexts:
            continue
        seen_contexts.add(ctx)
        unique_rows.append(row)
    print(f"Unique contexts: {len(unique_rows)}")

    # Stats
    total = 0
    answer_in_graph = 0
    answer_in_k_hop = {k: 0 for k in range(1, MAX_HOPS + 1)}  # 1-hop to 10-hop
    question_entity_found = 0
    skipped = 0

    for row in unique_rows:
        gt_answer = row.get("ground_truth_answer", "").strip()
        prompt = row.get("phase2_prompt", "")
        question = extract_question(prompt)

        if not gt_answer or not question:
            skipped += 1
            continue

        # Take first non-empty DB
        triplets = []
        for db_col in db_cols:
            raw = row.get(db_col, "")
            if raw.strip():
                triplets = parse_db_column(raw)
                if triplets:
                    break

        if not triplets:
            skipped += 1
            continue

        total += 1

        # Build undirected graph (following Wikontic methodology)
        G = nx.Graph()
        all_entities = set()
        for e, r, v in triplets:
            G.add_edge(e, v)
            all_entities.add(e)
            all_entities.add(v)

        graph_nodes = list(G.nodes())

        # 1. Is the answer entity anywhere in the graph?
        answer_nodes = find_matching_nodes(gt_answer, graph_nodes)
        if answer_nodes:
            answer_in_graph += 1

        # 2. Find question entities in the graph
        q_entity_nodes = []
        for ent in all_entities:
            ent_norm = normalize(ent)
            q_norm = normalize(question)
            if len(ent_norm) >= 3 and ent_norm in q_norm:
                if ent in G:
                    q_entity_nodes.append(ent)

        if q_entity_nodes:
            question_entity_found += 1

        # 3. Check answer reachability at each k-hop (1 through 10)
        if q_entity_nodes and answer_nodes:
            # Compute 10-hop neighborhood once, then check each k
            # More efficient: get shortest path lengths from all q entities
            all_distances = {}
            for qe in q_entity_nodes:
                if qe not in G:
                    continue
                lengths = nx.single_source_shortest_path_length(G, qe, cutoff=MAX_HOPS)
                for node, dist in lengths.items():
                    if node not in all_distances or dist < all_distances[node]:
                        all_distances[node] = dist

            # For each k, check if any answer node is within k hops
            min_answer_dist = None
            for ans_node in answer_nodes:
                if ans_node in all_distances:
                    d = all_distances[ans_node]
                    if min_answer_dist is None or d < min_answer_dist:
                        min_answer_dist = d

            if min_answer_dist is not None:
                for k in range(1, MAX_HOPS + 1):
                    if min_answer_dist <= k:
                        answer_in_k_hop[k] += 1

    pct = lambda n, d: f"{n / d:.1%}" if d > 0 else "N/A"

    print(f"\n{'='*65}")
    print(f"  Coverage Analysis (Wikontic-style, undirected graph)")
    print(f"{'='*65}")
    print(f"  Questions evaluated:           {total}")
    print(f"  Skipped:                       {skipped}")
    print(f"  Question Entity Coverage:       {question_entity_found} / {total} ({pct(question_entity_found, total)})")
    print(f"\n  Contains Answer (full graph):   {answer_in_graph} / {total} ({pct(answer_in_graph, total)})")
    print(f"  {'─'*50}")
    for k in range(1, MAX_HOPS + 1):
        print(f"  Contains Answer ({k:2d}-hop):       {answer_in_k_hop[k]:>3d} / {total} ({pct(answer_in_k_hop[k], total)})")
    print(f"{'='*65}")
    print(f"  Note: Undirected graph, substring match (case-insensitive,")
    print(f"  punctuation removed), following Wikontic methodology.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()