"""
KG Case Study Visualization for Paper (v2)
============================================
Usage:
    python kg_case_study.py --csv path/to/final_v2.2_375_with_gt.csv
    python kg_case_study.py --csv data.csv --pick 2 --output fig5.png
    python kg_case_study.py --csv data.csv --list-only

Dependencies:
    pip install networkx matplotlib numpy
"""

import ast
import csv
import re
import sys
import argparse
import textwrap
from pathlib import Path

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ═══════════════════════════════════════════════
# 1. DATA PARSING
# ═══════════════════════════════════════════════

def parse_triplets_from_cell(cell_text):
    if not cell_text or str(cell_text).strip() in ('', 'nan', 'None'):
        return []
    try:
        result = ast.literal_eval(str(cell_text).strip())
        if isinstance(result, list):
            return [(str(h).strip(), str(r).strip(), str(t).strip()) for h, r, t in result]
    except:
        pass
    return []


def get_all_db_columns(header):
    db_cols = []
    for i, col in enumerate(header):
        if re.match(r'generated_db_\d+', col.strip()):
            db_cols.append((col.strip(), i))
    return db_cols


def load_cases(csv_path):
    cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        db_cols = get_all_db_columns(header)

        prompt_idx = header.index('phase2_prompt') if 'phase2_prompt' in header else None
        answer_idx = header.index('ground_truth_answer') if 'ground_truth_answer' in header else None

        if prompt_idx is None or answer_idx is None:
            print(f"ERROR: Missing columns. Available: {header}")
            sys.exit(1)

        print(f"Found {len(db_cols)} DB columns: {[c[0] for c in db_cols]}")

        for row_idx, row in enumerate(reader):
            if len(row) <= max(prompt_idx, answer_idx):
                continue
            question = row[prompt_idx].strip()
            answer = row[answer_idx].strip()

            for db_name, db_idx in db_cols:
                if db_idx >= len(row):
                    continue
                triplets = parse_triplets_from_cell(row[db_idx])
                if triplets:
                    cases.append({
                        'row': row_idx,
                        'db_col': db_name,
                        'question': question,
                        'answer': answer,
                        'triplets': triplets,
                    })
    return cases


# ═══════════════════════════════════════════════
# 2. CASE SELECTION
# ═══════════════════════════════════════════════

def find_answer_nodes(G, answer_text):
    answer_lower = answer_text.lower().strip()
    matches = []
    for node in G.nodes():
        node_lower = node.lower().strip()
        if answer_lower == node_lower:
            matches.append((node, 1.0))
        elif answer_lower in node_lower or node_lower in answer_lower:
            matches.append((node, 0.8))
    matches.sort(key=lambda x: -x[1])
    return matches


def find_question_entities(G, question_text):
    q_lower = question_text.lower()
    matches = []
    for node in G.nodes():
        if len(node) < 3:
            continue
        if node.lower() in q_lower:
            matches.append(node)
    matches.sort(key=lambda x: -len(x))
    return matches


def score_case(case):
    triplets = case['triplets']
    G = nx.DiGraph()
    for h, r, t in triplets:
        G.add_edge(h, t, relation=r)

    n_nodes = G.number_of_nodes()
    if n_nodes < 5 or n_nodes > 35:
        return -1, {}

    answer_matches = find_answer_nodes(G, case['answer'])
    if not answer_matches:
        return -1, {}
    answer_node = answer_matches[0][0]

    q_entities = find_question_entities(G, case['question'])
    if not q_entities:
        return -1, {}

    best_path = None
    best_q_entity = None
    for qe in q_entities:
        try:
            # DIRECTED path only — every hop follows edge direction
            path = nx.shortest_path(G, qe, answer_node)
            if best_path is None or (2 <= len(path) <= 6 and len(path) > len(best_path or [])):
                best_path = path
                best_q_entity = qe
        except nx.NetworkXNoPath:
            continue

    if best_path is None or len(best_path) < 2:
        return -1, {}

    hops = len(best_path) - 1
    score = 0
    # Strongly prefer 4-hop
    if hops == 4:
        score += 50
    elif hops == 3:
        score += 30
    elif hops == 2:
        score += 15
    elif hops == 1:
        score += 2
    if 8 <= n_nodes <= 18:
        score += 20
    elif 5 <= n_nodes <= 25:
        score += 10
    # Prefer shorter, simpler questions (big bonus)
    q_len = len(case['question'])
    if q_len < 80:
        score += 25
    elif q_len < 120:
        score += 15
    elif q_len < 200:
        score += 5
    # Prefer short answers
    if len(case['answer']) < 30:
        score += 15
    elif len(case['answer']) < 50:
        score += 8
    if answer_matches[0][1] == 1.0:
        score += 15

    info = {
        'G': G, 'answer_node': answer_node, 'q_entity': best_q_entity,
        'path': best_path, 'hops': hops, 'n_nodes': n_nodes,
        'n_edges': G.number_of_edges(),
    }
    return score, info


def select_best_case(cases):
    print(f"Scoring {len(cases)} cases...")
    scored = []
    for case in cases:
        score, info = score_case(case)
        if score > 0:
            scored.append((score, case, info))
    scored.sort(key=lambda x: -x[0])
    print(f"Found {len(scored)} viable cases.\n")

    print("=" * 70)
    print("TOP 10 CANDIDATES (use --pick N to select, default 0)")
    print("=" * 70)
    for i, (score, case, info) in enumerate(scored[:10]):
        print(f"\n[{i}] score={score} | {info['hops']}-hop | "
              f"{info['n_nodes']} nodes, {info['n_edges']} edges")
        print(f"    Q: {case['question'][:120]}")
        print(f"    A: {case['answer'][:80]}")
        # Show path with edge relations
        G = info['G']
        path = info['path']
        path_str = path[0]
        for j in range(len(path) - 1):
            rel = G.edges[path[j], path[j+1]].get('relation', '?')
            path_str += f' --({rel})--> {path[j+1]}'
        print(f"    Path: {path_str}")
        print(f"    Row: {case['row']}, Col: {case['db_col']}")
    return scored


# ═══════════════════════════════════════════════
# 3. VISUALIZATION
# ═══════════════════════════════════════════════

def wrap_label(text, width=15):
    if len(text) <= width:
        return text
    return '\n'.join(textwrap.wrap(text, width=width))


def prune_graph(G, path, max_dist=1):
    """Keep only nodes within max_dist hops of the reasoning path."""
    U = G.to_undirected()
    keep = set()
    for p_node in path:
        keep.add(p_node)
        if max_dist >= 1:
            for n1 in U.neighbors(p_node):
                keep.add(n1)
                if max_dist >= 2:
                    for n2 in U.neighbors(n1):
                        keep.add(n2)
    return G.subgraph(keep).copy()


def draw_case_study(case, info, output_path='kg_case_study.png', prune_dist=1):
    G_full = info['G']
    path = info['path']
    answer_node = info['answer_node']
    q_entity = info['q_entity']

    G = prune_graph(G_full, path, max_dist=prune_dist)
    path_set = set(path)

    print(f"  Full: {G_full.number_of_nodes()} nodes -> Pruned: {G.number_of_nodes()} nodes")

    # ── Colors ──
    C = {
        'bg':        '#FFFFFF',
        'q_fill':    '#C0392B',
        'a_fill':    '#27AE60',
        'path_fill': '#2980B9',
        'path_edge': '#2980B9',
        'ctx_fill':  '#D5D8DC',
        'ctx_edge':  '#E5E7E9',
        'text':      '#2C3E50',
        'edge_lbl':  '#5D6D7E',
        'step':      '#E67E22',
    }

    # ── Layout ──
    try:
        pos = nx.kamada_kawai_layout(G, scale=3.0)
    except:
        pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)

    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=300)
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['bg'])
    ax.axis('off')

    # ── Edge classification ──
    path_edges = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            path_edges.add((u, v))
        elif G.has_edge(v, u):
            path_edges.add((v, u))

    other_edges = [(u, v) for u, v in G.edges() if (u, v) not in path_edges]
    reasoning_edges = [(u, v) for u, v in G.edges() if (u, v) in path_edges]

    # ── Context edges ──
    nx.draw_networkx_edges(
        G, pos, edgelist=other_edges, ax=ax,
        edge_color=C['ctx_edge'], width=1.0, alpha=0.55,
        arrows=True, arrowsize=10, arrowstyle='-|>',
        connectionstyle='arc3,rad=0.08',
        min_source_margin=15, min_target_margin=15,
    )

    # ── Reasoning edges ──
    nx.draw_networkx_edges(
        G, pos, edgelist=reasoning_edges, ax=ax,
        edge_color=C['path_edge'], width=3.0, alpha=0.9,
        arrows=True, arrowsize=16, arrowstyle='-|>',
        connectionstyle='arc3,rad=0.08',
        min_source_margin=15, min_target_margin=15,
    )

    # ── Nodes ──
    for node in G.nodes():
        if node == q_entity:
            color, size, lw, ec = C['q_fill'], 380, 1.8, '#922B21'
        elif node == answer_node:
            color, size, lw, ec = C['a_fill'], 380, 1.8, '#1E8449'
        elif node in path_set:
            color, size, lw, ec = C['path_fill'], 280, 1.4, '#1F618D'
        else:
            color, size, lw, ec = C['ctx_fill'], 180, 0.6, '#ABB2B9'

        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], ax=ax,
            node_color=color, node_size=size,
            edgecolors=ec, linewidths=lw,
            node_shape='o', alpha=0.93,
        )

    # ── Node labels ──
    for node in G.nodes():
        x, y = pos[node]
        label = wrap_label(node, width=14)
        is_key = node in path_set

        ax.text(
            x, y - 0.20, label,
            fontsize=10.5 if is_key else 8.5,
            fontweight='bold' if is_key else 'normal',
            color=C['text'], ha='center', va='top',
            fontfamily='serif',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                      edgecolor='none', alpha=0.88),
        )

    # ── Step numbers on path ──
    circled = "①②③④⑤⑥⑦⑧⑨⑩"
    for i, node in enumerate(path):
        x, y = pos[node]
        step = circled[i] if i < len(circled) else str(i + 1)
        ax.text(
            x + 0.13, y + 0.17, step,
            fontsize=13, fontweight='bold',
            color='#2C3E50', ha='center', va='center',
            fontfamily='sans-serif',
        )

    # ── Edge labels ──
    pe_labels = {}
    oe_labels = {}
    for u, v, data in G.edges(data=True):
        rel = data.get('relation', '')
        if len(rel) > 25:
            rel = rel[:23] + '...'
        if (u, v) in path_edges:
            pe_labels[(u, v)] = rel
        else:
            oe_labels[(u, v)] = rel

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=pe_labels, ax=ax,
        font_size=9.5, font_color=C['path_edge'],
        font_weight='bold', font_family='serif',
        bbox=dict(boxstyle='round,pad=0.08', facecolor='#EBF5FB',
                  edgecolor='none', alpha=0.95),
        rotate=True, label_pos=0.45,
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=oe_labels, ax=ax,
        font_size=8.0, font_color=C['edge_lbl'],
        font_family='serif',
        bbox=dict(boxstyle='round,pad=0.06', facecolor='white',
                  edgecolor='none', alpha=0.85),
        rotate=True, label_pos=0.45,
    )

    # ── Title ──
    q_disp = case['question'][:160] + ('...' if len(case['question']) > 160 else '')
    ax.set_title(
        f"Q: {q_disp}\nA: {case['answer']}",
        fontsize=10, fontfamily='serif', color=C['text'],
        loc='left', pad=18,
    )

    plt.tight_layout(pad=2.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=C['bg'], edgecolor='none')
    plt.close()
    print(f"\n✅  Saved: {output_path}  ({Path(output_path).stat().st_size / 1024:.0f} KB)")


# ═══════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='KG Case Study Visualization for Paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python kg_case_study.py --csv data.csv --list-only        # browse candidates
          python kg_case_study.py --csv data.csv --pick 0           # draw best candidate
          python kg_case_study.py --csv data.csv --pick 2 --prune 2 # show more context
        """))
    parser.add_argument('--csv', required=True)
    parser.add_argument('--pick', type=int, default=0)
    parser.add_argument('--output', default='kg_case_study.png')
    parser.add_argument('--prune', type=int, default=1,
                        help='Keep nodes within N hops of path (default 1)')
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()

    print("Loading data...")
    cases = load_cases(args.csv)
    print(f"Loaded {len(cases)} entries.\n")

    scored = select_best_case(cases)
    if args.list_only:
        return
    if not scored:
        sys.exit(1)

    pick = min(args.pick, len(scored) - 1)
    score, case, info = scored[pick]

    print(f"\nDrawing candidate [{pick}]...")
    draw_case_study(case, info, output_path=args.output, prune_dist=args.prune)
    print(f"\nLaTeX: \\includegraphics[width=\\textwidth]{{{args.output}}}")


if __name__ == '__main__':
    main()