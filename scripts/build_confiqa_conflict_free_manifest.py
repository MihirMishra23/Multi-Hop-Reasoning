#!/usr/bin/env python3
"""Build deterministic conflict-free ConFiQA condition manifests.

The optimization variables choose the original or counterfactual path for each
of the fixed 1,000 seed-42 rows.  Feasibility requires every direct
``(head, relation)`` key to have one distinct tail at most.  SciPy/HiGHS proves
the maximum cardinality; a second feasibility pass selects the unique
lexicographically earliest bit vector (prefer CF at earlier evaluation
positions) for each requested cardinality.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack

DATASET_SHA256 = "dbb76f361831b754b87219344d80a386eaf83078ffe5888272dbe9d8e6c0eede"
SEED = 42
UNIVERSE_SIZE = 1000
ALGORITHM_VERSION = "confiqa-forward-key-milp-lex-v1"

Triplet = Tuple[str, str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_hash(values: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def triplet_hash(triplets: Sequence[Triplet]) -> str:
    payload = "\n".join(
        json.dumps(list(triplet), ensure_ascii=False, separators=(",", ":"))
        for triplet in triplets
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_path(value: str) -> List[Triplet]:
    parsed = ast.literal_eval(value)
    return [tuple(str(part) for part in triplet) for triplet in parsed]


def deduplicate(triplets: Sequence[Triplet]) -> List[Triplet]:
    return list(dict.fromkeys(triplets))


def add_inverses(triplets: Sequence[Triplet]) -> List[Triplet]:
    direct = deduplicate(triplets)
    return deduplicate(direct + [(tail, relation, head) for head, relation, tail in direct])


def ambiguity_stats(triplets: Sequence[Triplet]) -> Dict[str, int]:
    values: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for head, relation, tail in triplets:
        values[(head, relation)].add(tail)
    sizes = [len(tails) for tails in values.values() if len(tails) > 1]
    return {
        "key_count": len(values),
        "ambiguous_key_count": len(sizes),
        "excess_tail_count": sum(size - 1 for size in sizes),
        "max_distinct_tails": max(sizes, default=1 if values else 0),
    }


def build_constraint_matrix(
    variants: Sequence[Tuple[Sequence[Triplet], Sequence[Triplet]]]
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, int]:
    occurrences: Dict[Tuple[str, str], Dict[str, set[Tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row_index, pair in enumerate(variants):
        for choice, triplets in enumerate(pair):
            for head, relation, tail in triplets:
                occurrences[(head, relation)][tail].add((row_index, choice))

    forbidden = set()
    fixed = set()
    for tails in occurrences.values():
        groups = list(tails.values())
        for left_index in range(len(groups)):
            for right_index in range(left_index + 1, len(groups)):
                for left_literal in groups[left_index]:
                    for right_literal in groups[right_index]:
                        i, a = left_literal
                        j, b = right_literal
                        if i == j:
                            if a == b:
                                fixed.add((i, 1 - a))
                            continue
                        if i > j:
                            i, a, j, b = j, b, i, a
                        forbidden.add((i, a, j, b))

    rows: List[int] = []
    columns: List[int] = []
    values: List[float] = []
    lower: List[float] = []
    upper: List[float] = []
    constraint_index = 0

    for variable, required_value in sorted(fixed):
        rows.append(constraint_index)
        columns.append(variable)
        values.append(1.0)
        lower.append(float(required_value))
        upper.append(float(required_value))
        constraint_index += 1

    for i, a, j, b in sorted(forbidden):
        rows.extend((constraint_index, constraint_index))
        columns.extend((i, j))
        if (a, b) == (1, 1):
            coefficients, lo, hi = (1.0, 1.0), -np.inf, 1.0
        elif (a, b) == (1, 0):
            coefficients, lo, hi = (1.0, -1.0), -np.inf, 0.0
        elif (a, b) == (0, 1):
            coefficients, lo, hi = (-1.0, 1.0), -np.inf, 0.0
        else:
            coefficients, lo, hi = (1.0, 1.0), 1.0, np.inf
        values.extend(coefficients)
        lower.append(lo)
        upper.append(hi)
        constraint_index += 1

    matrix = coo_matrix(
        (values, (rows, columns)), shape=(constraint_index, len(variants))
    ).tocsr()
    return matrix, np.asarray(lower), np.asarray(upper), len(forbidden)


def solve_maximum(matrix: csr_matrix, lower: np.ndarray, upper: np.ndarray) -> int:
    variable_count = matrix.shape[1]
    result = milp(
        c=-np.ones(variable_count),
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.fun is None or getattr(result, "mip_gap", 0.0) != 0.0:
        raise RuntimeError(f"HiGHS did not prove the maximum: {result.message}")
    return int(round(-result.fun))


def solve_lexicographic(
    matrix: csr_matrix, lower: np.ndarray, upper: np.ndarray, target: int
) -> np.ndarray:
    """Return the unique bit vector preferring CF at earlier ordered rows."""
    variable_count = matrix.shape[1]
    cardinality = csr_matrix(np.ones((1, variable_count)))
    target_matrix = vstack((matrix, cardinality), format="csr")
    target_lower = np.concatenate((lower, [float(target)]))
    target_upper = np.concatenate((upper, [float(target)]))
    fixed_lower = np.zeros(variable_count)
    fixed_upper = np.ones(variable_count)
    selected = 0

    for index in range(variable_count):
        remaining = variable_count - index
        needed = target - selected
        if needed == 0:
            fixed_upper[index:] = 0.0
            break
        if needed == remaining:
            fixed_lower[index:] = 1.0
            fixed_upper[index:] = 1.0
            break

        fixed_lower[index] = 1.0
        fixed_upper[index] = 1.0
        result = milp(
            c=np.zeros(variable_count),
            integrality=np.ones(variable_count),
            bounds=Bounds(fixed_lower, fixed_upper),
            constraints=LinearConstraint(target_matrix, target_lower, target_upper),
        )
        if result.success:
            selected += 1
        else:
            fixed_lower[index] = 0.0
            fixed_upper[index] = 0.0

    selection = fixed_lower.astype(int)
    if int(selection.sum()) != target:
        raise RuntimeError(f"Expected {target} selected rows, found {selection.sum()}")
    return selection


def condition_record(
    label: str,
    target: int,
    selection: np.ndarray,
    ordered_ids: Sequence[int],
    variants: Sequence[Tuple[Sequence[Triplet], Sequence[Triplet]]],
) -> Dict[str, Any]:
    selected_ids = [ordered_ids[index] for index, value in enumerate(selection) if value]
    direct_triplets = [
        triplet
        for index, pair in enumerate(variants)
        for triplet in pair[int(selection[index])]
    ]
    unique_direct = deduplicate(direct_triplets)
    inverse_only = deduplicate(
        [(tail, relation, head) for head, relation, tail in unique_direct]
    )
    database_triplets = add_inverses(unique_direct)
    forward_stats = ambiguity_stats(unique_direct)
    if forward_stats["ambiguous_key_count"]:
        raise RuntimeError(f"{label} is not forward-key conflict-free: {forward_stats}")
    return {
        "label": label,
        "target_counterfactual_count": target,
        "actual_counterfactual_count": len(selected_ids),
        "selected_cf_source_ids_sha256": ordered_hash(selected_ids),
        "selected_cf_source_ids": selected_ids,
        "triplets": {
            "ordered_direct_count": len(direct_triplets),
            "ordered_direct_sha256": triplet_hash(direct_triplets),
            "unique_direct_count": len(unique_direct),
            "unique_direct_sha256": triplet_hash(unique_direct),
            "database_with_inverses_count": len(database_triplets),
            "database_with_inverses_sha256": triplet_hash(database_triplets),
        },
        "ambiguity": {
            "forward_direct": forward_stats,
            "inverse_derived_only": ambiguity_stats(inverse_only),
            "actual_database_with_inverses": ambiguity_stats(database_triplets),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actual_sha256 = sha256_file(args.data)
    if actual_sha256 != DATASET_SHA256:
        raise ValueError(f"Unexpected ConFiQA SHA-256: {actual_sha256}")
    rows = json.loads(args.data.read_text(encoding="utf-8"))
    ordered_ids = np.random.default_rng(SEED).permutation(len(rows)).tolist()[:UNIVERSE_SIZE]
    variants = [
        (
            parse_path(rows[source_id]["orig_path_labeled"]),
            parse_path(rows[source_id]["cf_path_labeled"]),
        )
        for source_id in ordered_ids
    ]
    matrix, lower, upper, forbidden_count = build_constraint_matrix(variants)
    maximum = solve_maximum(matrix, lower, upper)
    if maximum != 356:
        raise RuntimeError(f"Expected the pinned forward-key maximum 356, proved {maximum}")

    selections = {
        100: solve_lexicographic(matrix, lower, upper, 100),
        maximum: solve_lexicographic(matrix, lower, upper, maximum),
    }
    manifest = {
        "schema_version": 1,
        "dataset": {
            "name": "ConFiQA-MR",
            "sha256": DATASET_SHA256,
            "source_row_count": len(rows),
        },
        "selection_universe": {
            "seed": SEED,
            "shuffle_algorithm": "numpy.random.Generator(PCG64).permutation",
            "retained_count": UNIVERSE_SIZE,
            "ordered_source_ids_sha256": ordered_hash(ordered_ids),
            "ordered_source_ids": ordered_ids,
        },
        "smoke_query_selection": {
            "count": 50,
            "ordered_source_ids_sha256": ordered_hash(ordered_ids[:50]),
            "ordered_source_ids": ordered_ids[:50],
            "note": "same ordered questions for both conditions and both methods",
        },
        "selection_algorithm": {
            "name": "forward-key binary MILP with lexicographic feasibility fixing",
            "version": ALGORITHM_VERSION,
            "objective": "maximum CF count; prefer CF at earlier retained positions",
            "solver": "scipy.optimize.milp (HiGHS)",
            "scipy_version_when_generated": scipy_version,
            "binary_variables": len(ordered_ids),
            "forbidden_literal_pairs": forbidden_count,
            "maximum_counterfactual_count": maximum,
            "optimality_gap": 0.0,
        },
        "conditions": {
            "cf_100_conflict_free": condition_record(
                "CF-100-conflict-free", 100, selections[100], ordered_ids, variants
            ),
            "cf_356_conflict_free": condition_record(
                "CF-356-conflict-free", maximum, selections[maximum], ordered_ids, variants
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "maximum": maximum}, indent=2))


if __name__ == "__main__":
    main()
