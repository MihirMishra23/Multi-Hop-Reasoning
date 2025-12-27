import json
import os
import re
from typing import Dict, List, Optional

from tqdm import tqdm


def parse_db_lookups(
    file_path: str,
    pattern: Optional[str] = None,
    verbose: bool = False,
    show_progress: bool = False,
) -> Dict[str, List[str]]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    pattern = pattern or r"\s?\[dblookup\(([^,]+),\s*([^,]+)\)\s*->\s*(.*?)\]"

    results = {"entities": [], "relationships": [], "return_values": [], "triplets": []}

    iterator = tqdm(data, desc="Parsing annotations") if show_progress else data
    for text in iterator:
        matches = re.findall(pattern, text)
        for match in matches:
            entity = match[0]
            relationship = match[1]
            value = match[2]
            results["entities"].append(entity)
            results["relationships"].append(relationship)
            results["return_values"].append(value)
            results["triplets"].append([entity, relationship, value])

            if verbose:
                print(f"Entity: {entity}")
                print(f"Relationship: {relationship}")
                print(f"Value: {value}")
                print("-" * 40)

    return results


def save_database(results: Dict[str, List[str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)
