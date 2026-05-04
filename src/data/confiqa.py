"""Loader for ConFiQA dataset with unified schema.

Exposes `load_confiqa(split, limit=None, seed=None, setting='orig')` returning an HF Dataset
with fields:

- id: str
- question: str
- answers: List[str]
- contexts: List[str]
- context_titles: List[str]
- supporting_facts: List[Dict[str, Any]]  (empty for ConFiQA)
- golden_contexts: List[str]
- golden_triplets: List[tuple]  (triplets based on setting)

Notes:
- ConFiQA is loaded from a local JSON file
- setting='orig' uses orig_context, orig_answer, orig_triplets for all examples
- setting='cf' uses cf_context, cf_answer, cf_triplets for all examples
- setting='cf_100' uses cf for first 100 examples, orig for rest
- setting='cf_500' uses cf for first 500 examples, orig for rest
"""

import ast
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset as HFDataset  # type: ignore

logger = logging.getLogger(__name__)

# Path to the ConFiQA dataset
CONFIQA_PATH = "/share/j_sun/lmlm_multihop/confiqa/ConFiQA-MR.json"


def _parse_triplets(triplet_str: str) -> List[Tuple[str, str, str]]:
    """Parse triplet string into list of tuples.

    Args:
        triplet_str: String like "[('A', 'B', 'C'), ('D', 'E', 'F')]"

    Returns:
        List of tuples like [('A', 'B', 'C'), ('D', 'E', 'F')]
    """
    if not triplet_str:
        return []

    try:
        parsed = ast.literal_eval(triplet_str)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass

    return []


def _load_confiqa(path: str = CONFIQA_PATH) -> List[Dict[str, Any]]:
    """Load ConFiQA dataset from JSON file.

    Args:
        path: Path to ConFiQA-MC.json

    Returns:
        List of examples
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ConFiQA dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def _normalize_confiqa(data: List[Dict[str, Any]], setting: str = "orig") -> HFDataset:
    """Normalize ConFiQA data to unified schema.

    Args:
        data: Raw ConFiQA examples
        setting: 'orig', 'cf', 'cf_100', or 'cf_500' - determines which context/answer/triplets to use

    Returns:
        HFDataset with unified schema
    """
    examples = []

    logger.info(f"Normalizing ConFiQA with setting: {setting}")
    for idx, ex in enumerate(data):
        # Parse both orig and cf triplets
        triplets_orig = _parse_triplets(ex.get("orig_path_labeled", ""))
        triplets_cf = _parse_triplets(ex.get("cf_path_labeled", ""))

        # Determine whether to use cf or orig for this example
        use_cf = False
        if setting == "cf":
            use_cf = True
        elif setting == "cf_100" and idx < 100:
            use_cf = True
        elif setting == "cf_500" and idx < 500:
            use_cf = True

        # Select context, answer, and triplets based on use_cf
        if use_cf:
            context = ex.get("cf_context", "")
            answer = ex.get("cf_answer", "")
            aliases = ex.get("cf_alias", [])
            golden_triplets = triplets_cf
        else:
            context = ex.get("orig_context", "")
            answer = ex.get("orig_answer", "")
            aliases = ex.get("orig_alias", [])
            golden_triplets = triplets_orig

        # Build answer list (primary answer + aliases)
        answers = [answer] if answer else []
        if isinstance(aliases, list):
            answers.extend([str(a).strip() for a in aliases if str(a).strip()])

        # Remove duplicates while preserving order
        seen = set()
        unique_answers = []
        for ans in answers:
            if ans and ans not in seen:
                seen.add(ans)
                unique_answers.append(ans)

        examples.append({
            "id": str(idx),
            "question": str(ex.get("question", "")),
            "answers": unique_answers,
            "contexts": [context] if context else [],
            "context_titles": ["ConFiQA Context"],
            "golden_contexts": [context] if context else [],
            "supporting_facts": [],  # ConFiQA doesn't have sentence-level supporting facts
            "golden_triplets": golden_triplets,
        })

    return HFDataset.from_list(examples)


def load_confiqa(
    split: str = "test",
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    setting: str = "orig",
    confiqa_path: str = CONFIQA_PATH,
) -> HFDataset:
    """Load ConFiQA with unified schema.

    Args:
        split: Dataset split (ConFiQA only has test, so this is ignored)
        source: Data source (ignored, always loads from local file)
        limit: Optional max number of rows to return
        seed: Optional random seed for shuffling
        setting: 'orig', 'cf', 'cf_100', or 'cf_500' - which version of context/answers to use
        confiqa_path: Path to ConFiQA-MC.json

    Returns:
        HFDataset with unified schema
    """
    # Load raw data
    raw_data = _load_confiqa(confiqa_path)

    # Normalize to unified schema first
    ds = _normalize_confiqa(raw_data, setting=setting)

    # Shuffle if seed provided
    if seed is not None:
        ds = ds.shuffle(seed=seed)

    # Limit if requested
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # Debug logging
    logger.info(f"DEBUG load_confiqa: Loading {len(ds)} examples with setting='{setting}'")
    if len(ds) > 0:
        logger.info(f"DEBUG load_confiqa: First example answer: {ds[0]['answers']}")

    return ds
