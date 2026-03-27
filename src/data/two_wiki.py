from data.hotpotqa import _normalize_examples_pylist, _normalize_hf_dataset, _normalize_split
from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore
from typing import Optional
import json

TWO_WIKI_PATH = "/share/j_sun/rtn27/datasets/two_wiki_dev.json"
def load_2wiki(
    setting: str,
    split: str,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> HFDataset:
    """Load HotpotQA with unified schema.

    Args:
        setting: "distractor" or "fullwiki".
        split: "train", "dev"/"validation", or "test" (where available).
        source: "auto" (prefer local), "local", or "hf".
        limit: optional max number of rows to return.
        seed: optional random seed for shuffling. If provided, dataset will be shuffled deterministically.
    """
    split = _normalize_split(split)

    if split != 'validation':
        raise NotImplementedError("Please use the 'dev / validation' split for 2wiki.")

    with open(TWO_WIKI_PATH) as f:
        data = json.load(f)  # type: ignore

    
    ds = _normalize_2wiki_data(data)
    # Shuffle with seed if provided
    if seed is not None:
        ds = ds.shuffle(seed=seed)
        
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds



def _normalize_2wiki_data(data):
    rows = []
    for ex in data:
        ex_id = ex.get("_id") or ex.get("id") or ""
        question = ex.get("question") or ""
        answer = ex.get("answer")
        answers = []
        if isinstance(answer, str):
            answers = [answer]
        elif isinstance(answer, list):
            answers = [str(a) for a in answer]
        else:
            answers = ex.get("answers") or []
            answers = [str(a) for a in answers]
        contexts = _build_contexts(ex.get("context"))
        supporting_facts = _build_supporting_facts(ex.get("supporting_facts"))
        golden_contexts = _build_golden_contexts(ex.get("context"), ex.get("supporting_facts"))
        rows.append(
            {
                "id": str(ex_id),
                "question": str(question),
                "answers": answers,
                "contexts": contexts,
                "supporting_facts": supporting_facts,
                "golden_contexts": golden_contexts,
            }
        )
    return HFDataset.from_list(rows)


def _build_contexts(context: list[list]):
    """Build all context paragraphs from 2wiki format.

    Format: [[title, [sent1, sent2, ...]], [title, [sent1, ...]], ...]
    """
    if not context:
        return []
    result = []
    for sub_list in context:
        if len(sub_list) >= 2:
            title = sub_list[0]
            sentences = sub_list[1]
            # Format: "Title: sentence1 sentence2 ..."
            paragraph = f"{title}: " + " ".join(sentences).strip()
            result.append(paragraph)
    return result


def _build_supporting_facts(sf_field):
    """Build supporting facts from 2wiki format.

    Format: [[title, sent_id], [title, sent_id], ...]
    Returns: [{"title": str, "sentence_id": int}, ...]
    """
    if not sf_field:
        return []
    result = []
    for item in sf_field:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sent_id = item[0], item[1]
            result.append({"title": str(title), "sentence_id": int(sent_id)})
    return result


def _build_golden_contexts(context: list[list], supporting_facts):
    """Build golden context strings - only contexts whose title is in supporting_facts.

    Args:
        context: [[title, [sent1, ...]], ...] format
        supporting_facts: [[title, sent_id], ...] format
    Returns:
        List of context strings for supporting titles only
    """
    if not context or not supporting_facts:
        return []

    # Extract supporting titles from supporting_facts
    supporting_titles = set()
    for item in supporting_facts:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            supporting_titles.add(item[0])

    # Build contexts only for supporting titles
    result = []
    for sub_list in context:
        if len(sub_list) >= 2:
            title = sub_list[0]
            if title in supporting_titles:
                sentences = sub_list[1]
                paragraph = f"{title}: " + " ".join(sentences).strip()
                result.append(paragraph)
    return result