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
        rows.append(
            {
                "id": str(ex_id),
                "question": str(question),
                "answers": answers,
                "contexts": contexts,
                "supporting_facts" : "Empty supporting facts because Ryan is lazy and this actually useless.", #TODO: Actually implement this
            }
        )
    return HFDataset.from_list(rows)


def _build_contexts(context: list[list]):
    result = []
    for sub_list in context:
        #sub_list[0] is the title
        paragraph = sub_list[0] + " ".join(sub_list[1]).strip()
        result.append(paragraph)
    return result
    