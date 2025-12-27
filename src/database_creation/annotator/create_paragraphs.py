import json
import os
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset
from tqdm import tqdm


def load_qa_json(input_file: str) -> List[Dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_paragraphs(
    qa_data: Iterable[Dict[str, Any]],
    show_progress: bool = False,
) -> List[List[str]]:
    paragraphs_by_example: List[List[str]] = []
    iterator = tqdm(qa_data, desc="Preparing paragraphs") if show_progress else qa_data
    for example in iterator:
        context = example["context"]
        sentences = []
        for title, sentence_list in context:
            paragraph = f"{title}\n" + "".join(sentence_list)
            sentences.append(paragraph)
        paragraphs_by_example.append(sentences)
    return paragraphs_by_example


def prepare_paragraphs(
    qa_data: Iterable[Dict[str, Any]],
    seed: int = 42,
    limit: Optional[int] = None,
    show_progress: bool = False,
) -> Dict[str, List[str]]:
    paragraphs_by_example = build_paragraphs(qa_data, show_progress=show_progress)
    dataset = Dataset.from_list([{"paragraphs": p} for p in paragraphs_by_example])
    dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return {"paragraphs": [example["paragraphs"] for example in dataset]}


def save_paragraphs(paragraphs: Dict[str, List[str]], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(paragraphs, f, indent=2, ensure_ascii=False)
