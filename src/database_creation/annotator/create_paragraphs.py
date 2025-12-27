import json
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset


def load_qa_json(input_file: str) -> List[Dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_paragraphs(qa_data: Iterable[Dict[str, Any]]) -> List[List[str]]:
    paragraphs_by_example: List[List[str]] = []
    for example in qa_data:
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
    flatten: bool = True,
) -> Dict[str, List[str]]:
    paragraphs_by_example = build_paragraphs(qa_data)
    dataset = Dataset.from_list([{"paragraphs": p} for p in paragraphs_by_example])
    dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    if flatten:
        paragraphs = [p for example in dataset for p in example["paragraphs"]]
    else:
        paragraphs = [example["paragraphs"] for example in dataset]

    return {"paragraphs": paragraphs}


def save_paragraphs(paragraphs: Dict[str, List[str]], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(paragraphs, f, indent=2, ensure_ascii=False)
