import copy
import json
import os
from typing import Generator, Iterable, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def load_prompt_template(prompt_path: str) -> List[dict]:
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = json.load(f)
    return prompt + [{"role": "assistant", "content": ""}]


def normalize_paragraphs(paragraphs: Iterable) -> List[str]:
    paragraphs = list(paragraphs)
    if paragraphs and isinstance(paragraphs[0], list):
        return [p for group in paragraphs for p in group]
    return paragraphs


def to_chat_template(prompt_template: List[dict], user_content: str) -> List[dict]:
    prompt_template_copy = copy.deepcopy(prompt_template)
    prompt_template_copy[1]["content"] = user_content
    return prompt_template_copy


def iter_annotate_batches(
    paragraphs: Iterable,
    prompt_template: List[dict],
    model_path: str,
    batch_size: int = 8,
    device: str = "cuda",
    start_index: int = 0,
    max_new_tokens: int = 2048,
    initial_annotations: Optional[List[str]] = None,
) -> Generator[Tuple[int, List[str], List[str]], None, None]:
    paragraphs = normalize_paragraphs(paragraphs)
    annotated_data = list(initial_annotations) if initial_annotations else []
    nb_paragraphs = len(paragraphs)

    model = LlamaForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    for i in range(start_index, nb_paragraphs, batch_size):
        paragraphs_text = paragraphs[i : i + batch_size]
        chat_templates = [to_chat_template(prompt_template, p) for p in paragraphs_text]
        tokenized_prompts = tokenizer.apply_chat_template(
            chat_templates,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            continue_final_message=True,
            padding_side="right",
        ).to(device)
        input_lengths = tokenized_prompts.attention_mask.sum(dim=1).to(device)
        with torch.no_grad():
            generate_ids = model.generate(**tokenized_prompts, max_new_tokens=max_new_tokens)

        batch_results = []
        for j in range(generate_ids.shape[0]):
            res = tokenizer.decode(generate_ids[j, input_lengths[j] :], skip_special_tokens=True)
            batch_results.append(res)
            annotated_data.append(res)

        yield i, batch_results, annotated_data


def annotate_paragraphs(
    paragraphs: Iterable,
    prompt_template: List[dict],
    model_path: str,
    batch_size: int = 8,
    device: str = "cuda",
    start_index: int = 0,
    max_new_tokens: int = 2048,
    initial_annotations: Optional[List[str]] = None,
) -> List[str]:
    annotated = []
    for _, _, annotated_data in iter_annotate_batches(
        paragraphs=paragraphs,
        prompt_template=prompt_template,
        model_path=model_path,
        batch_size=batch_size,
        device=device,
        start_index=start_index,
        max_new_tokens=max_new_tokens,
        initial_annotations=initial_annotations,
    ):
        annotated = annotated_data
    return annotated


def save_annotations(annotations: List[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)
