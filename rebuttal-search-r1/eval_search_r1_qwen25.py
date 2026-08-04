#!/usr/bin/env python3
"""Batched ConFiQA evaluation for the public Qwen2.5 Search-R1 checkpoints.

The prompt, `<search>`/`<information>` loop, E5 retriever, top-k=3, and default
temperature=0.7 follow Search-R1's public `infer.py` at commit 598e61b.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import get_dataset, selected_rows_provenance  # noqa: E402
from eval.metrics import exact_match_score, f1_score  # noqa: E402

SEARCH_R1_COMMIT = "598e61bd1d36895726d28a8d06b3a15bed19f5d3"
E5_MODEL = "intfloat/e5-base-v2"
E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
MODEL_REVISIONS = {
    "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo": "bd4f5b0e8c19ed95fd0795295a02ed4a6a256daa",
    "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo": "713cbe32f48a45a855da9cd09a0d980c2e3166e6",
}
PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


def extract_search(text: str) -> Optional[str]:
    matches = re.findall(r"<search>(.*?)</search>", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[-1].strip() if matches else ""


def format_passages(retrieval_result: List[Dict[str, Any]]) -> str:
    formatted = ""
    for index, item in enumerate(retrieval_result):
        contents = item["document"]["contents"]
        title = contents.split("\n")[0]
        text = "\n".join(contents.split("\n")[1:])
        formatted += f"Doc {index + 1}(Title: {title}) {text}\n"
    return formatted


class HttpRetriever:
    def __init__(self, url: str, topk: int, timeout: float):
        self.url = url
        self.topk = topk
        self.timeout = timeout

    def search(self, query: str) -> str:
        response = requests.post(
            self.url,
            json={"queries": [query], "topk": self.topk, "return_scores": True},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return format_passages(response.json()["result"][0])


class LocalRetriever:
    def __init__(self, index_path: str, corpus_path: str, topk: int, device: str):
        import faiss
        from transformers import AutoModel, AutoTokenizer

        self.faiss = faiss
        self.index = faiss.read_index(index_path)
        self.corpus = [
            json.loads(line) for line in open(corpus_path, encoding="utf-8") if line.strip()
        ]
        self.topk = topk
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(E5_MODEL, revision=E5_REVISION)
        self.model = AutoModel.from_pretrained(E5_MODEL, revision=E5_REVISION).to(device).eval()

    def search(self, query: str) -> str:
        import torch

        tokens = self.tokenizer(
            ["query: " + query],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            output = self.model(**tokens, return_dict=True)
            hidden = output.last_hidden_state.masked_fill(
                ~tokens["attention_mask"][..., None].bool(), 0.0
            )
            embedding = hidden.sum(dim=1) / tokens["attention_mask"].sum(dim=1)[..., None]
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        scores, indices = self.index.search(embedding.cpu().numpy().astype("float32"), self.topk)
        result = [
            {"document": self.corpus[int(index)], "score": float(score)}
            for index, score in zip(indices[0], scores[0])
            if index >= 0
        ]
        return format_passages(result)


def run_eval(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_revision = args.model_revision or MODEL_REVISIONS.get(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=model_revision)
    llm = LLM(
        model=args.model_path,
        revision=model_revision,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
    )
    if args.retrieval_url:
        retriever = HttpRetriever(args.retrieval_url, args.retrieval_topk, args.timeout)
    else:
        if not args.retrieval_index or not args.retrieval_corpus:
            raise ValueError(
                "Provide --retrieval-url or both --retrieval-index and --retrieval-corpus"
            )
        retriever = LocalRetriever(
            args.retrieval_index,
            args.retrieval_corpus,
            args.retrieval_topk,
            args.retriever_device,
        )

    dataset = get_dataset(
        name="confiqa",
        setting=args.confiqa_setting,
        split="test",
        source="auto",
        limit=args.num_samples,
        seed=args.seed,
    )
    prompts = []
    for question in dataset["question"]:
        content = PROMPT.format(question=question.strip().rstrip("?") + "?")
        if tokenizer.chat_template:
            content = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
            )
        prompts.append(content)

    raw_outputs = [""] * len(dataset)
    traces: List[List[Dict[str, Any]]] = [[] for _ in dataset]
    finished = [False] * len(dataset)
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["</search>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    for turn in range(args.max_turns):
        active = [index for index, done in enumerate(finished) if not done]
        if not active:
            break
        outputs = llm.generate([prompts[index] for index in active], sampling, use_tqdm=True)
        for output, index in zip(outputs, active):
            generated = output.outputs[0].text
            raw_outputs[index] += generated
            answer = extract_answer(generated)
            query = extract_search(generated)
            if answer or not query:
                finished[index] = True
                continue
            information = retriever.search(query)
            traces[index].append({"turn": turn + 1, "query": query, "retrieved": information})
            prompts[index] += f"\n\n{generated}<information>{information}</information>\n\n"

    per_question = []
    em_total = 0.0
    f1_total = 0.0
    for index, row in enumerate(dataset):
        prediction = extract_answer(raw_outputs[index])
        em = max(
            (float(exact_match_score(prediction, gold)) for gold in row["answers"]), default=0.0
        )
        token_f1 = max((f1_score(prediction, gold)[0] for gold in row["answers"]), default=0.0)
        em_total += em
        f1_total += token_f1
        per_question.append(
            {
                "id": row["id"],
                "is_counterfactual": row["is_counterfactual"],
                "question": row["question"],
                "gold": row["answers"],
                "prediction": prediction,
                "em": em,
                "f1": token_f1,
                "trace": traces[index],
                "raw_output": raw_outputs[index],
            }
        )

    counterfactual_count = sum(bool(value) for value in dataset["is_counterfactual"])
    result = {
        "metadata": {
            "model": args.model_path,
            "model_revision": model_revision,
            "search_r1_code_revision": SEARCH_R1_COMMIT,
            "dataset_provenance": selected_rows_provenance(
                "confiqa",
                dataset["id"],
                seed=args.seed,
                setting=args.confiqa_setting,
                counterfactual_count=counterfactual_count,
            ),
            "generation": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "max_turns": args.max_turns,
            },
            "retrieval": {
                "model": E5_MODEL,
                "model_revision": E5_REVISION,
                "topk": args.retrieval_topk,
                "corpus": args.retrieval_corpus,
                "index": args.retrieval_index,
                "url": args.retrieval_url,
            },
        },
        "summary": {
            "n": len(dataset),
            "counterfactual_count": counterfactual_count,
            "em": em_total / max(1, len(dataset)),
            "f1": f1_total / max(1, len(dataset)),
        },
        "results": per_question,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    logging.info("Wrote %s: %s", output_path, result["summary"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--confiqa-setting", choices=["orig", "cf", "cf_100", "cf_500"], required=True
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Room for multi-turn observations; the Qwen2.5 checkpoints support this context.",
    )
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--retrieval-url")
    parser.add_argument("--retrieval-index")
    parser.add_argument("--retrieval-corpus")
    parser.add_argument("--retrieval-topk", type=int, default=3)
    parser.add_argument("--retriever-device", default="cpu")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_eval(args)


if __name__ == "__main__":
    main()
