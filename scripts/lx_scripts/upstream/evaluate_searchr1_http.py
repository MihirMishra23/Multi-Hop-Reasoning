"""
Evaluate autoregressive models on 2WikiMultiHopQA dataset.

Supports both HuggingFace transformers and vLLM backends.

Authors: Xi
Created: 2025-01-25
"""

import argparse
import os
import json
import uuid
import re
import time
from datetime import datetime
from typing import Optional, List, Dict
import torch
import torch.distributed as dist
from tqdm import tqdm
import transformers
import torch
import random
from datasets import load_dataset
import requests

import string
from torch.utils.data import Dataset

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def print_accuracy_report(metrics, show_errors=True, n_examples=5):
    print(f"\n{'='*50}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"{'='*50}")
    if show_errors and metrics.get("results"):
        errors = [r for r in metrics["results"] if not r["correct"]]
        if errors:
            print(f"\n  First {min(n_examples, len(errors))} errors:")
            for r in errors[:n_examples]:
                print(f"    Q: {r['question'][:80]}...")
                print(f"    Gold: {r['gold_answer']}  |  Pred: {r['predicted_answer']}\n")


class WikiMultiHopQAEvalDataset(Dataset):
    def __init__(self, data_paths, max_samples=None):
        self.samples = []
        for path in data_paths:
            if path.endswith(".jsonl"):
                with open(path) as f:
                    self.samples.extend([json.loads(l) for l in f])
            else:
                with open(path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.samples.extend(data)
                    elif isinstance(data, dict) and "data" in data:
                        self.samples.extend(data["data"])
        normalized = []
        for x in self.samples:
            q = x.get("question", x.get("query", ""))
            a = x.get("answer", x.get("answers", ""))
            if isinstance(a, list):
                a = a[0] if a else ""
            normalized.append({"question": q, "answer": str(a)})
        self.samples = normalized
        if max_samples:
            self.samples = self.samples[:max_samples]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# LOCAL CHANGE: pyserini now lives in the retrieval server process
# from pyserini.search.lucene import LuceneSearcher

class SparseBM25SRetriever:
    def __init__(self, index_path, corpus_path=None):
        print(f"[BM25] Loading index from: {index_path}")
        self.searcher = LuceneSearcher(index_path)
        self.corpus = {}
        if corpus_path and os.path.exists(corpus_path):
            print(f"[BM25] Loading corpus from: {corpus_path}")
            with open(corpus_path) as f:
                for line in f:
                    doc = json.loads(line.strip())
                    self.corpus[str(doc["id"])] = doc["contents"]
            print(f"[BM25] {len(self.corpus)} docs loaded")

    def retrieve(self, query, top_k=3):
        hits = self.searcher.search(query, k=top_k)
        results = []
        for hit in hits:
            try:
                raw = hit.lucene_document.get("raw")
                if raw:
                    doc = json.loads(raw)
                    results.append(doc.get("contents", raw))
                    continue
            except Exception:
                pass
            if hit.docid in self.corpus:
                results.append(self.corpus[hit.docid])
            else:
                results.append(f"[doc {hit.docid}]")
        return results

Retriever = SparseBM25SRetriever 
def extract_final_answer(x): return x  


# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

retriever: SparseBM25SRetriever = None

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    # LOCAL CHANGE: use the HTTP retrieval server instead of an in-process
    # pyserini LuceneSearcher. Identical BM25 index and top_k; this only moves
    # retrieval out of process so it can run in the pyserini env while the
    # model runs in the vLLM/torch env. Restores the block upstream had
    # commented out. RETRIEVAL_URL comes from the environment.
    import os as _os
    _url = _os.environ.get("RETRIEVAL_URL", "http://127.0.0.1:8000/retrieve")
    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True
    }
    raw_results = requests.post(_url, json=payload).json()['result'][0]

    results = []
    for item in raw_results:
        if isinstance(item, dict) and 'document' in item:
            results.append(item['document']['contents'])
        elif isinstance(item, str):
            results.append(item)
        else:
            results.append(str(item))

    def _passages2string(retrieval_result: List[str]) -> str:
        format_reference = ''
        for idx, content in enumerate(retrieval_result):
            format_reference += f"Doc {idx+1} {content}\n"
        return format_reference

    return _passages2string(results)


def do_inference(question):
    start_time = time.time()
    question = question.strip()
    if question[-1] != '?':
        question += '?'
    curr_eos = [151645, 151643] # for Qwen2.5 series model
    
    # Prepare the message
    prompt = f"""Answer the given question. \
    You must conduct reasoning inside <think> and </think> first every time you get new information. \
    After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
    You can search as many times as your want. \
    If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


    # Initialize the stopping criteria
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

    cnt = 0

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    print(prompt)
    # Encode the chat-formatted prompt and move it to the correct device
    while cnt < 10:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        print("Generating...")
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0
        )
        print(f"Generation completed: {outputs.shape[1]-input_ids.shape[1]} new tokens.")

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(output_text)
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print(f'searching "{tmp_query}"...')
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        print(search_text)

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL)
    if answer_match:
        output_text = answer_match.group(1).strip()

    print(output_text)

    return output_text, time.time() - start_time


def evaluate_hf(
    model,
    tokenizer,
    dataset: WikiMultiHopQAEvalDataset,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.95,
    device: str = "cuda",
    use_chat_template: bool = True,
    show_progress: bool = True,
    return_results: bool = False,
    retriever: Retriever = None,
    retrieve_top_k: int = 10,
    retrieve_token_gap: int = 20,  # Fixed typo
    enable_dynamic_retrieval: bool = False,  # New flag
) -> Dict:
    """
    Evaluate using HuggingFace transformers backend with optional dynamic retrieval.
    
    Args:
        enable_dynamic_retrieval: If True, retrieve new docs every retrieve_token_gap tokens
        retrieve_token_gap: Number of tokens to generate before next retrieval
    """
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        return {
            "question": [x["question"] for x in batch],
            "answer": [x["answer"] for x in batch],
            "aliases": [x.get("aliases", []) for x in batch],
            "id": [i for i in range(len(batch))],
            "supporting_docs": [x.get("supporting_docs") for x in batch],
            "reasoning": [x.get("reasoning") for x in batch],
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    

    correct = 0
    total = 0
    results = [] if return_results else None
    
    # Generation config
    gen_kwargs = {
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p if temperature > 0 else None,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        iterator = tqdm(dataloader, disable=not show_progress, desc="Evaluating")
        
        for batch in iterator:
            questions: List[str] = batch["question"]
            gold_answers: List[str] = batch["answer"]
            
            # Process each question in batch
            gen_texts = []
            gen_times = []
            
            for question in questions:
                # Static retrieval: retrieve once at the beginning
                gen_text, gen_time = do_inference(question=question)
            
                gen_texts.append(gen_text)
                gen_times.append(gen_time)
            
            # Evaluate
            for question, gold_answer, aliases, gen_text, gen_time in zip(questions, gold_answers, batch["aliases"], gen_texts, gen_times):
                predicted_answer = gen_text
                pred_normalized = normalize_answer(predicted_answer)
                gold_normalized = normalize_answer(gold_answer)
                is_correct = (pred_normalized == gold_normalized or any(pred_normalized == normalize_answer(a) for a in aliases))
                
                if is_correct:
                    correct += 1
                total += 1
                
                if return_results:
                    results.append({
                        "question": question,
                        "gold_answer": gold_answer,
                        "predicted_answer": predicted_answer,
                        "full_generation": gen_text,
                        "correct": is_correct,
                        "gen_time": gen_time,
                    })
                    print(f"Predicted answer: {predicted_answer}, Gold answer: {gold_answer}")
                    print(results[-1])
                if show_progress:
                    print(f"correct: {is_correct}, answer: {gold_answer}, predicted: {predicted_answer}")
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"[Eval] Final Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results if return_results else [],
    }


def main():
    global retriever
    parser = argparse.ArgumentParser(
        description="Evaluate autoregressive models on 2WikiMultiHopQA"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf"],
        default="hf",
        help="Backend to use for generation (only hf supported)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    
    # Generation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (only for HF backend)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat template"
    )
    
    # System arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (only for HF backend)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size (only for vLLM backend)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results JSON"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional filename for results JSON"
    )
    parser.add_argument(
        "--no_report",
        action="store_true",
        help="Don't print accuracy report to console"
    )
    parser.add_argument(
        "--retrieve_top_k",
        type=int,
        default=3,
        help="Number of passages to retrieve"
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        required=True,
        help="Path to corpus file"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        required=True,
        help="Path to index file"
    )
    parser.add_argument(
        "--retrieve_token_gap",
        type=int,
        default=20,
        help="Number of tokens to generate before next retrieval"
    )
    parser.add_argument(
        "--enable_dynamic_retrieval",
        action="store_true",
        help="Enable dynamic retrieval"
    )
    
    args = parser.parse_args()
    
    args.model_path = "searchr1"

    # Load dataset
    print(f"[Eval] Loading dataset from {args.data_path}")
    dataset = WikiMultiHopQAEvalDataset(
        data_paths=[args.data_path],
        max_samples=args.max_samples if args.max_samples != -1 else None
    )
    print(f"[Eval] Dataset size: {len(dataset)}")
    
    # LOCAL CHANGE: retrieval happens over HTTP (see search()), so no
    # in-process pyserini index is built. --index_path/--corpus_path are still
    # accepted and are what the retrieval server was started with.
    retriever = None
    
    # Run evaluation
    print(f"[Eval] Loading model from (HF backend)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    metrics = evaluate_hf(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        use_chat_template=not args.no_chat_template,
        show_progress=True,
        return_results=True,
        retrieve_top_k=args.retrieve_top_k,
        retrieve_token_gap=args.retrieve_token_gap,
        enable_dynamic_retrieval=args.enable_dynamic_retrieval,
    )
    
    # Print report
    if not args.no_report:
        print_accuracy_report(metrics, show_errors=True, n_examples=5)
    
    print(f"\n[Eval] Final Accuracy: {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total']})")
    
    args.checkpoint = "searchr1"

    # Save results
    if args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            ts = datetime.now().strftime("%m-%d_%H-%M")
            short_uuid = uuid.uuid4().hex[:8]
            
            model_name = args.model_path.split('/')[-1] if "checkpoint" not in args.model_path else args.model_path.split('/')[-2]+"-ckpt"+args.model_path.split('/')[-1].split("checkpoint-")[-1]
            dynamic_retrieval = f"-dyn{args.retrieve_token_gap}" if args.enable_dynamic_retrieval else ""
            if args.output_name:
                base_name = args.output_name.replace('.json', '')
                filename = f"{base_name}-{ts}.json"
            else:
                filename = f"eval_ar-{model_name}-n{args.max_samples}{dynamic_retrieval}-{ts}.json"
            
            out_path = os.path.join(args.output_dir, filename)
            
            payload = {
                "config": vars(args),
                "metrics": {
                    "accuracy": metrics["accuracy"],
                    "correct": metrics["correct"],
                    "total": metrics["total"],
                },
                "results": metrics.get("results", []),
            }
            
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            
            print(f"[Eval] Saved results to: {out_path}")
        except Exception as e:
            print(f"[Eval] Failed to save results: {e}")


if __name__ == "__main__":
    main()