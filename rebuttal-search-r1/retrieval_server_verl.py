#!/usr/bin/env python3
"""
Local dense retrieval server that speaks verl's /retrieve protocol.

Adapted from verl-project/verl
  examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py
with two changes for our setup:

  1. FAISS stays on CPU by default. The PyPI faiss-gpu-cu12 wheel does not
     ship Blackwell (SM 10.0) kernels, so faiss.index_cpu_to_all_gpus crashes
     on B200 with "no kernel image available for execution on the device".
     Pass --faiss_gpu to force GPU (will only work on Ampere/Hopper).
  2. E5 encoder stays on GPU — it is tiny (~440 MB) and dwarfed by the
     training process's vLLM/FSDP allocations on the same card.

Input  (POST /retrieve):
    {"queries": ["...", ...], "topk": 3, "return_scores": true}
Output:
    {"result": [
        [{"document": {"id": ..., "contents": "..."}, "score": float}, ...],
        ...
    ]}
"""

import argparse
from typing import Optional

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_corpus(corpus_path: str):
    return datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)


def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]


def pooling(pooler_output, last_hidden_state, attention_mask, method="mean"):
    if method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if method == "cls":
        return last_hidden_state[:, 0]
    if method == "pooler":
        return pooler_output
    raise NotImplementedError(method)


class Encoder:
    def __init__(self, model_name, model_path, pooling_method="mean", max_length=256, use_fp16=False):
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
        if use_fp16:
            self.model = self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    @torch.no_grad()
    def encode(self, query_list, is_query=True):
        if isinstance(query_list, str):
            query_list = [query_list]
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            query_list = [prefix + q for q in query_list]
        inputs = self.tokenizer(
            query_list, max_length=self.max_length,
            padding=True, truncation=True, return_tensors="pt",
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        out = self.model(**inputs, return_dict=True)
        emb = pooling(out.pooler_output, out.last_hidden_state, inputs["attention_mask"], self.pooling_method)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        emb = emb.detach().cpu().numpy().astype(np.float32, order="C")
        torch.cuda.empty_cache()
        return emb


class DenseRetriever:
    def __init__(self, index_path, corpus_path, encoder, topk=3, batch_size=128, faiss_gpu=False):
        self.index = faiss.read_index(index_path)
        if faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        self.corpus = load_corpus(corpus_path)
        self.encoder = encoder
        self.topk = topk
        self.batch_size = batch_size

    def batch_search(self, query_list, num=None, return_score=False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        results, scores = [], []
        for s in tqdm(range(0, len(query_list), self.batch_size), desc="retrieve"):
            qb = query_list[s : s + self.batch_size]
            emb = self.encoder.encode(qb)
            bs, bi = self.index.search(emb, k=num)
            flat = sum(bi.tolist(), [])
            docs = load_docs(self.corpus, flat)
            results.extend([docs[i * num : (i + 1) * num] for i in range(len(bi))])
            scores.extend(bs.tolist())
        if return_score:
            return results, scores
        return results, [[] for _ in results]


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()
retriever: Optional[DenseRetriever] = None


@app.post("/retrieve")
def retrieve(req: QueryRequest):
    topk = req.topk or retriever.topk
    results, scores = retriever.batch_search(req.queries, num=topk, return_score=req.return_scores)
    resp = []
    for i, single in enumerate(results):
        if req.return_scores:
            resp.append([{"document": d, "score": s} for d, s in zip(single, scores[i], strict=True)])
        else:
            resp.append(single)
    return {"result": resp}


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    parser.add_argument("--faiss_gpu", action="store_true",
                        help="Move FAISS index to GPU. NOT supported on Blackwell with current faiss-gpu-cu12.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global retriever
    encoder = Encoder(model_name=args.retriever_name, model_path=args.retriever_model)
    retriever = DenseRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        encoder=encoder,
        topk=args.topk,
        faiss_gpu=args.faiss_gpu,
    )
    print(f"✓ loaded {len(retriever.corpus)} passages; FAISS on {'GPU' if args.faiss_gpu else 'CPU'}")
    print(f"✓ serving on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
