#!/usr/bin/env python3
"""Serve the E5/FAISS retriever using verl's Search-R1 `/retrieve` protocol."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from reproduction_manifest import artifact


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DenseRetriever:
    def __init__(self, index_path, corpus_path, model_name, model_revision, device, topk):
        self.index = faiss.read_index(str(index_path))
        with Path(corpus_path).open(encoding="utf-8") as stream:
            self.corpus = [json.loads(line) for line in stream if line.strip()]
        if self.index.ntotal != len(self.corpus):
            raise ValueError(f"Index has {self.index.ntotal} vectors but corpus has {len(self.corpus)} rows")
        manifest_path = Path(index_path).with_name(Path(index_path).name + ".manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"Missing index provenance: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "schema_version": 1,
            "corpus_sha256": _sha256(corpus_path),
            "retriever_model": model_name,
            "retriever_revision": model_revision,
            "vectors": int(self.index.ntotal),
            "dimension": int(self.index.d),
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Index provenance mismatch: {mismatches}")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision, use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, revision=model_revision
        ).to(self.device).eval()
        self.topk = topk

    @torch.no_grad()
    def encode(self, queries):
        inputs = self.tokenizer(
            ["query: " + query for query in queries],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs)
        hidden = output.last_hidden_state.masked_fill(~inputs["attention_mask"][..., None].bool(), 0.0)
        embedding = hidden.sum(1) / inputs["attention_mask"].sum(1)[..., None]
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding.cpu().numpy().astype(np.float32, order="C")

    def search(self, queries, topk):
        scores, indices = self.index.search(self.encode(queries), topk)
        return [
            [(self.corpus[int(doc_id)], float(score)) for doc_id, score in zip(row_ids, row_scores)]
            for row_ids, row_scores in zip(indices, scores)
        ]


app = FastAPI()
retriever: Optional[DenseRetriever] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is still loading")
    topk = request.topk or retriever.topk
    results = retriever.search(request.queries, topk)
    if request.return_scores:
        payload = [
            [{"document": document, "score": score} for document, score in row]
            for row in results
        ]
    else:
        payload = [[document for document, _ in row] for row in results]
    return {"result": payload}


def main():
    parser = argparse.ArgumentParser()
    e5 = artifact("e5_base_v2")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--model", default=e5["repo_id"])
    parser.add_argument("--revision", default=e5["revision"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global retriever
    retriever = DenseRetriever(
        args.index_path,
        args.corpus_path,
        args.model,
        args.revision,
        args.device,
        args.topk,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
