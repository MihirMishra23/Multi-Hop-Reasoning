#!/usr/bin/env python3
"""Build the E5-base-v2 FAISS index consumed by the retrieval service."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def _load_corpus(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    corpus = _load_corpus(args.corpus_path)
    if not corpus:
        raise ValueError(f"Empty corpus: {args.corpus_path}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    embeddings = []

    for start in tqdm(range(0, len(corpus), args.batch_size), desc="Encoding corpus"):
        texts = ["passage: " + row["contents"] for row in corpus[start : start + args.batch_size]]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu().numpy())

    matrix = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "e5_Flat.index"
    faiss.write_index(index, str(index_path))
    print(f"Wrote {index.ntotal:,} vectors to {index_path}")


if __name__ == "__main__":
    main()
