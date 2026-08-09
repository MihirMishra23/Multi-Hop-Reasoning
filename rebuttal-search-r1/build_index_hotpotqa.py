#!/usr/bin/env python3
"""Build the E5-base-v2 FAISS index consumed by the retrieval service."""

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from reproduction_manifest import artifact


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def _load_corpus(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    e5 = artifact("e5_base_v2")
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=e5["repo_id"])
    parser.add_argument("--revision", default=e5["revision"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    corpus = _load_corpus(args.corpus_path)
    if not corpus:
        raise ValueError(f"Empty corpus: {args.corpus_path}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, revision=args.revision).to(device).eval()
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
    manifest_path = index_path.with_name(index_path.name + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus_sha256": _sha256(args.corpus_path),
                "retriever_model": args.model,
                "retriever_revision": args.revision,
                "vectors": int(index.ntotal),
                "dimension": int(matrix.shape[1]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {index.ntotal:,} vectors to {index_path}")
    print(f"Wrote index provenance to {manifest_path}")


if __name__ == "__main__":
    main()
