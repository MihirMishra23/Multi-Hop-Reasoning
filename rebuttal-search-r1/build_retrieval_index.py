#!/usr/bin/env python3
"""Build Search-R1's normalized E5 Flat inner-product index."""

import argparse
import json
from pathlib import Path

E5_MODEL = "intfloat/e5-base-v2"
E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"


def mean_pool(last_hidden_state, attention_mask):
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import faiss
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    records = [json.loads(line) for line in open(args.corpus, encoding="utf-8") if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL, revision=E5_REVISION)
    model = AutoModel.from_pretrained(E5_MODEL, revision=E5_REVISION).to(args.device).eval()
    embeddings = []
    for start in range(0, len(records), args.batch_size):
        texts = ["passage: " + row["contents"] for row in records[start : start + args.batch_size]]
        tokens = tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(args.device) for key, value in tokens.items()}
        with torch.no_grad():
            output = model(**tokens, return_dict=True)
            batch = mean_pool(output.last_hidden_state, tokens["attention_mask"])
            batch = torch.nn.functional.normalize(batch, dim=-1)
        embeddings.append(batch.cpu().numpy())

    matrix = np.concatenate(embeddings).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(
        json.dumps(
            {"index": str(output_path), "passages": len(records), "dimensions": matrix.shape[1]}
        )
    )


if __name__ == "__main__":
    main()
