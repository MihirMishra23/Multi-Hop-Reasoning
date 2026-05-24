#!/usr/bin/env python3
"""Build FAISS index for HotpotQA corpus using E5 embeddings."""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import faiss


def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_corpus(corpus_path, model_name, batch_size=32, max_length=512):
    """Encode corpus using E5 model."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Loading corpus from: {corpus_path}")
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))

    print(f"Loaded {len(corpus)} passages")

    # Encode in batches
    all_embeddings = []

    for i in tqdm(range(0, len(corpus), batch_size), desc="Encoding"):
        batch = corpus[i:i + batch_size]
        texts = ["passage: " + item["contents"] for item in batch]  # E5 prefix

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = mean_pooling(outputs, encoded['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    embeddings_matrix = np.vstack(all_embeddings).astype('float32')
    print(f"Embeddings shape: {embeddings_matrix.shape}")

    return embeddings_matrix, corpus


def build_faiss_index(embeddings, index_type="Flat"):
    """Build FAISS index."""
    dimension = embeddings.shape[1]

    if index_type == "Flat":
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    elif index_type == "IVF":
        quantizer = faiss.IndexFlatIP(dimension)
        nlist = min(100, embeddings.shape[0] // 10)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Move to GPU if available
    if faiss.get_num_gpus() > 0:
        print("Using GPU for indexing")
        index = faiss.index_cpu_to_all_gpus(index)

    print(f"Adding {embeddings.shape[0]} vectors to index...")
    index.add(embeddings)

    # Move back to CPU for saving
    if faiss.get_num_gpus() > 0:
        index = faiss.index_gpu_to_cpu(index)

    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="intfloat/e5-base-v2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--index_type", default="Flat", choices=["Flat", "IVF"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode corpus
    embeddings, corpus = encode_corpus(
        args.corpus_path,
        args.model_name,
        batch_size=args.batch_size
    )

    # Build index
    print(f"Building {args.index_type} index...")
    index = build_faiss_index(embeddings, index_type=args.index_type)

    # Save index
    index_path = output_dir / "e5_Flat.index"
    faiss.write_index(index, str(index_path))
    print(f"✓ Saved index to {index_path}")

    # Save corpus mapping (for retrieval)
    corpus_mapping_path = output_dir / "corpus_mapping.jsonl"
    with open(corpus_mapping_path, 'w', encoding='utf-8') as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Saved corpus mapping to {corpus_mapping_path}")

    print("\n✓ Indexing complete!")


if __name__ == "__main__":
    main()
