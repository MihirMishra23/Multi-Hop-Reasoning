#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_PATH="${SCRIPT_DIR}/data/index/e5_Flat.index"
CORPUS_PATH="${SCRIPT_DIR}/data/hotpotqa_corpus.jsonl"
PORT=${1:-8000}

echo "════════════════════════════════════════════════════════════════"
echo "Starting Local Retrieval Server"
echo "════════════════════════════════════════════════════════════════"

if [ ! -f "${INDEX_PATH}" ]; then
    echo "❌ Error: Index not found at ${INDEX_PATH}"
    echo "Please run build_index.sh first"
    exit 1
fi

if [ ! -f "${CORPUS_PATH}" ]; then
    echo "❌ Error: Corpus not found at ${CORPUS_PATH}"
    echo "Please run prepare_hotpotqa_corpus.py first"
    exit 1
fi

echo "Index:  ${INDEX_PATH}"
echo "Corpus: ${CORPUS_PATH}"
echo "Port:   ${PORT}"
echo ""

# Activate retriever environment
eval "$(conda shell.bash hook)"
conda activate retriever

# Create retrieval server script
cat > "${SCRIPT_DIR}/retrieval_server.py" << 'EOF'
#!/usr/bin/env python3
"""Local retrieval server using E5 and FAISS."""

import argparse
import json
import numpy as np
import torch
import faiss
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModel
import uvicorn


app = FastAPI()

# Global variables
tokenizer = None
model = None
index = None
corpus = None
device = None


class SearchQuery(BaseModel):
    query: str
    top_k: int = 3


class SearchResult(BaseModel):
    contents: List[str]
    scores: List[float]


def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_query(query_text):
    """Encode query using E5 model."""
    # E5 uses "query: " prefix for queries
    texts = [f"query: {query_text}"]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = mean_pooling(outputs, encoded['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


@app.post("/search", response_model=SearchResult)
async def search(query: SearchQuery):
    """Search for relevant passages."""
    # Encode query
    query_embedding = encode_query(query.query)

    # Search
    scores, indices = index.search(query_embedding, query.top_k)

    # Get results
    results = []
    result_scores = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(corpus):
            results.append(corpus[idx]["contents"])
            result_scores.append(float(score))

    return SearchResult(contents=results, scores=result_scores)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


def load_resources(index_path, corpus_path, model_name):
    """Load index, corpus, and model."""
    global tokenizer, model, index, corpus, device

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Loading index from: {index_path}")
    index = faiss.read_index(str(index_path))

    # Move to GPU if available
    if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
        print("Moving index to GPU")
        index = faiss.index_cpu_to_all_gpus(index)

    print(f"Loading corpus from: {corpus_path}")
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))

    print(f"✓ Loaded {len(corpus)} passages")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--model_name", default="intfloat/e5-base-v2")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    load_resources(args.index_path, args.corpus_path, args.model_name)

    print(f"\n✓ Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
EOF

chmod +x "${SCRIPT_DIR}/retrieval_server.py"

# Launch server
echo "Starting retrieval server..."
python "${SCRIPT_DIR}/retrieval_server.py" \
    --index_path="${INDEX_PATH}" \
    --corpus_path="${CORPUS_PATH}" \
    --model_name="intfloat/e5-base-v2" \
    --port="${PORT}"
