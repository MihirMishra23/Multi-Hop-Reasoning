#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lmlm

python scripts/run_agent.py --dataset hotpotqa --setting distractor --split validation --method icl --model llama-3.2-1b-instruct --batch-size 5 --batch-number 1 --num-batches 50 --seed 42

# FullWiki RAG example (uses global corpus)
# python scripts/run_agent.py --dataset hotpotqa --setting fullwiki --split validation --method rag --model llama-3.2-1b-instruct --rag-corpus-path /share/j_sun/lmlm_multihop/datasets/hotpot_dev_fullwiki_v1.json --batch-size 5 --batch-number 1 --num-batches 50 --seed 42
