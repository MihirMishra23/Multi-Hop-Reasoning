#!/bin/bash
#SBATCH --job-name=search_r1_eval
#SBATCH --partition=aimi,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/share/j_sun/mx253/Search-R1/logs/search_r1_eval_%j.out
#SBATCH --error=/share/j_sun/mx253/Search-R1/logs/search_r1_eval_%j.err
#SBATCH --requeue

# Environment setup
source activate searchr1
export HF_HOME=/share/j_sun/mx253/home_caches/.cache/huggingface

cd /share/j_sun/mx253/Search-R1

# Run evaluation with local BM25 retriever (no server needed)
python evaluate_searchr1_7b.py \
    --data_path ./val_dataset/popqa_1000.json \
    --index_path ./index/popqa_1000/bm25 \
    --corpus_path ./corpus/popqa_corpus_chunked_800.jsonl \
    --output_dir ./results/popqa_7b