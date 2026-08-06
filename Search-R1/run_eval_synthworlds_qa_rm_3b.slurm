#!/bin/bash
#SBATCH --job-name=sr1_2wiki_3b
#SBATCH --partition=aimi,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/share/j_sun/mx253/Search-R1/logs/sr1_2wiki_3b_%j.out
#SBATCH --error=/share/j_sun/mx253/Search-R1/logs/sr1_2wiki_3b_%j.err
#SBATCH --requeue

source activate searchr1
export HF_HOME=/share/j_sun/mx253/home_caches/.cache/huggingface
cd /share/j_sun/mx253/Search-R1

python evaluate_searchr1.py \
    --data_path ./val_dataset/synthworlds_qa_rm_val.json \
    --index_path ./index/synthworlds_qa_rm/bm25 \
    --corpus_path ./corpus/synthworlds_qa_rm_corpus.jsonl \
    --output_dir ./results/synthworlds_qa_rm_3b