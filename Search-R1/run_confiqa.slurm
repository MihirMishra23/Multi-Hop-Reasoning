#!/bin/bash
#SBATCH --job-name=sr1_confiqa_mix
#SBATCH --partition=aimi,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_h100_nvl:1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/share/j_sun/mx253/Search-R1/logs/sr1_confiqa_mix_%A_%a.out
#SBATCH --error=/share/j_sun/mx253/Search-R1/logs/sr1_confiqa_mix_%A_%a.err
#SBATCH --requeue
#SBATCH --array=0-3

source activate searchr1
export HF_HOME=/share/j_sun/mx253/home_caches/.cache/huggingface
cd /share/j_sun/mx253/Search-R1

# 4 tasks: 2 datasets x 2 models
DATASETS=(mr_mix100 mr_mix500)
MODELS=(3b 7b)

# Map array index -> dataset + model
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 2))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID % 2))

DATASET=${DATASETS[$DATASET_IDX]}
MODEL=${MODELS[$MODEL_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: dataset=${DATASET}, model=${MODEL}"

mkdir -p ./results/confiqa_${DATASET}_${MODEL}

if [ "$MODEL" == "3b" ]; then
    SCRIPT="evaluate_searchr1.py"
else
    SCRIPT="evaluate_searchr1_7b.py"
fi

python $SCRIPT \
    --data_path ./val_dataset/confiqa_${DATASET}_1000.json \
    --index_path ./index/confiqa_${DATASET}_1000/bm25 \
    --corpus_path ./corpus/confiqa_${DATASET}_1000_corpus.jsonl \
    --output_dir ./results/confiqa_${DATASET}_${MODEL} \
    --output_name confiqa_${DATASET}_1000_${MODEL}