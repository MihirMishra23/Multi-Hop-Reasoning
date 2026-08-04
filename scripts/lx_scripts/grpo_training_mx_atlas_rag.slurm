#!/bin/bash

# Helping collaborator (mx253) run one GRPO job from her folder on 4x H100.
# Submit:
#   sbatch /home/lz586/icl/Multi-Hop-Reasoning/scripts/lx_scripts/grpo_training_mx_atlas_rag.slurm

#SBATCH -J grpo_mx_atlas_rag
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_mx_atlas_rag_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_mx_atlas_rag_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --get-user-env
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t infinite
#SBATCH --partition=jjs533
#SBATCH --gres=gpu:nvidia_h100_nvl:4
#SBATCH --requeue

source activate mem

# Disable MLflow integration (transformers auto-enables it because mlflow is
# installed; with 4 ranks racing on the default experiment dir we get
# "Could not find experiment with ID 0"). We don't use it here.
export DISABLE_MLFLOW_INTEGRATION=TRUE

# Reduce CUDA fragmentation; helps fit B200 preset (batch=16 originally) on H100 NVL (94GB).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /share/j_sun/mx253/Multi-Hop-Reasoning

bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/lx_scripts/grpo_train_mx_atlas_rag.sh \
    --gpu_type H100 \
    --learning_rate 5e-6 \
    --model_path /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_1phase \
    --dataset_name hotpotqa \
    --database_path /share/j_sun/mx253/Multi-Hop-Reasoning/kb_baselines/autoschemakg/output/kbs/hotpotqa_train_82347_8100_seed42_kb.json \
    --save_dir /share/j_sun/lz586/checkpoints/atlas_rag \
    --train_size 7000 \
    --reward_func f1 \
    --total_batch_size 512 \
    --phase1_prompt_type sft \
    --retrieval_threshold 0.6 \
    --top_k 4 \
    --save_steps 50

echo "GRPO Training (mx atlas_rag) complete!"
