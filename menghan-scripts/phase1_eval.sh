#!/bin/bash
# submit_phase1_eval.sh

BASE="/share/j_sun/mx253/Multi-Hop-Reasoning"
SCRIPT="$BASE/reward_hacking_evaluate/phase1/evaluate_phase1.py"

# ── Job 1: grpo_1.7b ──
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=eval-phase1-grpo-1.7b
#SBATCH --partition=aimi,jjs533,default_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=dean-compute-02
#SBATCH --time=5:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/eval_phase1_grpo_1.7b_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/eval_phase1_grpo_1.7b_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning

python reward_hacking_evaluate/phase1/evaluate_phase1.py \
    --csv KG_results/grpo_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.csv \
    --output reward_hacking_evaluate/phase1/phase1_results_grpo_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.json \
    --num_rows 100

echo "Done: grpo_1.7b"
EOF

# ── Job 2: new_sft_1.7b ──
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=eval-phase1-new-sft-1.7b
#SBATCH --partition=aimi,jjs533,default_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=dean-compute-02
#SBATCH --time=5:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/eval_phase1_new_sft_1.7b_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/eval_phase1_new_sft_1.7b_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning

python reward_hacking_evaluate/phase1/evaluate_phase1.py \
    --csv KG_results/new_sft_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.csv \
    --output reward_hacking_evaluate/phase1/phase1_results_new_sft_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.json \
    --num_rows 100

echo "Done: new_sft_1.7b"
EOF

echo "Both jobs submitted!"