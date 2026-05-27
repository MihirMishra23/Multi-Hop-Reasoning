#!/bin/bash
# Eval the AutoSchemaKG (atlas-rag) static KBs with the KBevo-GRPO 1.7B Phase-2
# reasoner. Mirrors KBevo Table 5 eval setup, with:
#
#   --use-inverses    ← ENABLED (Phase-2 reasoner sees both (e,r,v) and (v,r,e))
#                       NOTE: the matching KBevo main eval pass also nominally
#                       sets --use-inverses, but its concat-all-db code path
#                       silently no-ops the flag (dead code in TopkRetriever).
#                       Per design decision (24-May-2026), we enable real inverse
#                       expansion here while leaving the KBevo dead-code bug to
#                       be fixed in a follow-up sweep.
#
#   Reasoner:   KBevo-GRPO 1.7B ckpt-500
#   Retrieval:  MiniLM-L6-v2, threshold 0.6, top-k 4
#   Sampling:   --use-train-params (T=1.0, top_p=0.95, vllm_top_k=4)
#   Dataset:    dev split, n=1000, seed=42 (matches KBevo main eval)
#
# Submits one sbatch per dataset for which the atlas-rag KB JSON exists.
# Skips datasets where the result file already exists (idempotent).
#
# Run on a login node:
#   bash menghan-scripts/eval_atlas_kb_all.slurm
#   bash menghan-scripts/eval_atlas_kb_all.slurm hotpotqa musique   # subset

cd /share/j_sun/mx253/Multi-Hop-Reasoning

# ── Parse args ───────────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    DATASETS=("$@")
else
    DATASETS=(hotpotqa musique 2wiki)
fi

GPU="${GPU:-b200}"
case "${GPU}" in
    b200)
        GRES="gpu:nvidia_b200:1"
        TIME_BUDG="1:30:00"
        ;;
    a6000)
        GRES="gpu:nvidia_rtx_a6000:1"
        TIME_BUDG="3:00:00"
        ;;
    *)
        echo "Error: GPU must be 'b200' or 'a6000', got '${GPU}'" >&2
        exit 1
        ;;
esac

# ── Fixed config ─────────────────────────────────────────────────────────────
N_QUESTIONS=1000
SEED=42
SPLIT=dev
SETTING=distractor
TOP_K=4
SIMILARITY_THRESHOLD=0.6

REASONER_PATH=/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500

KB_PKG=kb_baselines/autoschemakg
KB_DIR="${KB_PKG}/output/kbs"
EVAL_OUTPUT_DIR="${KB_PKG}/output/eval_runs"
SAVE_VERSION="_atlas_inv0524"   # 'inv' marker — these results have real inverse expansion

mkdir -p "${EVAL_OUTPUT_DIR}"

echo "GPU type:      ${GPU}  (GRES=${GRES}, time=${TIME_BUDG})"
echo "Reasoner:      ${REASONER_PATH##*/main/}"
echo "Datasets:      ${DATASETS[*]}"
echo "use-inverses:  ENABLED"
echo

for DATASET in "${DATASETS[@]}"; do
    KB_JSON="${KB_DIR}/${DATASET}_dev_n${N_QUESTIONS}_seed${SEED}_kb.json"

    if [[ ! -f "${KB_JSON}" ]]; then
        echo "Skipping ${DATASET}: KB not yet built (${KB_JSON})"
        continue
    fi

    JOB_NAME="atlas_eval_${DATASET}"
    TMP_SCRIPT=$(mktemp /tmp/atlas_eval_${DATASET}_XXXXXX.sh)

    cat > "${TMP_SCRIPT}" << SCRIPT_EOF
#!/bin/bash
set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate lmlm_b200
cd /share/j_sun/mx253/Multi-Hop-Reasoning

echo "============================================================"
echo "atlas-rag KB eval: ${DATASET}  (n=${N_QUESTIONS}, seed=${SEED}, --use-inverses)"
echo "KB: ${KB_JSON}"
echo "============================================================"

PYTHONPATH=src python src/eval_multihop.py \\
    --model-path "${REASONER_PATH}" \\
    --method two_phase \\
    --dataset "${DATASET}" --split "${SPLIT}" --setting "${SETTING}" \\
    --total-count ${N_QUESTIONS} \\
    --batch-size 64 \\
    --seed ${SEED} \\
    --save-every 64 \\
    --start-index 0 \\
    --top-k ${TOP_K} \\
    --similarity-threshold ${SIMILARITY_THRESHOLD} \\
    --use-inverses \\
    --use-train-params \\
    --concat-all-db \\
    --use-contexts all \\
    --edc-kb-path "${KB_JSON}" \\
    --output-dir "${EVAL_OUTPUT_DIR}" \\
    --save-version "${SAVE_VERSION}" \\
    --eval \\
    --resume

echo
echo "==== DONE: ${DATASET} ===="
echo "Find result with:"
echo "  find ${EVAL_OUTPUT_DIR} -name 'results_${DATASET}_dev_*${SAVE_VERSION#_}*.json'"
SCRIPT_EOF

    echo "Submitting: ${JOB_NAME}"
    sbatch \
        -J "${JOB_NAME}" \
        -o "/home/mx253/slurm_output/lmlm_multihop/${JOB_NAME}_%j.out" \
        -e "/home/mx253/slurm_output/lmlm_multihop/${JOB_NAME}_%j.err" \
        -N 1 -n 1 \
        --cpus-per-task=8 \
        --mem=128G \
        -t "${TIME_BUDG}" \
        --partition=aimi,default_partition \
        --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03 \
        --gres="${GRES}" \
        --requeue \
        "${TMP_SCRIPT}"
    rm -f "${TMP_SCRIPT}"
done

echo
echo "All submitted. Status:"
echo "  squeue -u \$USER -n atlas_eval_hotpotqa,atlas_eval_musique,atlas_eval_2wiki"
