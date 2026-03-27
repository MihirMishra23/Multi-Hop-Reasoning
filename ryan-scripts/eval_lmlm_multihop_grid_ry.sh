#
# Setup instructions — eval_lmlm_multihop_grid.sh
#
# Methods supported: direct, icl, rag, lmlm
# Grid search over: DATASETS, MODEL_PATHS, TOP_K, RETRIEVAL_THRESHOLD
#
# Prereqs:
# - Activate your environment and install repo deps:
#   pip install -e .
# - (RAG only) Install FlashRAG:
#   cd src/tools
#   git clone https://github.com/RUC-NLPIR/FlashRAG.git
#   cd FlashRAG
#   pip install -e .
#
# Run:
#   bash scripts/eval_lmlm_multihop_grid.sh --method lmlm
#
# Grid search example:
#   bash scripts/eval_lmlm_multihop_grid.sh \
#     --method lmlm \
#     --grid-search \
#     --datasets hotpotqa,musique \
#     --top-k-values 1,4 \
#     --retrieval-threshold-values 0.9,0.6 \
#     --model-paths /path/to/model1,/path/to/model2
#
# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48
LLM_MODEL=gpt-4
USE_INVERSES="--use-inverses" # or "--use-inverses"
PHASE_1="--phase-1"  # Set to "--phase-1" to enable phase-1 mode (dynamic database building)
PHASE_1=""
NUM_SAMPLES=32
SAVE_VERSION=""
METHODS=("lmlm")
DATASETS=("2wiki")
DATASETS=( "hotpotqa" )
DATASETS=("hotpotqa" "2wiki" "musique" )
LIMIT=32


SPLIT="train_train1k"
SPLIT="dev"

# Grid search parameters
GRID_SEARCH=true
TOP_K_VALUES=(4 1)
TOP_K_VALUES=(4)
RETRIEVAL_THRESHOLD_VALUES=(0.6 0.9)
RETRIEVAL_THRESHOLD_VALUES=(0.6)
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/ryan-march-sft-exp/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_classic_retrieval /share/j_sun/lmlm_multihop/checkpoints/ryan-march-sft-exp/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_return_triplets)
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_two_phase_hotpotqa_ep5_bsz48-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-nak/checkpoint-1750 /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_two_phase_hotpotqa_ep5_bsz48-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-nak/checkpoint-1750)
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_two_phase_hotpotqa_ep5_bsz48-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-nak/checkpoint-1750)
MODEL_PATHS=(/share/j_sun/rtn27/checkpoints/lmlm_multi_hop//Qwen3-4B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_classic_retrieval)
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/ryan-march-sft-exp/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_classic_retrieval)
# Parse command line arguments
MODEL_PATHS=(/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_classic_retrieval_6k)
CUSTOM_DATABASE_PATH="phase1_database_Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_classic_retrieval_6k_hotpotqa_dev_2026-03-16_01-01-55.json"
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/ryan-march-sft-exp/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_classic_retrieval)
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-1314)
CUSTOM_DATABASE_PATH=phase1_database_Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_classic_retrieval_hotpotqa_train_val1k_2026-03-18_01-56-28.json
CUSTOM_DATABASE_PATH=""

MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/ryan-march-sft-exp/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_classic_retrieval /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-3066)
MODEL_PATHS=(/share/j_sun/rtn27/checkpoints/lmlm_multi_hop//Qwen3-4B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_classic_retrieval /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-1314 /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-3066)
MODEL_PATHS=(/share/j_sun/rtn27/checkpoints/lmlm_multi_hop//Qwen3-4B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_classic_retrieval /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-1314 /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-1752 /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-2190 /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-2628 /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-3066)
MODEL_PATHS=(/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount_dynamic-th0.6-topk4-nak/checkpoint-876)
MODEL_PATHS=(/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k)
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/ryan-march-sft-exp/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_classic_retrieval)
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/ryan-march-sft-exp/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_classic_retrieval /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-3066)

MODEL_PATHS=(
    /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N8-K4-B64-M2-b0.0-step500-n7000-f1-2ph-prcontext_only-wcount_dynamic-th0.6-topk4-nak/checkpoint-500
    /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-2ph-prcontext_only-wcount-th0.6-topk4-nak/checkpoint-500
    /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-em-2ph-prcontext_only-wcount-th0.6-topk4-nak/checkpoint-500
    /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs1024-N32-K4-B32-M7-b0.0-step500-n7000-em-2ph-prcontext_only-wcount-th0.6-topk4-nak/checkpoint-200
    /share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-v2-2ph-prcontext_only-wcount-th0.6-topk4-nak/checkpoint-200
)

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --database-path)
            CUSTOM_DATABASE_PATH="$2"
            shift 2
            ;;
        --method)
            METHODS=("$2")
            shift 2
            ;;
        --llm_model)
            LLM_MODEL="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --use-inverses)
            USE_INVERSES="--use-inverses"
            shift
            ;;
        --phase-1)
            PHASE_1="--phase-1"
            shift
            ;;
        --save_version)
            SAVE_VERSION="$2"
            shift 2
            ;;
        --grid-search)
            GRID_SEARCH=true
            shift
            ;;
        --top-k-values)
            IFS=',' read -ra TOP_K_VALUES <<< "$2"
            shift 2
            ;;
        --retrieval-threshold-values)
            IFS=',' read -ra RETRIEVAL_THRESHOLD_VALUES <<< "$2"
            shift 2
            ;;
        --model-paths)
            IFS=',' read -ra MODEL_PATHS <<< "$2"
            shift 2
            ;;
        --datasets)
            IFS=',' read -ra DATASETS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

MAX_TOKENS=1024
BATCH_SIZE_DIRECT=64
BATCH_SIZE_ICL=1
BATCH_SIZE_RAG=1

# Detect GPU type and set LMLM batch size
echo "Detecting GPU type..."
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "GPU detected: ${GPU_INFO}"

if [[ "${GPU_INFO}" == *"H100"* ]]; then
    BATCH_SIZE_LMLM=64
    echo "Setting LMLM batch size to 64 for H100"
elif [[ "${GPU_INFO}" == *"6000 Ada"* ]] || [[ "${GPU_INFO}" == *"RTX 6000 Ada"* ]]; then
    BATCH_SIZE_LMLM=32
    echo "Setting LMLM batch size to 32 for 6000 Ada"
else
    BATCH_SIZE_LMLM=32
    echo "Setting LMLM batch size to 16 (default)"
fi

echo "LMLM Batch Size: ${BATCH_SIZE_LMLM}"

OUTPUT_DIR=./march-23-linxi-eval
SETTING=distractor
SAVE_EVERY=64
SEED=42


# If grid search is enabled, use MODEL_PATHS array; otherwise use single MODEL_PATH
if [ "$GRID_SEARCH" = true ]; then
    if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
        MODEL_PATHS=("$MODEL_PATH")
    fi

    echo "=== Grid Search Mode ==="
    echo "Datasets: ${DATASETS[@]}"
    echo "Models: ${MODEL_PATHS[@]}"
    echo "TOP_K values: ${TOP_K_VALUES[@]}"
    echo "RETRIEVAL_THRESHOLD values: ${RETRIEVAL_THRESHOLD_VALUES[@]}"
    echo "Methods: ${METHODS[@]}"
    echo "Use inverses: ${USE_INVERSES}"
    echo "Phase-1 mode: ${PHASE_1:-disabled}"
    echo "Total combinations: $((${#DATASETS[@]} * ${#MODEL_PATHS[@]} * ${#TOP_K_VALUES[@]} * ${#RETRIEVAL_THRESHOLD_VALUES[@]} * ${#METHODS[@]}))"
    echo "======================="

    for DATASET in "${DATASETS[@]}"; do
        # Clear variables at the start of each dataset iteration
        SUB_SPLIT=""

        # Set dataset-specific START_IDX, DEFAULT_NUM_SAMPLES, and map to actual HF split
        if [ "${DATASET}" = "hotpotqa" ]; then
            if [ "${SPLIT}" = "dev" ]; then
                DEFAULT_NUM_SAMPLES=1000
                START_IDX=0
                HF_SPLIT="dev"
            elif [ "${SPLIT}" = "debug_dev" ]; then
                DEFAULT_NUM_SAMPLES=1000
                START_IDX=0
                HF_SPLIT="dev"
            elif [ "${SPLIT}" = "train_val100" ]; then
                DEFAULT_NUM_SAMPLES=100
                START_IDX=90347
                HF_SPLIT="train"
            elif [ "${SPLIT}" = "train_val1k" ]; then
                DEFAULT_NUM_SAMPLES=1000
                START_IDX=82347
                HF_SPLIT="train"
                SUB_SPLIT="train"
                LIMIT=1000
            elif [ "${SPLIT}" = "train_train1k" ]; then
                DEFAULT_NUM_SAMPLES=1000
                START_IDX=89347
                HF_SPLIT="train"
                SUB_SPLIT="train"
            else
                echo "Error: Invalid SPLIT '${SPLIT}' for hotpotqa"
                exit 1
            fi
        elif [ "${DATASET}" = "musique" ]; then
            if [ "${SPLIT}" = "dev" ]; then
                DEFAULT_NUM_SAMPLES=1000
                START_IDX=0
                HF_SPLIT="dev"
            else
                echo "Error: Invalid SPLIT '${SPLIT}' for musique"
                exit 1
            fi
        elif [ "${DATASET}" = "two_wiki" ] || [ "${DATASET}" = "2wiki" ]; then
            if [ "${SPLIT}" = "dev" ]; then
                DEFAULT_NUM_SAMPLES=1000
                START_IDX=0
                HF_SPLIT="dev"
            else
                echo "Error: Invalid SPLIT '${SPLIT}' for 2wiki"
                exit 1
            fi
        else
            echo "Error: Unknown dataset '${DATASET}'"
            exit 1
        fi

        echo ""
        echo "=== Processing Dataset: ${DATASET} (split: ${SPLIT} -> HF split: ${HF_SPLIT}) ==="
        echo ""

    for MODEL in "${MODEL_PATHS[@]}"; do
        # Extract model name for logging
        MODEL_NAME=$(basename "$MODEL")

        # Check for adaptive-k flag
        if [[ "${MODEL}" == *"-nak"* ]]; then
            ADAPTIVE_K=""
        else
            ADAPTIVE_K="--adaptive-k"
        fi

        # Check for return-triplets flag
        if [[ "${MODEL}" == *"return_triplets"* ]]; then
            RETURN_TRIPLETS="--return-triplets"
        else
            RETURN_TRIPLETS=""
        fi

        # Build database for this model (Phase 1) or use custom database or skip if phase-1 mode
        if [ -n "${PHASE_1}" ]; then
            echo ""
            echo "=========================================="
            echo "Skipping Phase 1: Using --phase-1 mode"
            echo "Database will be built dynamically from golden contexts"
            echo "=========================================="
            BUILT_DATABASE=""
        elif [ -n "${CUSTOM_DATABASE_PATH}" ]; then
            echo ""
            echo "=========================================="
            echo "Skipping Phase 1: Using custom database"
            echo "Database: ${CUSTOM_DATABASE_PATH}"
            echo "=========================================="
            BUILT_DATABASE="${CUSTOM_DATABASE_PATH}"

            if [ ! -f "${BUILT_DATABASE}" ]; then
                echo "Error: Custom database file ${BUILT_DATABASE} does not exist"
                exit 1
            fi
        else
            TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
            BUILT_DATABASE="${OUTPUT_DIR}/phase1_database_${MODEL_NAME}_${DATASET}_${SPLIT}_${TIMESTAMP}.json"

            echo ""
            echo "=========================================="
            echo "Phase 1: Building database for ${MODEL_NAME}"
            echo "Output: ${BUILT_DATABASE}"
            echo "=========================================="

            # Build sub_split arg only if SUB_SPLIT is set
            SUB_SPLIT_ARG=""
            if [ -n "${SUB_SPLIT}" ]; then
                SUB_SPLIT_ARG="--sub_split=${SUB_SPLIT}"
            fi

            echo "The HF_SPLIT is ${HF_SPLIT}"
            echo "The SUB_SPLIT_ARG is ${SUB_SPLIT_ARG}"
            echo "The LIMIT is ${LIMIT}"
            echo "The HF_SPLIT is ${HF_SPLIT}"

            python scripts/run_phase_1.py \
                --model_path="${MODEL}" \
                --dataset="${DATASET}" \
                --split="${HF_SPLIT}" \
                --seed=${SEED} \
                --batch_size=${BATCH_SIZE_LMLM} \
                --output_file="${BUILT_DATABASE}" \
                --limit="${LIMIT}" \
                ${SUB_SPLIT_ARG}

            if [ ! -f "${BUILT_DATABASE}" ]; then
                echo "Error: Database file ${BUILT_DATABASE} was not created"
                exit 1
            fi

            echo "Database created successfully: ${BUILT_DATABASE}"
        fi
        echo ""

        for TOP_K in "${TOP_K_VALUES[@]}"; do
            for THRESHOLD in "${RETRIEVAL_THRESHOLD_VALUES[@]}"; do
                echo ""
                echo "=========================================="
                echo "Grid search: MODEL=$MODEL_NAME, TOP_K=$TOP_K, THRESHOLD=$THRESHOLD"
                echo "=========================================="

                for METHOD in "${METHODS[@]}"; do
                    echo "Running method: ${METHOD}"

                    # Create unique save version for this grid combination
                    GRID_SAVE_VERSION="${SAVE_VERSION}_${DATASET}_${MODEL_NAME}_k${TOP_K}_th${THRESHOLD}"

                    if [ -n "${PHASE_1}" ]; then
                        GRID_SAVE_VERSION="${GRID_SAVE_VERSION}_dynamic_phase1"
                    fi

                    # If USE_INVERSES is set, add 'use_inv' to the save version
                    if [ -n "${USE_INVERSES}" ]; then
                        GRID_SAVE_VERSION="${GRID_SAVE_VERSION}_use_inv"
                    fi

                    if [ "${METHOD}" = "lmlm" ]; then
                        # Build sub_split arg only if SUB_SPLIT is set
                        # When using sub_split, do NOT pass --start-index (sub_split handles indexing)
                        SUB_SPLIT_ARG_PHASE2=""
                        START_INDEX_ARG="--start-index ${START_IDX}"
                        if [ -n "${SUB_SPLIT}" ]; then
                            SUB_SPLIT_ARG_PHASE2="--sub-split ${SUB_SPLIT}"
                            START_INDEX_ARG=""  # Don't use start-index when sub_split is set
                        fi

                        # Only pass --database-path if not in phase-1 mode
                        DATABASE_ARG=""
                        if [ -z "${PHASE_1}" ]; then
                            DATABASE_ARG="--database-path ${BUILT_DATABASE}"
                        fi

                        python src/eval_multihop.py \
                            --model-path ${MODEL} \
                            ${DATABASE_ARG} \
                            --method ${METHOD} \
                            --max-tokens ${MAX_TOKENS} \
                            --batch-size ${BATCH_SIZE_LMLM} \
                            --total-count ${LIMIT} \
                            --output-dir ${OUTPUT_DIR}/ \
                            --save-version ${GRID_SAVE_VERSION} \
                            --split ${HF_SPLIT} \
                            --setting ${SETTING} \
                            --dataset ${DATASET} \
                            --seed ${SEED} \
                            --save-every ${SAVE_EVERY} \
                            ${START_INDEX_ARG} \
                            --top-k ${TOP_K} \
                            --similarity-threshold ${THRESHOLD} \
                            ${ADAPTIVE_K} \
                            ${RETURN_TRIPLETS} \
                            ${USE_INVERSES} \
                            ${PHASE_1} \
                            ${SUB_SPLIT_ARG_PHASE2} \
                            --eval 
                    else
                        if [ "${METHOD}" = "icl" ]; then
                            BATCH_SIZE=${BATCH_SIZE_ICL}
                        elif [ "${METHOD}" = "rag" ]; then
                            BATCH_SIZE=${BATCH_SIZE_RAG}
                        else
                            BATCH_SIZE=${BATCH_SIZE_DIRECT}
                        fi
                        # Build sub_split arg for non-LMLM methods
                        SUB_SPLIT_ARG_OTHER=""
                        START_INDEX_ARG_OTHER="--start-index ${START_IDX}"
                        if [ -n "${SUB_SPLIT}" ]; then
                            SUB_SPLIT_ARG_OTHER="--sub-split ${SUB_SPLIT}"
                            START_INDEX_ARG_OTHER=""  # Don't use start-index when sub_split is set
                        fi

                        python src/eval_multihop.py \
                            --method ${METHOD} \
                            --model ${LLM_MODEL} \
                            --max-tokens ${MAX_TOKENS} \
                            --batch-size ${BATCH_SIZE} \
                            --total-count ${NUM_SAMPLES} \
                            --output-dir ${OUTPUT_DIR}/ \
                            --split ${HF_SPLIT} \
                            --setting ${SETTING} \
                            --dataset ${DATASET} \
                            --seed ${SEED} \
                            --save-every ${SAVE_EVERY} \
                            ${START_INDEX_ARG_OTHER} \
                            ${SUB_SPLIT_ARG_OTHER} \
                            --eval 
                    fi
                done
            done
        done
    done
    done  # Close DATASET loop
else
    # Original single-run mode
    if [[ "${MODEL_PATH}" == *"-nak"* ]]; then
        ADAPTIVE_K=""
    else
        ADAPTIVE_K="--adaptive-k"
    fi

    # th-3
    if [[ "${MODEL_PATH}" == *"-th-3"* ]]; then
        RETURN_TRIPLETS="--return-triplets"
    else
        RETURN_TRIPLETS=""
    fi

    for METHOD in "${METHODS[@]}"; do
        echo "Running method: ${METHOD}"
        if [ "${METHOD}" = "lmlm" ]; then
            # Build sub_split arg only if SUB_SPLIT is set
            # When using sub_split, do NOT pass --start-index (sub_split handles indexing)
            SUB_SPLIT_ARG_PHASE2=""
            START_INDEX_ARG="--start-index ${START_IDX}"
            if [ -n "${SUB_SPLIT}" ]; then
                SUB_SPLIT_ARG_PHASE2="--sub-split ${SUB_SPLIT}"
                START_INDEX_ARG=""  # Don't use start-index when sub_split is set
            fi

            # Only pass --database-path if not in phase-1 mode
            DATABASE_ARG=""
            if [ -z "${PHASE_1}" ]; then
                DATABASE_ARG="--database-path ${DATABASE_PATH}"
            fi

            python src/eval_multihop.py \
                --model-path ${MODEL_PATH} \
                ${DATABASE_ARG} \
                --method ${METHOD} \
                --max-tokens ${MAX_TOKENS} \
                --batch-size ${BATCH_SIZE_LMLM} \
                --total-count ${NUM_SAMPLES} \
                --output-dir ${OUTPUT_DIR}/ \
                --save-version ${SAVE_VERSION} \
                --split ${SPLIT} \
                --setting ${SETTING} \
                --dataset ${DATASETS[0]} \
                --seed ${SEED} \
                --save-every ${SAVE_EVERY} \
                ${START_INDEX_ARG} \
                ${ADAPTIVE_K} \
                ${RETURN_TRIPLETS} \
                ${USE_INVERSES} \
                ${PHASE_1} \
                ${SUB_SPLIT_ARG_PHASE2} \
                --limit ${LIMIT} \
                --eval 
        else
            if [ "${METHOD}" = "icl" ]; then
                BATCH_SIZE=${BATCH_SIZE_ICL}
            elif [ "${METHOD}" = "rag" ]; then
                BATCH_SIZE=${BATCH_SIZE_RAG}
            else
                BATCH_SIZE=${BATCH_SIZE_DIRECT}
            fi
            # Build sub_split arg for non-LMLM methods in single-run mode
            SUB_SPLIT_ARG_OTHER=""
            START_INDEX_ARG_OTHER="--start-index ${START_IDX}"
            if [ -n "${SUB_SPLIT}" ]; then
                SUB_SPLIT_ARG_OTHER="--sub-split ${SUB_SPLIT}"
                START_INDEX_ARG_OTHER=""  # Don't use start-index when sub_split is set
            fi

            python src/eval_multihop.py \
                --method ${METHOD} \
                --model ${LLM_MODEL} \
                --max-tokens ${MAX_TOKENS} \
                --batch-size ${BATCH_SIZE} \
                --total-count ${NUM_SAMPLES} \
                --output-dir ${OUTPUT_DIR}/ \
                --split ${SPLIT} \
                --setting ${SETTING} \
                --dataset ${DATASETS[0]} \
                --seed ${SEED} \
                --save-every ${SAVE_EVERY} \
                ${START_INDEX_ARG_OTHER} \
                ${SUB_SPLIT_ARG_OTHER} \
                --eval 
        fi
    done
fi
