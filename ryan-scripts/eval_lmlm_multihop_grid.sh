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
SPLIT=train_train1k
USE_INVERSES="--use-inverses" # or "--use-inverses"
NUM_SAMPLES=1000
SAVE_VERSION=""
METHODS=("lmlm")
DATASETS=( "hotpotqa" )
DATASETS=("2wiki"  "musique"  "hotpotqa"  )
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
MODEL_PATHS=(/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k)
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/ryan-march-sft-exp/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_classic_retrieval)
# Parse command line arguments
MODEL_PATHS=(/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_classic_retrieval_6k)
CUSTOM_DATABASE_PATH="phase1_database_Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_classic_retrieval_6k_hotpotqa_dev_2026-03-16_01-01-55.json"
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/ryan-march-sft-exp/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_classic_retrieval)
MODEL_PATHS=(/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-1314)
CUSTOM_DATABASE_PATH=""
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
BATCH_SIZE_LMLM=32
OUTPUT_DIR=./output
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
    echo "Total combinations: $((${#DATASETS[@]} * ${#MODEL_PATHS[@]} * ${#TOP_K_VALUES[@]} * ${#RETRIEVAL_THRESHOLD_VALUES[@]} * ${#METHODS[@]}))"
    echo "======================="

    for DATASET in "${DATASETS[@]}"; do
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
            elif [ "${SPLIT}" = "train_train1k" ]; then
                DEFAULT_NUM_SAMPLES=1000
                START_IDX=89347
                HF_SPLIT="train"
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

        # Build database for this model (Phase 1) or use custom database
        if [ -n "${CUSTOM_DATABASE_PATH}" ]; then
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
            BUILT_DATABASE="phase1_database_${MODEL_NAME}_${DATASET}_${SPLIT}_${TIMESTAMP}.json"

            echo ""
            echo "=========================================="
            echo "Phase 1: Building database for ${MODEL_NAME}"
            echo "Output: ${BUILT_DATABASE}"
            echo "=========================================="

            python scripts/run_phase_1.py \
                --model_path="${MODEL}" \
                --dataset="${DATASET}" \
                --split="${HF_SPLIT}" \
                --seed=${SEED} \
                --nb_examples=${NUM_SAMPLES} \
                --batch_size=${BATCH_SIZE_LMLM} \
                --output_file="${BUILT_DATABASE}"

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

                    # If USE_INVERSES is set, add 'use_inv' to the save version
                    if [ -n "${USE_INVERSES}" ]; then
                        GRID_SAVE_VERSION="${GRID_SAVE_VERSION}_use_inv"
                    fi

                    if [ "${METHOD}" = "lmlm" ]; then
                        python src/eval_multihop.py \
                            --model-path ${MODEL} \
                            --database-path ${BUILT_DATABASE} \
                            --method ${METHOD} \
                            --max-tokens ${MAX_TOKENS} \
                            --batch-size ${BATCH_SIZE_LMLM} \
                            --total-count ${NUM_SAMPLES} \
                            --output-dir ${OUTPUT_DIR}/ \
                            --save-version ${GRID_SAVE_VERSION} \
                            --split ${HF_SPLIT} \
                            --setting ${SETTING} \
                            --dataset ${DATASET} \
                            --seed ${SEED} \
                            --save-every ${SAVE_EVERY} \
                            --start-index ${START_IDX} \
                            --top-k ${TOP_K} \
                            --similarity-threshold ${THRESHOLD} \
                            ${ADAPTIVE_K} \
                            ${RETURN_TRIPLETS} \
                            ${USE_INVERSES} \
                            --eval \
                            --resume
                    else
                        if [ "${METHOD}" = "icl" ]; then
                            BATCH_SIZE=${BATCH_SIZE_ICL}
                        elif [ "${METHOD}" = "rag" ]; then
                            BATCH_SIZE=${BATCH_SIZE_RAG}
                        else
                            BATCH_SIZE=${BATCH_SIZE_DIRECT}
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
                            --start-index ${START_IDX} \
                            --eval \
                            --resume
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
            python src/eval_multihop.py \
                --model-path ${MODEL_PATH} \
                --database-path ${DATABASE_PATH} \
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
                --start-index ${START_IDX} \
                ${ADAPTIVE_K} \
                ${RETURN_TRIPLETS} \
                ${USE_INVERSES} \
                --eval \
                --resume
        else
            if [ "${METHOD}" = "icl" ]; then
                BATCH_SIZE=${BATCH_SIZE_ICL}
            elif [ "${METHOD}" = "rag" ]; then
                BATCH_SIZE=${BATCH_SIZE_RAG}
            else
                BATCH_SIZE=${BATCH_SIZE_DIRECT}
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
                --start-index ${START_IDX} \
                --eval \
                --resume
        fi
    done
fi
