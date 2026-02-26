#
# Setup instructions — eval_lmlm_multihop.sh
#
# Methods supported: direct, icl, rag, lmlm
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
#   bash scripts/eval_lmlm_multihop.sh --method direct
#
# Common overrides:
#   bash scripts/eval_lmlm_multihop.sh \
#     --method icl \
#     --llm_model gpt-4 \
#     --dataset 2wiki \
#     --split dev \
#     --num_samples 100
#
MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1
# MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop//Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed
# uncomment above to use two_phase model
LLM_MODEL=gpt-4
DATASET=hotpotqa
SPLIT=dev
USE_INVERSES="true" # or "--use-inverses"
USE_TRAIN_PARAMS=""   # set to "--use-train-params" to use grpo_train.sh sampling params instead of greedy
NUM_SAMPLES=1000
SAVE_VERSION="put-anything-here" #use this to add info to save path
TOP_K=4
METHODS=("lmlm")
# METHODS=("direct" "icl" "rag" "lmlm")
# uncomment above to eval on all methods
SIMILARITY_THRESHOLD=0.6


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
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
        --use-train-params)
            USE_TRAIN_PARAMS="--use-train-params"
            shift
            ;;
        --save_version)
            SAVE_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done


if [ "${DATASET}" = "hotpotqa" ]; then
    if [ "${SUB_SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_validation_42_1000_all_context_database.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
        SPLIT="dev"
    elif [ "${SUB_SPLIT}" = "dev-debug" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/generated_database_validation_42_1000.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
        SPLIT="dev"
    elif [ "${SUB_SPLIT}" = "train-val100" ]; then
        echo "Using eval set from GRPO"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
        DEFAULT_NUM_SAMPLES=100
        START_IDX=90347
        SPLIT="train"
    elif [ "${SUB_SPLIT}" = "train-val1k" ]; then
        echo "Using train set from GRPO"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=82347
        SPLIT="train"
    elif [ "${SUB_SPLIT}" = "train-val1k-debug" ]; then
        echo "Using train set from GRPO"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_all_context_exp_prompt.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=82347
        SPLIT="train"
    elif [ "${SUB_SPLIT}" = "train_train1k" ]; then
        echo "Using train set from GRPO"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=89347
        SPLIT="train"
    else
        echo "Error: SUB_SPLIT must be either 'train' or 'dev', got '${SUB_SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "musique" ]; then
    if [ "${SUB_SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/musique_validation_42_1000_all_context_database.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
        SPLIT="dev"
    elif [ "${SUB_SPLIT}" = "train" ]; then
        echo "There is no train database made for musique"
        exit 1
    else
        echo "Error: SUB_SPLIT must be either 'train' or 'dev', got '${SUB_SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "mquake" ] || [ "${DATASET}" = "mquake-remastered" ]; then
    if [ "${SUB_SPLIT}" = "eval-edit" ]; then
        # Evaluate with edited database against new_answer
        # edited gt database
        DATABASE_PATH="/share/j_sun/lz586/memgpt/dataset/mquake/mquake6334-all-gt-new-database-multi-hop.json"
        # DATABASE_PATH=/share/j_sun/lz586/memgpt/dataset/mquake/mquake6334-eval1k-gt-new-database-multi-hop.json
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
        SPLIT="test"
        ANSWER_TYPE="new_answer"
    elif [ "${SUB_SPLIT}" = "eval-edit-new" ]; then
        # Evaluate with edited database against new_answer
        # edited gt database
        DATABASE_PATH="/share/j_sun/lz586/memgpt/dataset/mquake/mquake6334-all-gt-edit-database-multi-hop.json"
        # DATABASE_PATH=/share/j_sun/lz586/memgpt/dataset/mquake/mquake6334-eval1k-gt-new-database-multi-hop.json
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
        SPLIT="test"
        ANSWER_TYPE="new_answer"
    elif [ "${SUB_SPLIT}" = "eval-original" ]; then
        # Evaluate with original database against original answer
        # gt database
        DATABASE_PATH="/share/j_sun/lz586/memgpt/dataset/mquake/mquake6334-all-gt-org-database-multi-hop.json"
        # DATABASE_PATH=/share/j_sun/lz586/memgpt/dataset/mquake/mquake6334-eval1k-gt-orig-database-multi-hop.json
        # DATABASE_PATH="/share/j_sun/lmlm_multihop/database/mquake-remastered/mquake_remastered_cf6334_database.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
        SPLIT="test"
        ANSWER_TYPE="answer"
    elif [ "${SUB_SPLIT}" = "train" ]; then
        # DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/mquake_remastered_cf6334_database.json"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/mquake-remastered/mquake_remastered_cf6334_database.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
        SPLIT="train"
        ANSWER_TYPE="answer"
    else
        echo "Error: SUB_SPLIT must be 'eval-edit', 'eval-original', or 'train' for mquake, got '${SUB_SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "two_wiki" ] || [ "${DATASET}" = "2wiki" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/as2637/database/2wiki_db.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
    else
        echo "Error: SPLIT must be either 'train' or 'dev', got '${SPLIT}'"
        exit 1
    fi
else
    echo "Error: DATASET must be one of 'hotpotqa', 'musique', or 'two_wiki', got '${DATASET}'"
    exit 1
fi

if [ "${NUM_SAMPLES}" -gt "${DEFAULT_NUM_SAMPLES}" ]; then
    NUM_SAMPLES="${DEFAULT_NUM_SAMPLES}"
fi

MAX_TOKENS=1024
BATCH_SIZE_DIRECT=32
BATCH_SIZE_ICL=1
BATCH_SIZE_RAG=1
BATCH_SIZE_LMLM=64
OUTPUT_DIR=./output
SETTING=distractor
SAVE_EVERY=64
SEED=42

# BUG: SFT model does not support adaptive k
if [[ "${MODEL_PATH}" == *"-nak"* || "${MODEL_PATH}" != *"grpo"* ]]; then
    ADAPTIVE_K=""
    TOP_K=1
else
    # GRPO model. consistent with training
    ADAPTIVE_K="--adaptive-k"
    TOP_K=4
fi

if [ -n "${USE_INVERSES}" ]; then
    USE_INVERSES="--use-inverses"
else
    USE_INVERSES=""
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
            --dataset ${DATASET} \
            --seed ${SEED} \
            --save-every ${SAVE_EVERY} \
            --start-index ${START_IDX} \
            ${ADAPTIVE_K} \
            ${RETURN_TRIPLETS} \
            ${USE_INVERSES} \
            --top-k ${TOP_K} \
            --similarity-threshold ${SIMILARITY_THRESHOLD} \
            --eval
    elif [ "${METHOD}" = "two_phase" ]; then
        echo "ignoring database path"
        python src/eval_multihop.py \
            --model-path ${MODEL_PATH} \
            --method ${METHOD} \
            --max-tokens ${MAX_TOKENS} \
            --batch-size ${BATCH_SIZE_LMLM} \
            --total-count ${NUM_SAMPLES} \
            --output-dir ${OUTPUT_DIR}/ \
            --save-version ${SAVE_VERSION} \
            --split ${SPLIT} \
            --setting ${SETTING} \
            --dataset ${DATASET} \
            --seed ${SEED} \
            --save-every ${SAVE_EVERY} \
            --start-index ${START_IDX} \
            ${RETURN_TRIPLETS} \
            ${USE_INVERSES} \
            --top-k ${TOP_K} \
            --similarity-threshold ${SIMILARITY_THRESHOLD} \
            ${USE_TRAIN_PARAMS} \
            --eval
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
            --dataset ${DATASET} \
            --seed ${SEED} \
            --save-every ${SAVE_EVERY} \
            --start-index ${START_IDX} \
            --eval \
            --resume
    fi
done