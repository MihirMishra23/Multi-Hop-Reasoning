#
# eval_lmlm_multihop.sh — Evaluate lmlm / two_phase / direct / icl / rag methods
#
# Usage:
#   bash scripts/eval_lmlm_multihop.sh --method lmlm --dataset hotpotqa --split dev
#
# All parameters can be overridden via CLI flags.
#

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1
LLM_MODEL=gpt-4
DATASET=hotpotqa
SPLIT=dev
NUM_SAMPLES=1000
SAVE_VERSION="default"
TOP_K=4
ADAPTIVE_K=""           # set to "--adaptive-k" to enable adaptive retrieval
USE_INVERSES=""         # set to "--use-inverses" to enable inverse relations
USE_TRAIN_PARAMS=""     # set to "--use-train-params" to use training sampling params (T=1.0)
CONCAT_ALL_DB=""        # set to "--concat-all-db" to build unified database (two_phase only)
USE_CONTEXTS="golden"   # "golden" | "all" (two_phase only)
SIMILARITY_THRESHOLD=0.6
METHODS=("lmlm")
OUTPUT_DIR=./output/main_tables
SETTING=${SETTING:-distractor}


# ── Parse CLI arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)       MODEL_PATH="$2";       shift 2 ;;
        --dataset)          DATASET="$2";           shift 2 ;;
        --method)           METHODS=("$2");         shift 2 ;;
        --llm_model)        LLM_MODEL="$2";        shift 2 ;;
        --split)            SPLIT="$2";             shift 2 ;;
        --num_samples)      NUM_SAMPLES="$2";       shift 2 ;;
        --top-k)            TOP_K="$2";             shift 2 ;;
        --adaptive-k)       ADAPTIVE_K="--adaptive-k"; shift ;;
        --use-inverses)     USE_INVERSES="--use-inverses"; shift ;;
        --use-train-params) USE_TRAIN_PARAMS="--use-train-params"; shift ;;
        --concat-all-db)    CONCAT_ALL_DB="--concat-all-db"; shift ;;
        --use-contexts)     USE_CONTEXTS="$2";      shift 2 ;;
        --similarity-threshold) SIMILARITY_THRESHOLD="$2"; shift 2 ;;
        --save_version)     SAVE_VERSION="$2";      shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";        shift 2 ;;
        --setting)          SETTING="$2";          shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── Dataset / split → database path mapping ──────────────────────────────────
if [ "${DATASET}" = "hotpotqa" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_validation_42_1000_all_context_database.json"
        START_IDX=0
    elif [ "${SPLIT}" = "train_val1k" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
        START_IDX=82347
        SPLIT="train"
    elif [ "${SPLIT}" = "train_train1k" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
        START_IDX=89347
        SPLIT="train"
    else
        echo "Error: hotpotqa SPLIT must be 'dev', 'train_val1k', or 'train_train1k', got '${SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "musique" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/musique_validation_42_1000_all_context_database.json"
        START_IDX=0
    else
        echo "Error: musique SPLIT must be 'dev', got '${SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "mquake" ] || [ "${DATASET}" = "mquake-remastered" ]; then
    if [ "${SPLIT}" = "eval-edit" ]; then
        DATABASE_PATH=/share/j_sun/lmlm_multihop/database/mquake-remastered_corrected/mquake_cf6334_mis_db.json
        START_IDX=0
    elif [ "${SPLIT}" = "eval-original" ]; then
        DATABASE_PATH=/share/j_sun/lmlm_multihop/database/mquake-remastered_corrected/mquake_cf6334_orig_db.json
        START_IDX=0
    elif [ "${SPLIT}" = "train" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/mquake-remastered/mquake_remastered_cf6334_database.json"
        START_IDX=0
    else
        echo "Error: mquake SPLIT must be 'eval-edit', 'eval-edit-new', 'eval-original', or 'train', got '${SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "two_wiki" ] || [ "${DATASET}" = "2wiki" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/as2637/database/2wiki_db.json"
        START_IDX=0
    else
        echo "Error: 2wiki SPLIT must be 'dev', got '${SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "synthworlds" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        START_IDX=0
    else
        echo "Error: synthworlds SPLIT must be 'dev', got '${SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "trivia_qa" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        START_IDX=0
    else
        echo "Error: trivia_qa SPLIT must be 'dev', got '${SPLIT}'"
        exit 1
    fi
else
    echo "Error: DATASET must be 'hotpotqa', 'musique', 'mquake', '2wiki', 'synthworlds', or 'trivia_qa', got '${DATASET}'"
    exit 1
fi

# ── Fixed eval parameters ────────────────────────────────────────────────────
MAX_TOKENS=1024
BATCH_SIZE_DIRECT=32
BATCH_SIZE_ICL=1
BATCH_SIZE_RAG=1
BATCH_SIZE_LMLM=64
SAVE_EVERY=64
SEED=42

# Auto-detect return-triplets mode from model path
RETURN_TRIPLETS=""
if [[ "${MODEL_PATH}" == *"-th-3"* ]]; then
    RETURN_TRIPLETS="--return-triplets"
fi

# ── Run evaluation ───────────────────────────────────────────────────────────
for METHOD in "${METHODS[@]}"; do
    echo "=========================================="
    echo "Method: ${METHOD} | Dataset: ${DATASET} | Split: ${SPLIT}"
    echo "Model:  ${MODEL_PATH}"
    echo "TOP_K=${TOP_K} ADAPTIVE_K='${ADAPTIVE_K}' USE_INVERSES='${USE_INVERSES}'"
    echo "=========================================="

    if [ "${METHOD}" = "lmlm" ]; then
        python src/eval_multihop.py \
            --model-path "${MODEL_PATH}" \
            --database-path "${DATABASE_PATH}" \
            --method "${METHOD}" \
            --max-tokens ${MAX_TOKENS} \
            --batch-size ${BATCH_SIZE_LMLM} \
            --total-count ${NUM_SAMPLES} \
            --output-dir "${OUTPUT_DIR}/" \
            --save-version "${SAVE_VERSION}" \
            --split "${SPLIT}" \
            --setting ${SETTING} \
            --dataset "${DATASET}" \
            --seed ${SEED} \
            --save-every ${SAVE_EVERY} \
            --start-index ${START_IDX} \
            --top-k ${TOP_K} \
            --similarity-threshold ${SIMILARITY_THRESHOLD} \
            ${ADAPTIVE_K} \
            ${RETURN_TRIPLETS} \
            ${USE_INVERSES} \
            --eval \
            --resume

    elif [ "${METHOD}" = "two_phase" ]; then
        python src/eval_multihop.py \
            --model-path "${MODEL_PATH}" \
            --method "${METHOD}" \
            --max-tokens ${MAX_TOKENS} \
            --batch-size ${BATCH_SIZE_LMLM} \
            --total-count ${NUM_SAMPLES} \
            --output-dir "${OUTPUT_DIR}/" \
            --save-version "${SAVE_VERSION}" \
            --split "${SPLIT}" \
            --setting ${SETTING} \
            --dataset "${DATASET}" \
            --seed ${SEED} \
            --save-every ${SAVE_EVERY} \
            --start-index ${START_IDX} \
            --top-k ${TOP_K} \
            --similarity-threshold ${SIMILARITY_THRESHOLD} \
            ${RETURN_TRIPLETS} \
            ${USE_INVERSES} \
            ${USE_TRAIN_PARAMS} \
            ${CONCAT_ALL_DB} \
            --use-contexts "${USE_CONTEXTS}" \
            --eval \
            --resume
    else
        # direct / icl / rag
        if [ "${METHOD}" = "icl" ]; then
            BATCH_SIZE=${BATCH_SIZE_ICL}
        elif [ "${METHOD}" = "rag" ]; then
            BATCH_SIZE=${BATCH_SIZE_RAG}
        else
            BATCH_SIZE=${BATCH_SIZE_DIRECT}
        fi
        python src/eval_multihop.py \
            --method "${METHOD}" \
            --model "${LLM_MODEL}" \
            --max-tokens ${MAX_TOKENS} \
            --batch-size ${BATCH_SIZE} \
            --total-count ${NUM_SAMPLES} \
            --output-dir "${OUTPUT_DIR}/" \
            --split "${SPLIT}" \
            --setting ${SETTING} \
            --dataset "${DATASET}" \
            --seed ${SEED} \
            --save-every ${SAVE_EVERY} \
            --start-index ${START_IDX} \
            --eval \
            --resume
    fi
done
