#!/bin/bash
# Build the exact IRCoT Elasticsearch 7.10.2 indexes, then serve the released
# retrieval endpoint.  No GPU is requested because indexing/retrieval are CPU work.

#SBATCH --job-name=ircot_official_indexes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=default_partition
#SBATCH --output=/home/rtn27/ircot_index_service_%j.out
#SBATCH --error=/home/rtn27/ircot_index_service_%j.err

set -euo pipefail

ROOT="${IRCOT_ROOT:-/share/j_sun/rtn27/ircot_faithful}"
UPSTREAM="${IRCOT_UPSTREAM:-/share/j_sun/rtn27/ircot_upstream}"
ES_HOME="${ES_HOME:-${ROOT}/software/elasticsearch-7.10.2}"
ES_DATA="${ES_DATA:-${ROOT}/elasticsearch-data}"
ES_LOGS="${ES_LOGS:-${ROOT}/logs/elasticsearch}"
RETRIEVER_PORT="${RETRIEVER_PORT:-18000}"

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate multi-hop-reasoning

test "$(git -C "${UPSTREAM}" rev-parse HEAD)" = "3c1820f698eea5eeddb4fba3c56b64c961e063e4"
test -x "${ES_HOME}/bin/elasticsearch"
test -d "${ROOT}/data/raw_data"
mkdir -p "${ES_DATA}" "${ES_LOGS}" "${ROOT}/manifests"

export ES_JAVA_OPTS="-Xms24g -Xmx24g"
"${ES_HOME}/bin/elasticsearch" \
    -Epath.data="${ES_DATA}" \
    -Epath.logs="${ES_LOGS}" \
    -Ediscovery.type=single-node \
    -Enetwork.host=127.0.0.1 \
    -Ehttp.port=9200 &
es_pid=$!
retriever_pid=""
cleanup() {
    if [[ -n "${retriever_pid}" ]]; then kill "${retriever_pid}" 2>/dev/null || true; fi
    kill "${es_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:9200/_cluster/health >/dev/null; then break; fi
    sleep 2
done
curl -fsS http://127.0.0.1:9200/_cluster/health

cd "${ROOT}/data"
for dataset in 2wikimultihopqa musique hotpotqa; do
    expected=""
    case "${dataset}" in
        hotpotqa) expected=5233329 ;;
        2wikimultihopqa) expected=430225 ;;
        musique) expected=139416 ;;
    esac
    current=$(curl -fsS "http://127.0.0.1:9200/${dataset}/_count" 2>/dev/null \
        | python -c 'import json,sys; print(json.load(sys.stdin)["count"])' || echo 0)
    if [[ "${current}" == "${expected}" ]]; then
        echo "Reusing ${dataset} index with ${current} documents"
        continue
    fi
    if [[ "${current}" != "0" ]]; then
        echo "Refusing to replace ${dataset}: found ${current}, expected ${expected}" >&2
        exit 1
    fi
    python "${UPSTREAM}/retriever_server/build_index.py" "${dataset}"
    current=$(curl -fsS "http://127.0.0.1:9200/${dataset}/_count" \
        | python -c 'import json,sys; print(json.load(sys.stdin)["count"])')
    test "${current}" = "${expected}"
done

python "${SLURM_SUBMIT_DIR}/scripts/write_ircot_index_manifests.py" \
    --root "${ROOT}" --elasticsearch-url http://127.0.0.1:9200

uvicorn serve:app \
    --app-dir "${UPSTREAM}/retriever_server" \
    --host 0.0.0.0 \
    --port "${RETRIEVER_PORT}" &
retriever_pid=$!

hostname | tee "${ROOT}/manifests/retriever_hostname.txt"
printf '%s\n' "${RETRIEVER_PORT}" > "${ROOT}/manifests/retriever_port.txt"
wait "${retriever_pid}"
