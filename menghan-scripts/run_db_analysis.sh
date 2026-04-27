#!/bin/bash
# Run coverage_analysis.py and structure_analysis.py on every CSV in KG_results/
# Results saved to KG_results/analysis/<csv_stem>_coverage.txt and _structure.txt

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KG_RESULTS="${REPO_ROOT}/KG_results"
OUT_DIR="${KG_RESULTS}/analysis"
mkdir -p "${OUT_DIR}"

cd "${REPO_ROOT}"

for csv_file in "${KG_RESULTS}"/*.csv; do
    stem="$(basename "${csv_file}" .csv)"
    echo "===== Processing: ${stem} ====="

    echo "  [coverage] ..."
    python db_analysis/coverage_analysis.py --csv "${csv_file}" \
        > "${OUT_DIR}/${stem}_coverage.txt" 2>&1
    echo "  Saved: analysis/${stem}_coverage.txt"

    echo "  [structure] ..."
    python db_analysis/structure_analysis.py --csv "${csv_file}" \
        > "${OUT_DIR}/${stem}_structure.txt" 2>&1
    echo "  Saved: analysis/${stem}_structure.txt"

    echo ""
done

echo "===== All done. Results in ${OUT_DIR} ====="
