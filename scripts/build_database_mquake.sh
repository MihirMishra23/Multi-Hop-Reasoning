#!/bin/bash
ROOT_DIR="$(dirname "${BASH_SOURCE[0]}")"/..

NB_EXAMPLES=1
USING_TRIPLES=original
DATABASE_SAVE_DIR=/share/j_sun/lmlm_multihop/database/mquake-remastered

python ${ROOT_DIR}/src/database_creation/build_database_mquake.py \
    --nb-examples ${NB_EXAMPLES} \
    --using-triples ${USING_TRIPLES} \
    --database-save-dir ${DATABASE_SAVE_DIR}