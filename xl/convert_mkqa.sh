#!/bin/bash
source ${HOME}/miniconda3/bin/activate pd 
export PROJECT_HOME=$(git rev-parse --show-toplevel)

export SOURCE=${HOME}/ed/pd/mkqa.jsonl
export DEST_DIR=${HOME}/ed/pd/PAQ/data/annotated_datasets/data/mkqa

mkdir -p ${DEST_DIR}

# python ${PROJECT_HOME}/scratch/convert_mkqa_to_nq_format.py -h

# python ${PROJECT_HOME}/scratch/convert_mkqa_to_nq_format.py \
#     --source ${SOURCE} \
#     --dest ${DEST_DIR} \
#     --combine

python ${PROJECT_HOME}/xl/convert_mkqa_to_nq_format.py \
    --source ${SOURCE} \
    --dest ${DEST_DIR} \
    --combine \
    --use_alias \
    --nlang "bi"
