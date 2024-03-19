#!/bin/bash

# source ${HOME}/miniconda3/bin/activate pd 
source /rds/user/${USER}/hpc-work/pd/pd/bin/activate

export PROJECT_HOME=$(git rev-parse --show-toplevel)

# export SOURCE=${HOME}/ed/pd/mkqa.jsonl
export SOURCE=/rds/user/hpcsher1/hpc-work/pd/mkqa.jsonl
export DEST_DIR=${PROJECT_HOME}/data/annotated_datasets/data/mkqa

mkdir -p ${DEST_DIR}

python ${PROJECT_HOME}/xl/convert_mkqa_to_nq_format.py \
    --source ${SOURCE} \
    --dest ${DEST_DIR} \
    --combine \
    --nlang "bi"
