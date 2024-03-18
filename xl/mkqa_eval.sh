#!/bin/bash
source ${HOME}/miniconda3/bin/activate pd 

export PROJECT_HOME=$(git rev-parse --show-toplevel)
cd ${PROJECT_HOME}
LANG="en"

export RESULTS_FILE=${PROJECT_HOME}/results/mkqa_eval_test_${LANG}.jsonl
export QAS_TO_ANSWER=${PROJECT_HOME}/data/annotated_datasets/data/mkqa/${LANG}.jsonl

python -m paq.retrievers.retrieve \
    --model_name_or_path ${PROJECT_HOME}/data/models/retrievers/retriever_multi_base_256 \
    --qas_to_answer ${QAS_TO_ANSWER} \
    --qas_to_retrieve_from ${PROJECT_HOME}/data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl \
    --top_k 10 \
    --output_file ${RESULTS_FILE} \
    --faiss_index_path ${PROJECT_HOME}/data/indices/multi_base_256_hnsw_sq8.faiss \
    --fp16 \
    --memory_friendly_parsing \
    --verbose

python -m paq.evaluation.eval_retriever \
    --predictions ${RESULTS_FILE} \
    --references ${QAS_TO_ANSWER} \
    --hits_at_k 1,5,10

