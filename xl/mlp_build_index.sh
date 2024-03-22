#!/usr/bin/env bash
#SBATCH -J faiss_factory
#SBATCH -p PGR-Standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gresgpu:a40:2
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/rds/user/%u/hpc-work/pd/logs/%A_%a.out
#SBATCH --error=/rds/user/%u/hpc-work/pd/logs/%A_%a.out

# Activate Python Env
source ${HOME}/miniconda3/bin/activate pd

echo "$(which python)"
python -V

############################
# Job starts from here
############################

export PROJECT_HOME=$(git rev-parse --show-toplevel)
cd ${PROJECT_HOME}
MODEL_NAME="google/mt5-base"
FAISS_NAME="PAQ_mt5_base_sq8"
QAS_TO_EMBED=${PROJECT_HOME}/data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl
EMBED_OUTPUT_DIR=${PROJECT_HOME}/data/embeddings/${FAISS_NAME}
FAISS_OUTPUT_DIR=${PROJECT_HOME}/data/indices
BSZ=128
mkdir -p ${EMBED_OUTPUT_DIR}
mkdir -p ${FAISS_OUTPUT_DIR}

python -m paq.retrievers.embed \
    --model_name_or_path ${MODEL_NAME} \
    --qas_to_embed ${QAS_TO_EMBED} \
    --output_dir ${EMBED_OUTPUT_DIR} \
    --batch_size ${BSZ} \
    --verbose \
    --memory_friendly_parsing \
    --fp16 \
    --n_jobs -1 

python -m paq.retrievers.build_index \
    --embeddings_dir ${EMBED_OUTPUT_DIR} \
    --output_path ${FAISS_OUTPUT_DIR}/${FAISS_NAME}.faiss \
    --SQ8 \
    --verbose 

echo "Completed Index generation"
