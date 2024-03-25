#!/usr/bin/env bash
#SBATCH -J faiss_factory
#SBATCH -p cdtgpucluster
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --exclude=charles[11-19]
#SBATCH --cpus-per-task=28
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/%u/pd/logs/%A_%a.out
#SBATCH --error=/home/%u/pd/logs/%A_%a.out

# Activate Python Env
source ${HOME}/miniconda3/bin/activate pd

echo "$(which python)"
python -V

export EXP_CONFIG="$(cut -d ";" -f1 <<< "${LINE}")"

############################
# Job starts from here
############################

JOBID=${SLURM_ARRAY_JOB_ID}
TASKID=${SLURM_ARRAY_TASK_ID}
export LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" "$1")
EMBED_JOB_N="$(cut -d ";" -f1 <<< "${LINE}")"
EMBED_JOB_TOTAL="$(cut -d ";" -f2 <<< "${LINE}")"
echo "Embedding partition ${EMBED_JOB_N} of ${EMBED_JOB_TOTAL}"

export PROJECT_HOME=$(git rev-parse --show-toplevel)
cd ${PROJECT_HOME}
MODEL_NAME="google/mt5-base"
FAISS_NAME="PAQ_mt5_base_sq8"
QAS_TO_EMBED=${PROJECT_HOME}/data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl
EMBED_OUTPUT_DIR=${PROJECT_HOME}/data/embeddings/${FAISS_NAME}
FAISS_OUTPUT_DIR=${PROJECT_HOME}/data/indices
BSZ=1024
mkdir -p ${EMBED_OUTPUT_DIR}

python -m paq.retrievers.embed_partition \
    --model_name_or_path ${MODEL_NAME} \
    --qas_to_embed ${QAS_TO_EMBED} \
    --output_dir ${EMBED_OUTPUT_DIR} \
    --batch_size ${BSZ} \
    --verbose \
    --memory_friendly_parsing \
    --fp16 \
    --n_jobs ${EMBED_JOB_TOTAL} \
    --embed_job_n ${EMBED_JOB_N}

echo "Completed Index generation"
