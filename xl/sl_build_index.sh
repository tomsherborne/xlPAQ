#!/usr/bin/env bash
#SBATCH -J faiss_factory
#SBATCH -A TEAMER-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/rds/user/%u/hpc-work/pd/logs/%A_%a.out
#SBATCH --error=/rds/user/%u/hpc-work/pd/logs/%A_%a.out

numnodes=${SLURM_JOB_NUM_NODES}
numtasks=${SLURM_NTASKS}
mpi_tasks_per_node=$(echo "${SLURM_TASKS_PER_NODE}" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load cudnn/8.2.4.15-11.4/gcc-9.4.0-vuxt37p
module load python-3.9.6-gcc-5.4.0-sbr552h

Activate Python Env
source ${HOME}/sp/pd/pd/bin/activate

echo "$(which python)"
python -V

## CSD Boilerplate
export OMP_NUM_THREADS=1
JOBID=${SLURM_ARRAY_JOB_ID}
TASKID=${SLURM_ARRAY_TASK_ID}
echo -e "JobID: ${JOBID}\n======"
echo -e "TaskID: ${TASKID}\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo -e "\nnumtasks=${numtasks}, numnodes=${numnodes}, mpi_tasks_per_node=${mpi_tasks_per_node} (OMP_NUM_THREADS=${OMP_NUM_THREADS})"

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
