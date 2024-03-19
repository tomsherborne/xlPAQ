#!/usr/bin/env bash
#SBATCH -J retriever
#SBATCH -A TEAMER-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH --output=/rds/user/%u/hpc-work/pd/logs/%A.out
#SBATCH --error=/rds/user/%u/hpc-work/pd/logs/%A.out

numnodes=${SLURM_JOB_NUM_NODES}
numtasks=${SLURM_NTASKS}
mpi_tasks_per_node=$(echo "${SLURM_TASKS_PER_NODE}" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load cudnn/8.2.4.15-11.4/gcc-9.4.0-vuxt37p
module load python-3.9.6-gcc-5.4.0-sbr552h

# Activate Python Env
source ${HOME}/sp/pd/pd/bin/activate

echo "$(which python)"
python -V

### CSD Boilerplate
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

export LANG=`sed \"${SLURM_ARRAY_TASK_ID}q;d\" $1`
echo "Running MKQA Evaluation for language ${LANG}"

export JOBNAME="multi_base_256_hnsw_sq8_MKQA_eval_${LANG}"
export RESULTS_FILE=${PROJECT_HOME}/results/${JOBNAME}.jsonl
export QAS_TO_ANSWER=${PROJECT_HOME}/data/annotated_datasets/data/mkqa/${LANG}.jsonl
export FAISS_INDEX=${PROJECT_HOME}/data/indices/multi_base_256_hnsw_sq8.faiss
export NQ_KB=${PROJECT_HOME}/data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl
export RETRIEVER=${PROJECT_HOME}/data/models/retrievers/retriever_multi_base_256
export TOPK=50

python -m paq.retrievers.retrieve \
    --model_name_or_path  ${RETRIEVER} \
    --qas_to_answer ${QAS_TO_ANSWER} \
    --qas_to_retrieve_from  ${NQ_KB} \
    --top_k ${TOPK} \
    --output_file ${RESULTS_FILE} \
    --faiss_index_path ${FAISS_INDEX} \
    --fp16 \
    --memory_friendly_parsing \
    --verbose

python -m paq.evaluation.eval_retriever \
    --predictions ${RESULTS_FILE} \
    --references ${QAS_TO_ANSWER} \
    --hits_at_k 1,10,50

echo "Completed MKQA Evaluation for language ${LANG}"
