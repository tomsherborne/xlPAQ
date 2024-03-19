
numnodes=${SLURM_JOB_NUM_NODES}
numtasks=${SLURM_NTASKS}
mpi_tasks_per_node=$(echo "${SLURM_TASKS_PER_NODE}" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load cudnn/8.2.4.15-11.4/gcc-9.4.0-vuxt37p
#module load python/3.7
module load python-3.9.6-gcc-5.4.0-sbr552h

source /rds/user/hpcsher1/hpc-work/pd/pd/bin/activate

MODEL_NAME=
QAS_TO_EMBED=
OUTPUT_DIR=
BSZ=

python -m paq.retrievers.embed \
    --model_name_or_path ./data/models/retrievers/retriever_multi_base_256 \
    --qas_to_embed data/annotated_datasets/NQ-open.train-train.jsonl \
    --output_dir ./my_vectors_distributed \
    --fp16 \
    --batch_size 128 \
    --verbose \
    --memory_friendly_parsing \
    --n_jobs 10 \
    --slurm_partition my_clusters_partition \
    --slurm_comment "my embedding job"
    

python -m paq.retrievers.build_index \
    --embeddings_dir ./my_vectors_distributed \
    --output_path ./my_index.hnsw.faiss \
    --hnsw \
    --SQ8 \
    --store_n 32 \
    --ef_construction 128 \
    --ef_search 128 \
    --verbose
