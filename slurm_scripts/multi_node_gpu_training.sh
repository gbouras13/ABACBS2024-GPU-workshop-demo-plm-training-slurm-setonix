#!/bin/bash
######################### Batch Headers #########################
#SBATCH --partition=gpu-dev                                      # use partition `gpu` for GPU nodes
#SBATCH --account=director2187-gpu                               # IMPORTANT: use your own project and the -gpu suffix
#SBATCH --nodes=2                                            # 2 node
#SBATCH --ntasks-per-node=1                                   # NOTE: this needs to be `1` on SLURM clusters`
#SBATCH --time 0-01:59:30                                     # time limit for the job (up to 24 hours: `0-24:00:00`)
#SBATCH --job-name=plm-demo-single-node-8-gpus                        # job name
#SBATCH --output=J-%x.%j.out                                  # output log file
#SBATCH --error=J-%x.%j.err                                   # error log file
#SBATCH --exclusive                                           # need this
#################################################################

# Load required modules
module load pawseyenv/2024.05
module load singularity/4.1.0-slurm

# Define the container image path
export SINGULARITY_CONTAINER="/scratch/pawsey1018/gbouras/gpu_demo/ABACBS2024-GPU-workshop-demo-plm-training-slurm-setonix/pytorch2.4.1_rocm6.1.sif"

# controls number of threads for dataloader
export OMP_NUM_THREADS=8
NUM_PYTORCH_PROCESSES=8

# for GPU communication

RDZV_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RDZV_HOST
export RDZV_PORT=29400

# The compute node executing the batch script.
echo "Rendezvous Node IP: $RDZV_HOST"


# Run Singularity container
# easier to run ddp with torchrun

srun -c 64 --jobid "$SLURM_JOBID" singularity exec --cleanenv  "$SINGULARITY_CONTAINER" \
        torchrun \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $NUM_PYTORCH_PROCESSES \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
	train_esm.py --train_hdf5_file train.h5  --eval_hdf5_file val.h5 -o esm650m_train_multinode_node  --epochs 1 --batch_size 24 -t 8 --steps 17  --eval_steps 10 --ddp