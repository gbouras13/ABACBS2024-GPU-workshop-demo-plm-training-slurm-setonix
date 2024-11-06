#!/bin/bash
######################### Batch Headers #########################
#SBATCH --partition=gpu-dev                                      # use partition `gpu` for GPU nodes
#SBATCH --account=director2187-gpu                              # IMPORTANT: use your own project and the -gpu suffix
#SBATCH --ntasks-per-node=1                                   # NOTE: this needs to be `1` on SLURM clusters`
#SBATCH --gres=gpu:1                                          # number of gpus you want
#SBATCH --time 0-01:59:30                                     # time limit for the job (up to 24 hours: `0-24:00:00`)
#SBATCH --job-name=plm-demo-single-gpu                # job name
#SBATCH --output=J-%x.%j.out                                  # output log file
#SBATCH --error=J-%x.%j.err                                   # error log file
#################################################################

# Load required modules
module load pawseyenv/2024.05
module load singularity/4.1.0-slurm

# Define the container image path
export SINGULARITY_CONTAINER="/scratch/pawsey1018/gbouras/gpu_demo/ABACBS2024-GPU-workshop-demo-plm-training-slurm-setonix/pytorch2.4.1_rocm6.1.sif"


# Run Singularity container
singularity exec --cleanenv  "$SINGULARITY_CONTAINER" \
	python train_esm.py --train_hdf5_file train.h5  --eval_hdf5_file val.h5 -o esm650m_train_single_gpu \
        --epochs 1 --batch_size 24 -t 8 --steps 258  --eval_steps 50
