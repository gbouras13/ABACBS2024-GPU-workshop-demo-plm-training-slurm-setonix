# ABACBS 2024 GPU Workshop - Demo PLM Finetuning

* This repository contains some demonstration code on how one could finetune the ESM-2 650M protein language model using single GPUs, a single GPU node and multiple GPU nodes (in a pretty hacky manner :) ) . It was designed for the "GPUs in Bioinformatics" Workshop at the ABACBS 2024 conference in Sydney, Australia. 

## Installation

* These scripts require the following Python dependencies:

```
pip install h5py numpy loguru biopython transformers
```

* This script also requires a GPU compatible version of PyTorch to be installed
* If you are on a system with NVIDIA GPUs, this _should_ just work

```
pip install torch torchvision torchaudio
```

* If you are on Pawsey (AMD), I recommend building a container using the `Dockerfile` which will install the ROCm compatible PyTorch version suitable for Pawsey


## Usage

### Data Prep

* convert the FASTA to hdf5 to set up the data for the dataloader (this is already done for you)

```
python convert_fasta_to_hdf5.py -f train.fasta --hdf5_file train.h5
python convert_fasta_to_hdf5.py -f eval.fasta --hdf5_file eval.h5
```

### Running Data

#### Single GPU

```
python train_esm.py --train_hdf5_file train.h5  --eval_hdf5_file val.h5 -o esm650m_train_single_gpu \
        --epochs 1 --batch_size 24 -t 8 --steps 258  --eval_steps 20
```

#### Multiple GPUs (1 or more nodes)

```

# controls number of threads for dataloader
export OMP_NUM_THREADS=8
NUM_PYTORCH_PROCESSES=8

# for GPU-GPU multi-node communication
RDZV_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RDZV_HOST
export RDZV_PORT=29400

# The compute node executing the batch script.
echo "Rendezvous Node IP: $RDZV_HOST"
        torchrun \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $NUM_PYTORCH_PROCESSES \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
	train_esm.py --train_hdf5_file train.h5  --eval_hdf5_file val.h5 -o esm650m_train_single_node  --epochs 1 --batch_size 24 -t 8 --steps 33  --eval_steps 20 --ddp

```

* You can find SLURM job scripts (designed for Pawsey) in `slurm_scripts` - you will of course need to change to your pawsey project name

