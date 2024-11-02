# ABACBS 2024 GPU Workshop - Demo PLM Finetuning

## Introduction

* These scripts require the following Python dependencies:

```
pip install h5py numpy loguru biopython transformers
```

* This script also requires a GPU compatible version of PyTorch to be installed
* If you are on a system with NVIDIA GPUs, this _should_ work

```
pip install torch torchvision torchaudio
```

* If you are on Pawsey, I recommend using the `Dockerfile` which will install the ROCm compatible PyTorch version suitable for Pawsey


## Usage

### Data Prep

```
python convert_fasta_to_hdf5.py -f train.fasta --hdf5_file train.h5
python convert_fasta_to_hdf5.py -f eval.fasta --hdf5_file eval.h5
```