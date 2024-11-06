FROM quay.io/pawsey/rocm-mpich-base:rocm6.1-mpich3.4.3-ubuntu22

ARG LIBFABRIC_VERSION=1.18.1

# Install required packages and dependencies
RUN   apt -y update \
      && apt -y install build-essential wget doxygen gnupg gnupg2 curl apt-transport-https software-properties-common  \
 git vim gfortran libtool python3-venv ninja-build \
      libnuma-dev python3-dev \
      && apt -y remove --purge --auto-remove cmake \
      && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null\
 | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
      && apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ jammy-rc main" \
      && apt -y update 

# Build and install libfabric
RUN (if [ -e /tmp/build ]; then rm -rf /tmp/build; fi;) \
      && mkdir -p /tmp/build \
      && cd /tmp/build \
      && wget https://github.com/ofiwg/libfabric/archive/refs/tags/v${LIBFABRIC_VERSION}.tar.gz \
      && tar xf v${LIBFABRIC_VERSION}.tar.gz \
      && cd libfabric-${LIBFABRIC_VERSION} \ 
      && ./autogen.sh \
      && ./configure \
      && make -j 16 \ 
      && make install

#
# Install base miniforge
#

RUN set -eux ; \
  curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh ; \
  bash ./Miniforge3-* -b -p /opt/miniforge3 -s ; \
  rm -rf ./Miniforge3-*
ENV PATH /opt/miniforge3/bin:$PATH
RUN conda clean -af -y

RUN pip install transformers biopython loguru pandas h5py numpy --no-cache-dir
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1 --no-cache-dir
RUN pip install pytorch-triton-rocm --no-cache-dir

RUN python3 -c "import torch; print(torch.__version__)" || echo "PyTorch is not installed before installing the project. "






