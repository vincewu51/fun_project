# Use NVIDIA CUDA 12.8 base image with cuDNN on Ubuntu 24.04 (Noble)
# Matches your system: Ubuntu 24.04.3 LTS, CUDA 12.8, Driver 570.169
FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    vim \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (matching your conda 25.9.0)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

# Initialize conda
RUN conda init bash

# Copy the environment file (exported from isaac-brain with Python 3.11.13)
COPY environment.yml /tmp/environment.yml

# Create the conda environment from your exact environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy && \
    rm /tmp/environment.yml

# Activate environment by default
RUN echo "source activate isaac-brain" >> ~/.bashrc
ENV PATH /opt/conda/envs/isaac-brain/bin:$PATH
ENV CONDA_DEFAULT_ENV isaac-brain

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
