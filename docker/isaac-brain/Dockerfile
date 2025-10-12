# Use NVIDIA CUDA 12.6 base image with cuDNN on Ubuntu 22.04
# The host driver (CUDA 12.8) is backward compatible with container CUDA versions
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies including cmake, build-essential, and ffmpeg build dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    vim \
    curl \
    build-essential \
    cmake \
    python3-dev \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
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

# Accept Conda Terms of Service
RUN conda config --set solver libmamba && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create Python 3.11 environment named lerobot
RUN conda create -y -n lerobot python=3.11 && \
    conda clean -afy

# Install ffmpeg in the lerobot environment
RUN /bin/bash -c "source activate lerobot && conda install -y ffmpeg -c conda-forge && conda clean -afy"

# Activate environment by default
ENV PATH=/opt/conda/envs/lerobot/bin:$PATH
ENV CONDA_DEFAULT_ENV=lerobot
RUN echo "source activate lerobot" >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Clone and install LeRobot with aloha, pusht, and xarm environments
RUN git clone https://github.com/huggingface/lerobot.git && \
    cd lerobot && \
    /opt/conda/envs/lerobot/bin/pip install -e ".[aloha,pusht,xarm]"

# Install IsaacSim 5.0 from NVIDIA PyPI repository
RUN /opt/conda/envs/lerobot/bin/pip install --upgrade pip && \
    /opt/conda/envs/lerobot/bin/pip install 'isaacsim[all,extscache]==5.0.0' --extra-index-url https://pypi.nvidia.com

# Clone and install IsaacLab v2.2.0 (compatible with IsaacSim 5.0.0)
# Install from the source/isaaclab subdirectory
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    git checkout -f v2.2.0 && \
    /opt/conda/envs/lerobot/bin/pip install --upgrade pip && \
    /opt/conda/envs/lerobot/bin/pip install -e source/isaaclab

# Clone and install leisaac
RUN git clone https://github.com/LightwheelAI/leisaac.git && \
    cd leisaac && \
    /opt/conda/envs/lerobot/bin/pip install -e source/leisaac

# Clone and install Isaac-GR00T
RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git && \
    cd Isaac-GR00T && \
    /opt/conda/envs/lerobot/bin/pip install -e ".[base]" && \
    /opt/conda/envs/lerobot/bin/pip install --no-build-isolation flash-attn==2.7.1.post4

# Default command
CMD ["/bin/bash"]
