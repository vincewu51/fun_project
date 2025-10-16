FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer) and add to PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Clone the repositories
RUN git clone https://github.com/StanfordVL/b1k-baselines.git --recurse-submodules && \
    git clone https://github.com/StanfordVL/BEHAVIOR-1K.git

# Set environment variable for BEHAVIOR-1K path
ENV PATH_TO_BEHAVIOR_1K=/workspace/BEHAVIOR-1K

# Install baselines/openpi dependencies
WORKDIR /workspace/b1k-baselines/baselines/openpi
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Install BEHAVIOR-1K dependencies
WORKDIR ${PATH_TO_BEHAVIOR_1K}
RUN export PATH="/workspace/b1k-baselines/baselines/openpi/.venv/bin:$PATH" && \
    uv pip install -e bddl && \
    uv pip install -e OmniGibson[eval]

# Set the working directory back to /workspace
WORKDIR /workspace

# Activate the virtual environment by default
ENV VIRTUAL_ENV=/workspace/b1k-baselines/baselines/openpi/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["/bin/bash"]
