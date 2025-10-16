# B1K Environment Docker Setup

This Docker environment sets up Ubuntu 24.04 with the BEHAVIOR-1K baselines and dataset repositories.

## What's Included

- Ubuntu 24.04 base image
- Git installed
- Two cloned repositories:
  - `b1k-baselines` (with submodules)
  - `BEHAVIOR-1K`

## Build Instructions

```bash
docker build -t b1k-environment .
```

## Run Instructions

### Basic Run (CPU only)
```bash
docker run -it b1k-environment
```

### Run with GPU Support
```bash
docker run --gpus all -it b1k-environment
```

To specify specific GPUs:
```bash
docker run --gpus '"device=0,1"' -it b1k-environment
```

This will start an interactive bash session inside the container with both repositories available in `/workspace`.

### GPU Setup Prerequisites

To use GPU in Docker, you need NVIDIA Container Toolkit installed on the host machine:

```bash
# Add package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker
```

Verify GPU access inside container:
```bash
nvidia-smi
```

## Push to Docker Hub

1. Log in to Docker Hub:
```bash
docker login
```

2. Tag the image with your Docker Hub username:
```bash
docker tag b1k-environment <your-dockerhub-username>/b1k-environment:latest
```

3. Push the image to Docker Hub:
```bash
docker push <your-dockerhub-username>/b1k-environment:latest
```

## Pull from Another Machine

1. Log in to Docker Hub (if the image is private):
```bash
docker login
```

2. Pull the image:
```bash
docker pull <your-dockerhub-username>/b1k-environment:latest
```

3. Run the pulled image:
```bash
docker run -it <your-dockerhub-username>/b1k-environment:latest
```

## Repositories

- **b1k-baselines**: https://github.com/StanfordVL/b1k-baselines.git
- **BEHAVIOR-1K**: https://github.com/StanfordVL/BEHAVIOR-1K.git
