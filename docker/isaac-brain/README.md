# Docker Setup for isaac-brain Environment

This Docker setup replicates your exact conda environment with GPU support for AI model training.

## System Specifications
- **Base Image**: nvidia/cuda:12.8.0-cudnn9-devel-ubuntu24.04
- **Ubuntu**: 24.04.3 LTS (Noble Numbat)
- **CUDA**: 12.8
- **Python**: 3.11.13
- **Conda**: 25.9.0
- **GPU**: NVIDIA GeForce RTX 5080 (16GB)

## Prerequisites

### 1. Install Docker Engine

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Set up the Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index
sudo apt-get update

# Install Docker Engine
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group (to run docker without sudo)
sudo usermod -aG docker $USER

# Verify Docker installation
sudo docker run hello-world
```

**Note**: After adding yourself to the docker group, log out and log back in for the changes to take effect.

### 2. Install NVIDIA Container Toolkit

```bash
# Download and install the GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add the repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package index
sudo apt-get update

# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker service
sudo systemctl restart docker
```

### 3. Verify GPU Access in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

You should see your GPU information displayed.

## Building the Image

```bash
cd /home/yifeng/workspace/fun_project/docker/isaac-brain

# Build the Docker image (this will take some time)
docker build -t isaac-brain:latest .

# Or use docker-compose
docker-compose build
```

## Running the Container

### Option 1: Using docker-compose (Recommended)
```bash
docker-compose up -d
docker-compose exec isaac-brain bash
```

### Option 2: Using docker run
```bash
docker run --gpus all \
  --name isaac-brain-training \
  --shm-size=16g \
  -v /home/yifeng/workspace:/workspace \
  -v /home/yifeng/workspace/datasets:/data \
  -v /home/yifeng/workspace/models:/models \
  -it isaac-brain:latest bash
```

## Verify GPU Access Inside Container

Once inside the container:
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## Training Your Models

Your conda environment `isaac-brain` is activated by default. You can start training immediately:

```bash
# Your training script
python train.py

# Or with your specific IsaacLab/IsaacSim workflows
python -m isaaclab ...
```

## Volume Mounts

The docker-compose setup includes these volume mounts:
- `/home/yifeng/workspace` → `/workspace` (your code)
- `/home/yifeng/workspace/datasets` → `/data` (training data)
- `/home/yifeng/workspace/models` → `/models` (saved models)

Create the models directory if it doesn't exist:
```bash
mkdir -p /home/yifeng/workspace/models
```

## Useful Commands

```bash
# Start container
docker-compose up -d

# Access running container
docker-compose exec isaac-brain bash

# Stop container
docker-compose down

# View logs
docker-compose logs -f

# Remove container and image
docker-compose down --rmi all
```

## Notes

- The container uses `--shm-size=16gb` for PyTorch DataLoader workers
- Network mode is set to `host` for better performance
- All GPUs are accessible via `NVIDIA_VISIBLE_DEVICES=all`
- The environment.yml includes all your pip packages from isaac-brain

## Troubleshooting

### GPU Access Issues
If you encounter GPU access issues:
1. Verify NVIDIA Container Toolkit is installed: `dpkg -l | grep nvidia-container-toolkit`
2. Check Docker daemon configuration includes nvidia runtime: `cat /etc/docker/daemon.json`
3. Test GPU access: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`
4. Restart Docker service: `sudo systemctl restart docker`

### Repository Issues (Ubuntu 24.04)
If you see errors about corrupted repository files:
```bash
# Remove corrupted file
sudo rm /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Reinstall using the stable deb repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
```

### Permission Issues
If you get permission denied errors when running docker commands:
```bash
# Check if you're in the docker group
groups

# If docker is not listed, add yourself
sudo usermod -aG docker $USER

# Log out and log back in for changes to take effect
# Or run: newgrp docker
```
