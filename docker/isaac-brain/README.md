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

1. **NVIDIA Docker Runtime** (nvidia-docker2)
   ```bash
   # Install nvidia-docker2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Verify GPU access in Docker**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
   ```

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

If you encounter GPU access issues:
1. Verify nvidia-docker2 is installed: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`
2. Check Docker daemon configuration includes nvidia runtime: `cat /etc/docker/daemon.json`
3. Restart Docker service: `sudo systemctl restart docker`
