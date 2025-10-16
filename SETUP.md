# Environment Setup Guide

This guide provides instructions for setting up the development environment for this project.

## Prerequisites

- **Python**: 3.11 or higher (3.13 currently in use)
- **CUDA**: 12.1+ (for GPU support)
- **Conda/Miniconda**: For environment management
- **Git**: For version control

## Option 1: Conda Environment (Recommended)

### 1. Create environment from YAML file

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate fun_project
```

### 2. Verify installation

```bash
# Check Python version
python --version  # Should be 3.11.x

# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

## Option 2: Pip Installation

If you prefer pip over conda:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements_full.txt

# Install PyTorch with CUDA support (if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Option 3: Docker (For Isaac Sim/IsaacLab)

For GPU-accelerated robotics workflows with Isaac Sim:

```bash
cd docker/isaac-brain
docker-compose up -d
docker-compose exec isaac-brain bash
```

See [docker/isaac-brain/README.md](docker/isaac-brain/README.md) for detailed setup.

## Project-Specific Setup

### 1. SAR Data Processing

```bash
pip install asf_search pyroSAR spatialist
```

### 2. LeRobot (Robotics)

LeRobot has complex dependencies. Install from source:

```bash
# Clone LeRobot repository
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install in development mode
pip install -e .
```

**Note**: The project uses LeRobot v0.3.3. For compatibility:

```bash
cd lerobot
git checkout v0.3.3
pip install -e .
```

### 3. Weights & Biases

Set up W&B for experiment tracking:

```bash
# Login to W&B
wandb login

# Or set API key as environment variable
export WANDB_API_KEY="your_api_key_here"
```

### 4. Hugging Face Hub

For downloading models and datasets:

```bash
# Install huggingface-cli
pip install huggingface-hub[cli]

# Login to Hugging Face
huggingface-cli login

# Download a model (example)
huggingface-cli download --repo-type model siyulw2025/smolVLA_orange --local-dir ~/models/smolVLA_orange
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY="your_openai_api_key"
GEMINI_API_KEY="your_gemini_api_key"
WANDB_API_KEY="your_wandb_api_key"
HF_TOKEN="your_huggingface_token"

# CUDA Settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Isaac Sim (if applicable)
PHYSX_GPU_FOUND=1
PHYSX_USE_GPU=1
GPU_FORCE_64BIT_PTR=1
PHYSX_GPU_HEAP_SIZE=64
```

## Verification

### Test Core Functionality

```bash
# Test PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Test computer vision
python -c "import cv2; print(cv2.__version__)"

# Test data processing
python -c "import h5py; import numpy as np; print('h5py and numpy working')"

# Test web frameworks
python -c "import gradio; print(gradio.__version__)"

# Test SAR processing
python -c "import asf_search; print('SAR tools installed')"
```

### Test Project Components

```bash
# Test basic neural network
cd basic_NN
python -c "import torch; import torchvision; print('Neural network dependencies OK')"

# Test LeRobot (if installed)
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('LeRobot OK')"

# Test modern robotics
python -c "import modern_robotics as mr; print('Modern Robotics OK')"
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

```bash
# Set PyTorch memory allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Or reduce batch size in your training scripts
```

### LeRobot Import Errors

Ensure you're using the correct version:

```bash
cd lerobot
git log --oneline | head -5  # Check current commit
git checkout v0.3.3  # Switch to compatible version
pip install -e .
```

### Missing Dependencies

If a package is missing:

```bash
# For conda environment
conda activate fun_project
conda install <package-name>

# Or use pip
pip install <package-name>
```

## Updating the Environment

### Update all packages

```bash
# Conda
conda env update -f environment.yml --prune

# Pip
pip install -r requirements_full.txt --upgrade
```

### Export current environment

```bash
# Export conda environment
conda env export > environment.yml

# Export pip requirements
pip freeze > requirements_frozen.txt
```

## Development Tools

### Code Formatting

```bash
# Format code with black
black .

# Sort imports
isort .

# Lint with flake8
flake8 .
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Additional Resources

- **PyTorch**: https://pytorch.org/get-started/locally/
- **LeRobot**: https://github.com/huggingface/lerobot
- **Isaac Sim**: https://developer.nvidia.com/isaac-sim
- **Weights & Biases**: https://docs.wandb.ai/
- **Hugging Face**: https://huggingface.co/docs

## Support

For project-specific issues, see the README.md or individual project documentation in subdirectories.
