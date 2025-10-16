#!/bin/bash
# Fun Project - Quick Setup Script
# Usage: ./setup.sh [conda|pip|docker]

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Fun Project - Environment Setup"
echo "=========================================="
echo ""

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check Python
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        print_success "Python found: $PYTHON_VERSION"
    else
        print_error "Python not found. Please install Python 3.11+"
        exit 1
    fi

    # Check CUDA (optional)
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_success "CUDA found: $CUDA_VERSION"
    else
        print_info "CUDA not found (optional for GPU support)"
    fi
}

# Setup with conda
setup_conda() {
    print_info "Setting up with Conda..."

    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Miniconda or Anaconda first."
        exit 1
    fi

    print_info "Creating conda environment from environment.yml..."
    conda env create -f environment.yml

    print_success "Conda environment 'fun_project' created successfully!"
    print_info "Activate with: conda activate fun_project"
}

# Setup with pip
setup_pip() {
    print_info "Setting up with pip..."

    # Create virtual environment
    print_info "Creating virtual environment..."
    python -m venv venv

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip

    # Install PyTorch with CUDA support
    print_info "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Install other requirements
    print_info "Installing other requirements..."
    pip install -r requirements_full.txt

    print_success "Pip environment created successfully!"
    print_info "Activate with: source venv/bin/activate"
}

# Setup with Docker
setup_docker() {
    print_info "Setting up with Docker..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi

    print_info "Building Docker image..."
    cd docker/isaac-brain
    docker-compose build

    print_success "Docker image built successfully!"
    print_info "Start container with: docker-compose up -d"
    print_info "Access container with: docker-compose exec isaac-brain bash"
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."

    # Test PyTorch
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" && \
    print_success "PyTorch working correctly"

    # Test other core packages
    python -c "import numpy; import matplotlib; import cv2" && \
    print_success "Core scientific packages working"

    # Test optional packages
    python -c "import h5py; import tqdm; import gradio" && \
    print_success "Optional packages working"
}

# Main setup logic
case "$1" in
    conda)
        check_prerequisites
        setup_conda
        print_info "Run 'conda activate fun_project' then run this script with 'verify' to test installation"
        ;;
    pip)
        check_prerequisites
        setup_pip
        verify_installation
        ;;
    docker)
        check_prerequisites
        setup_docker
        ;;
    verify)
        verify_installation
        ;;
    *)
        echo "Usage: $0 [conda|pip|docker|verify]"
        echo ""
        echo "Options:"
        echo "  conda   - Create conda environment from environment.yml"
        echo "  pip     - Create pip virtual environment and install requirements"
        echo "  docker  - Build Docker image for Isaac Sim/IsaacLab"
        echo "  verify  - Verify installation"
        echo ""
        echo "Examples:"
        echo "  ./setup.sh conda    # Setup with conda (recommended)"
        echo "  ./setup.sh pip      # Setup with pip"
        echo "  ./setup.sh docker   # Setup with Docker"
        exit 1
        ;;
esac

echo ""
print_success "Setup complete!"
