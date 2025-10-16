# Fun Project Collection

A diverse collection of robotics, machine learning, computer vision, and data processing experiments and utilities. This repository serves as a personal workspace for exploring various technologies and implementing solutions across multiple domains.

## Overview

This repository contains multiple independent projects covering:
- **Robotics**: LeRobot integration, forward/inverse kinematics, Isaac Sim teleoperation
- **Machine Learning**: Neural network implementations, optimizer comparisons, W&B experiments
- **Computer Vision**: Photo editing tools, MMDiT implementations
- **Audio Processing**: Speech-to-text transcription and AI summarization
- **SAR Data Processing**: Sentinel-1 data exploration and analysis
- **Containerization**: Docker environments for GPU-accelerated AI workflows
- **Algorithms**: LeetCode solutions and algorithmic practice

## Project Structure

```
fun_project/
├── audio_transcribe/      # Audio transcription and AI summarization
├── basic_NN/              # Neural network experiments (Newton vs Adam optimizer)
├── docker/                # Docker configurations for development environments
│   └── isaac-brain/       # GPU-enabled Isaac Sim + IsaacLab environment
├── FK_IK/                 # Forward and inverse kinematics utilities
├── lerobot/               # LeRobot dataset conversion and teleoperation scripts
├── leetcode/              # Algorithm practice and solutions
├── modern_robotics/       # Modern Robotics course implementations
├── photo_editor/          # Photo editing using Nano Banana (MMDiT)
├── sar/                   # SAR (Synthetic Aperture Radar) data exploration
└── wandb/                 # Weights & Biases experiment tracking utilities
```

## Key Projects

### 1. Audio Transcription & Summarization (audio_transcribe/)
Complete pipeline for transcribing audio files and generating AI summaries using local models:
- Speech-to-text with OpenAI Whisper (all model sizes)
- AI summarization with local LLMs (Ollama/llama.cpp)
- Multiple output formats (TXT, JSON, SRT, VTT)
- Batch processing support
- Privacy-first: All processing done locally

**Quick start:**
```bash
cd audio_transcribe
pip install -r requirements.txt

# Transcribe audio
python transcribe.py audio.mp3

# Summarize transcript
python summarize.py output/transcripts/audio.txt

# Or use the complete pipeline
python process_audio.py audio.mp3 --model base --summary-style meeting
```

See [audio_transcribe/README.md](audio_transcribe/README.md) for detailed instructions.

### 2. Neural Network Training (basic_NN/)
Comparison of optimization algorithms for training modified AlexNet on CIFAR-10:
- Custom Newton optimizer (second-order)
- Adam optimizer (first-order)
- Training metrics and performance analysis

**Usage:**
```bash
python basic_NN/main.py
```

### 3. LeRobot Integration (lerobot/)
Scripts for working with LeRobot framework including:
- HDF5 dataset conversion from Isaac Sim to LeRobot format
- XLeRobot teleoperation setup and calibration
- Data upload utilities to Hugging Face
- Motor setup and calibration configurations

**Key workflows:**
- Motor setup: `lerobot-setup-motors`
- Calibration: `lerobot-calibrate`
- Teleoperation data collection with Isaac Sim
- Model inference with Gr00t and SmolVLA

See [lerobot/notes.md](lerobot/notes.md) for detailed instructions.

### 4. Docker Environment (docker/isaac-brain/)
GPU-accelerated Docker container for AI model training with:
- NVIDIA CUDA 12.8 + cuDNN 9
- Python 3.11 with conda environment
- Isaac Sim 5.0 and IsaacLab 2.2.0 support
- Optimized for RTX 5080 (16GB)

**Quick start:**
```bash
cd docker/isaac-brain
docker-compose up -d
docker-compose exec isaac-brain bash
```

See [docker/isaac-brain/README.md](docker/isaac-brain/README.md) for complete setup instructions.

### 5. SAR Data Exploration (sar/)
Workflow for downloading and analyzing Sentinel-1 SAR data:
- Query and download from ASF (Alaska Satellite Facility)
- Metadata extraction with pyroSAR
- Backscatter statistics and visualization
- Time series analysis potential

**Features:**
- VV-polarization processing
- Histogram generation
- Quick preview images
- Statistical analysis

See [sar/README.md](sar/README.md) for more details.

### 6. Forward/Inverse Kinematics (FK_IK/)
Utilities for robotic kinematics calculations:
- Closed-loop control utilities
- Robot checkpoint management
- Integration with LeRobot/Isaac Sim workflows

### 7. Photo Editor (photo_editor/)
Image editing tool using Nano Banana and MMDiT (Multimodal Diffusion Transformer):
- Custom MMDiT implementation
- Interactive editing interface
- Jupyter notebook for experimentation

### 8. Weights & Biases Utilities (wandb/)
Experiment tracking and model inference utilities:
- Multi-run experiment management
- Pi0 policy inference scripts
- Quick testing utilities

## Requirements

Core dependencies vary by project. See the following files for detailed requirements:

- **`environment.yml`**: Conda environment specification (recommended)
- **`requirements_full.txt`**: Complete pip requirements
- **`requirement.txt`**: SAR-specific dependencies
- **`SETUP.md`**: Detailed setup guide with troubleshooting

### Quick Setup

```bash
# Option 1: Conda (recommended)
conda env create -f environment.yml
conda activate fun_project

# Option 2: Pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_full.txt

# Option 3: Quick setup script
./setup.sh conda  # or 'pip' or 'docker'
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Development Environment

This repository uses:
- **Python**: 3.11+
- **CUDA**: 12.8 (for GPU workloads)
- **Docker**: For containerized development
- **Conda**: For environment management
- **Git**: Version control

## Git Workflow

Current branch: `main`

Recent activity:
- LeRobot notes and configurations
- Docker setup for Isaac Sim/IsaacLab environments
- Deployment updates
- Recording management features

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fun_project
   ```

2. **Choose your project:**
   Navigate to the specific project directory and follow its README or inline documentation.

3. **Set up dependencies:**
   Install requirements specific to your chosen project.

4. **Docker users:**
   For GPU-accelerated workflows, see `docker/isaac-brain/` for containerized environment setup.

## Project-Specific Documentation

- **Audio Transcription**: See [audio_transcribe/README.md](audio_transcribe/README.md)
- **Neural Networks**: See inline comments in `basic_NN/main.py`
- **LeRobot**: See [lerobot/notes.md](lerobot/notes.md) for detailed workflows
- **Docker**: See [docker/isaac-brain/README.md](docker/isaac-brain/README.md)
- **SAR**: See [sar/README.md](sar/README.md)
- **Photo Editor**: See [photo_editor/README.md](photo_editor/README.md)

## Notes

- Projects are mostly independent and can be used separately
- Some projects (LeRobot, FK_IK, wandb) may share dependencies
- Docker environment is optional but recommended for GPU workloads
- See individual project directories for specific usage instructions

## Contributing

This is a personal experimentation repository. Feel free to explore and adapt code for your own projects.

## License

Not specified. Please contact the repository owner for licensing information.
