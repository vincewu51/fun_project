# BEHAVIOR-1K Repository Summary

## Overview

BEHAVIOR-1K is a comprehensive simulation benchmark for testing embodied AI agents on 1,000 everyday household activities. This monolithic repository provides everything needed to train and evaluate agents on human-centered tasks like cleaning, cooking, and organizing — activities selected from real human time-use surveys and preference studies.

**Main Website**: https://behavior.stanford.edu/

## System Requirements

- **OS**: Linux (Ubuntu 20.04+), Windows 10+
- **RAM**: 32GB+ recommended
- **VRAM**: 8GB+
- **GPU**: NVIDIA RTX 2080+

## Repository Structure

```
BEHAVIOR-1K/
├── OmniGibson/          # Core physics simulator and robotics environment
├── bddl/                # Behavior Domain Definition Language for task specification
├── joylo/               # JoyLo interface for robot teleoperation
├── asset_pipeline/      # DVC-based pipeline for 3D asset conversion (3ds Max to USD)
├── datasets/            # BEHAVIOR datasets (downloaded separately)
├── knowledgebase/       # Web interface for browsing BDDL knowledge base
├── docs/                # Documentation and website content
├── setup.sh             # Linux installation script
├── setup.ps1            # Windows installation script
└── mkdocs.yml           # Documentation configuration
```

## Core Components

### 1. OmniGibson (`/OmniGibson`)

The core physics simulator and robotics environment built on NVIDIA's Omniverse platform.

**Key Features**:
- Photorealistic visuals and physical realism
- Fluid and soft body simulation support
- Large-scale, high-quality scenes and objects
- Dynamic kinematic and semantic object states
- Mobile manipulator robots with modular controllers
- OpenAI Gym interface

**Main Directories**:
- `omnigibson/envs/` - Environment definitions
- `omnigibson/robots/` - Robot models and controllers
- `omnigibson/objects/` - Object models
- `omnigibson/object_states/` - Object state implementations (cooked, stained, etc.)
- `omnigibson/scenes/` - Scene definitions
- `omnigibson/controllers/` - Robot controller implementations
- `omnigibson/action_primitives/` - High-level action primitives
- `omnigibson/learning/` - Learning-related utilities
- `omnigibson/examples/` - Example scripts

**Key File**: `omnigibson/simulator.py` - Main simulator implementation

### 2. BDDL (`/bddl`)

Behavior Domain Definition Language - a predicate logic-based language for defining household activities.

**Purpose**: Defines each BEHAVIOR activity as a BDDL problem consisting of:
- `:objects` - Categorized object list
- `:init` - Initial conditions (ground literals)
- `:goal` - Goal conditions (logical expressions)

**Main Directories**:
- `bddl/activity_definitions/` - 1,000+ activity definitions (one directory per activity)
- `bddl/knowledge_base/` - Knowledge base models
- `bddl/generated_data/` - Generated metadata and mappings

**Key Files**:
- `bddl/activity.py` - Core activity condition handling
- `bddl/condition_evaluation.py` - Logic evaluator for goal conditions
- `bddl/object_taxonomy.py` - Object categorization and taxonomy
- `bddl/backend_abc.py` - Abstract backend interface for simulators

**Activity Structure**:
Each activity (e.g., `cleaning_the_pool`) has a `problem0.bddl` file defining objects, initial state, and goal conditions.

### 3. JoyLo (`/joylo`)

Interface for robot teleoperation using JoyLo hardware and Nintendo JoyCons.

**Purpose**: Enables human demonstration collection for imitation learning by teleoperating robots in OmniGibson.

**Hardware Support**:
- 6 DoF R1 JoyLo
- 7 DoF R1-Pro JoyLo
- Nintendo JoyCon controllers for navigation and control

**Key Features**:
- Real-time teleoperation in simulation
- Episode recording to HDF5 format
- Task-based data collection
- JoyCon button mapping for various functions

**Main Scripts**:
- `experiments/launch_nodes.py` - Launch recording environment
- `experiments/run_joylo.py` - Run JoyLo control node
- `scripts/calibrate_joints.py` - Joint calibration utility

### 4. Asset Pipeline (`/asset_pipeline`)

DVC-based pipeline for converting 3D assets from 3ds Max format to USD format.

**Purpose**: Process raw 3ds Max scene and object files into OmniGibson-compatible USD format.

**Requirements**: Windows 10/11, 3ds Max 2022, V-Ray 5 (Stanford-affiliated users only for raw assets)

**Main Directories**:
- `b1k_pipeline/` - Core pipeline scripts
  - `max/` - Scripts that run within 3ds Max
  - `usd_conversion/` - URDF to USD conversion scripts
- `artifacts/` - Pipeline output directory
  - `aggregate/` - Final dataset outputs
  - `og_dataset.zip` - Final packaged dataset
- `cad/` - Raw 3ds Max files (scenes and objects)
- `metadata/` - Metadata files and mappings

**Key Pipeline Stages**: Object listing, mesh export, URDF generation, USD conversion, validation

### 5. Knowledgebase (`/knowledgebase`)

Web application for browsing and exploring the BDDL knowledge base.

**Purpose**: Interactive dashboard for exploring tasks, scenes, objects, synsets, and other BDDL entities.

**Usage**:
```bash
# Generate static site (recommended)
pip install -r requirements_static.txt
python static_generator.py -o build

# Or run Flask dev server
pip install -r requirements.txt
flask --app knowledgebase.app run
```

**Key Files**:
- `knowledgebase/app.py` - Main Flask application
- `knowledgebase/views.py` - View classes (ListView, DetailView)
- `static_generator.py` - Static site generator

### 6. Datasets (`/datasets`)

Placeholder directory for BEHAVIOR datasets. Downloaded separately during setup using `--dataset` flag.

## Installation

### Quick Start (Recommended)

**Linux**:
```bash
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

**Windows**:
```powershell
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

### Installation Components

| Component | Flag | Description |
|-----------|------|-------------|
| OmniGibson | `--omnigibson` | Core simulator |
| BDDL | `--bddl` | Task specification language |
| JoyLo | `--joylo` | Teleoperation interface |
| Datasets | `--dataset` | Download BEHAVIOR datasets |
| New Env | `--new-env` | Create conda environment |
| Primitives | `--primitives` | Action primitives support |
| Eval | `--eval` | Evaluation support |

### Without Conda

Omit `--new-env` flag to use existing Python environment:
```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

## Navigation Guide

### Working with Activities/Tasks

1. **Browse available activities**: Check `bddl/bddl/activity_definitions/` - each subdirectory is one activity
2. **Understand an activity**: Read the `problem0.bddl` file in the activity's directory
3. **View activity in knowledgebase**: Run the knowledgebase web app and search for the task

### Working with Simulation

1. **Example scripts**: Check `OmniGibson/omnigibson/examples/`
2. **Robot control**: See `OmniGibson/omnigibson/robots/` and `OmniGibson/omnigibson/controllers/`
3. **Object states**: Check `OmniGibson/omnigibson/object_states/` for implementations of predicates (cooked, stained, etc.)
4. **Scenes**: Browse `OmniGibson/omnigibson/scenes/`

### Collecting Demonstrations with JoyLo

1. **Setup hardware**: Follow `joylo/README.md` for hardware configuration
2. **Connect JoyCons**: Use Bluetooth manager or bluetoothctl
3. **Start recording**:
   ```bash
   python experiments/launch_nodes.py --recording_path data.hdf5 --task_name <task>
   python experiments/run_joylo.py --gello_model r1pro --joint_config_file joint_config.yaml
   ```
4. **Available tasks**: Check `joylo/sampled_task/available_tasks.yaml`

### Understanding BDDL

1. **Activity structure**: Each activity has objects, initial conditions, and goal conditions
2. **Predicates**: Can be kinematic (ontop, inside, nextto) or non-kinematic (cooked, stained, frozen)
3. **Backend implementation**: See `OmniGibson/omnigibson/utils/bddl_utils.py` for OmniGibson's BDDL backend
4. **Object taxonomy**: Check `bddl/bddl/object_taxonomy.py`

### Documentation

- **Main docs**: `docs/` directory (built with MkDocs)
- **Build docs**: `mkdocs build` or `mkdocs serve`
- **Online docs**: https://behavior.stanford.edu/omnigibson/

## Key Concepts

### Activities
- 1,000 everyday household activities defined in BDDL format
- Each activity specifies required objects, initial state, and success criteria
- Process-agnostic: only specifies what state to achieve, not how

### Object States
- **Kinematic**: ontop, inside, nextto, touching, etc. (spatial relationships)
- **Non-kinematic**: cooked, stained, frozen, saturated, etc. (object properties)
- Implemented in `OmniGibson/omnigibson/object_states/`

### Scenes
- 50 large-scale interactive scenes representing different homes
- Each scene contains annotated objects and spatial structure
- Supports dynamic object placement and task sampling

### Robots
- Mobile manipulators with modular controller architecture
- Support for various action primitives (pick, place, navigate, etc.)
- Configurable via YAML files

## Development Workflow

### Running a Task in Simulation

1. Choose an activity from `bddl/bddl/activity_definitions/`
2. Check if pre-sampled task exists in datasets or generate new initial state
3. Load in OmniGibson environment
4. Control robot manually or with learned policy
5. Evaluate goal conditions using BDDL backend

### Creating New Activities

1. Write BDDL problem file with objects, init, and goal
2. Place in `bddl/bddl/activity_definitions/<activity_name>/problem0.bddl`
3. Ensure all required objects exist in object taxonomy
4. Test sampling and goal evaluation in OmniGibson

### Training Policies

1. Collect demonstrations using JoyLo or scripted policies
2. Save to HDF5 format
3. Use learning utilities in `OmniGibson/omnigibson/learning/`
4. Evaluate on sampled task instances

## Git Branches

- **main**: Development branch (current)
- **v3.7.1**: Latest stable release (recommended for most users)

## Important Notes

- Asset pipeline requires Windows and Stanford affiliation for raw 3ds Max files
- JoyLo requires Linux and specific hardware (JoyLo device, JoyCons, Bluetooth dongle)
- Datasets are large (multiple GB) and downloaded separately
- Documentation is built with MkDocs and served at behavior.stanford.edu
