# SmolVLA Training for BEHAVIOR Challenge 2025

Training infrastructure for SmolVLA on BEHAVIOR Challenge dataset with R1Pro robot.

## 📁 Files

| File | Description |
|------|-------------|
| `train_smolvla_lerobot.py` | Main training script using LeRobot's SmolVLA |
| `filter_allowed_state.py` | State filtering utilities (top-32 dims) |
| `README.md` | This file |
| `outputs/` | Saved model checkpoints |

## 🚀 Quick Start

### Test run (fast validation)
```bash
python train_smolvla_lerobot.py \
  --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
  --filter_state \
  --max_steps 10 \
  --batch_size 4 \
  --num_workers 0 \
  --chunk_size 10 \
  --wandb_disable
```

### Full training
```bash
python train_smolvla_lerobot.py \
  --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
  --filter_state \
  --batch_size 16 \
  --max_steps 100000 \
  --save_steps 5000 \
  --chunk_size 50 \
  --num_workers 0 \
  --lr 1e-4 \
  --wandb_project "behavior-smolvla"
```

**Important:** Use `--num_workers 0` to avoid video decoding errors in multi-processing mode.

## 📊 Key Information

### Dataset
- **Path**: `~/workspace/training_data/2025-challenge-demos-task0006`
- **Robot**: R1Pro (26 DOF humanoid)
- **Episodes**: 200
- **Action dim**: 23
- **State dim**: 256 → filtered to 217 for standard track

### State Dimensions
- **Full state**: 256 dimensions (all R1Pro proprioception)
- **SmolVLA filtered**: 32 dimensions (top-32 most important for manipulation)
- **Standard track allowed**: 217 dimensions (excludes privileged information)

**Top-32 Selection:**
- Arm joint positions (both arms): 14 dims
- Gripper states: 4 dims
- End-effector positions: 6 dims
- Trunk position: 4 dims
- Arm velocities: 4 dims

### Cameras Used (RGB Only)
- Head camera: 720x720 RGB
- Left wrist camera: 480x480 RGB
- Right wrist camera: 480x480 RGB

**Note:** Depth and segmentation cameras are excluded from training to reduce computational overhead.

## 💡 Key Arguments

**Dataset:**
- `--dataset_path`: Path to v3.0 dataset
- `--episodes`: Comma-separated episode indices (e.g., "0,1,2,3")

**State Filtering:**
- `--filter_state`: Filter to 217 allowed dimensions (**REQUIRED for standard track**)
- `--state_dim`: Override detected state dimension

**Training:**
- `--batch_size`: Batch size (default: 8)
- `--num_epochs`: Number of epochs (required if --max_steps not set)
- `--max_steps`: Maximum training steps (overrides --num_epochs)
- `--save_steps`: Save checkpoint every N steps (if None, saves per epoch)
- `--lr`: Learning rate (default: 1e-4)
- `--action_horizon`: Action chunk size (default: 10)

**Logging:**
- `--wandb_project`: W&B project name
- `--wandb_run_name`: W&B run name
- `--wandb_tags`: Comma-separated tags
- `--wandb_disable`: Disable W&B logging

## 🔍 State Filtering Details

**Full R1Pro State (256 dims):**
- Joint positions, velocities, efforts
- End-effector poses
- Robot base state
- **Includes privileged information** (base position, global pose)

**SmolVLA Top-32 Selection:**
The pretrained SmolVLA model requires max_state_dim=32. We select the 32 most important dimensions for manipulation:

1. **Arm joint positions** (14 dims): Left + right arm joint angles
2. **Gripper states** (4 dims): Left + right gripper positions
3. **End-effector positions** (6 dims): Left + right EEF XYZ
4. **Trunk position** (4 dims): Torso joint angles
5. **Arm velocities** (4 dims): Joint velocities for feedback

**Standard Track Filtering (Alternative):**
For BEHAVIOR standard track compliance, `filter_state()` provides 217 allowed dimensions by excluding:
- Base joint positions (6 dims)
- Global robot position (3 dims)
- Global robot orientation (9 dims)
- Other privileged information (21 dims total)

See `filter_allowed_state.py` for complete mappings.

## 📦 Dataset Format

Uses **LeRobot v3.0** format:
- Parquet files in `data/chunk-XXX/file-XXX.parquet`
- Metadata in `meta/info.json`
- Custom loader bypasses variable-length field issues

## ✅ Verified Working

- ✅ SmolVLA integration with pretrained model (lerobot/smolvla_base)
- ✅ Video loading from all 3 RGB cameras (head, left wrist, right wrist)
- ✅ State filtering (256 → 32 dims, top-32 selection)
- ✅ Action chunking (configurable chunk_size, default 50)
- ✅ Training loop with vision-language-action model
- ✅ Wandb logging support
- ✅ Checkpoint saving

## 🎯 Next Steps

1. **Train on full dataset** (200 episodes, ~1.5M frames)
2. **Tune hyperparameters** (learning rate, batch size, chunk_size)
3. **Evaluate on BEHAVIOR Challenge** test set
4. **Fix multi-processing issue** with video decoder (currently requires num_workers=0)
