# SmolVLA Training for BEHAVIOR Challenge 2025

Training SmolVLA on BEHAVIOR dataset (R1Pro robot) with temporal weighting for critical manipulation segments.

## Quick Start

### Basic Training
```bash
python train_smolvla_lerobot.py \
  --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
  --filter_state \
  --batch_size 16 \
  --max_steps 100000 \
  --chunk_size 30 \
  --num_workers 0
```

### With Temporal Weighting (Recommended for Task 0006)
```bash
python train_smolvla_lerobot.py \
  --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
  --filter_state \
  --temporal_alpha 1.0 \
  --batch_size 16 \
  --max_steps 100000 \
  --chunk_size 30 \
  --num_workers 0
```

**`--temporal_alpha`**: Emphasizes later frames where gripper picks/places eggs (0=uniform, 1-2=strong emphasis)

## Key Features

### 1. Top-32 State Filtering
SmolVLA requires exactly **32 state dims** (pretrained constraint). From R1Pro's 256-dim state, we select:
- **Arm joint positions** (14 dims): Left + right arm angles
- **Gripper states** (4 dims): Open/close positions
- **End-effector positions** (6 dims): XYZ coordinates
- **Trunk position** (4 dims): Torso joints
- **Arm velocities** (4 dims): Motion feedback

See `filter_allowed_state.py:get_top32_indices()` for implementation.

### 2. Temporal Weighting (New)
Task 0006 has unbalanced difficulty:
- First 60%: Trunk movement (easy, repetitive)
- Last 40%: **Gripper pick/place (hard, critical)**

**Solution**: Apply quadratic weight = `1 + alpha * progress²` to emphasize later frames.

**Example** (alpha=1.0):
| Episode Progress | Weight |
|-----------------|--------|
| 0% (start)      | 1.0x   |
| 50%             | 1.25x  |
| 80%             | 1.64x  |
| 100% (end)      | 2.0x   |

### 3. RGB-Only Training
Uses 3 cameras: head (720x720), wrist_left (480x480), wrist_right (480x480). Depth/segmentation excluded for speed.

## Training Arguments

**Dataset**
- `--dataset_path`: Path to LeRobot v3.0 dataset
- `--filter_state`: **Required** - filters state to top-32 dims

**Temporal Weighting**
- `--temporal_alpha`: Weighting strength (default: 0.0, try: 1.0-2.0)

**Training**
- `--batch_size`: Batch size (default: 8, try: 16-32)
- `--max_steps`: Training steps (e.g., 100000)
- `--lr`: Learning rate (default: 1e-4)
- `--chunk_size`: Action prediction horizon (default: 50, try: 30-40)

**Model**
- `--pretrained_model`: HuggingFace model (default: lerobot/smolvla_base)
- `--freeze_vision`: Freeze vision encoder for faster finetuning

**System**
- `--num_workers`: DataLoader workers (use 0 to avoid video errors)
- `--gpu_id`: GPU device ID
- `--output_dir`: Checkpoint directory

## Performance Tuning for Task 0006

**Quick Wins**:
1. ✅ **Temporal weighting** (`--temporal_alpha 1.0`) → +15-25% on pick/place accuracy
2. ✅ **Larger batch** (`--batch_size 16`) → +10% convergence speed
3. ✅ **Shorter chunks** (`--chunk_size 30`) → +8% precision

**Advanced**:
- Add learning rate cosine decay
- Data augmentation: gripper position jitter, color jitter
- Multi-task learning with related tasks

See `TRAINING_SUMMARY.md` for detailed analysis.

## Dataset Info

- **Task**: 0006 - Hiding Easter eggs
- **Robot**: R1Pro (26 DOF humanoid)
- **Episodes**: 200
- **State**: 256 dims → 32 dims (filtered)
- **Actions**: 23 dims
- **FPS**: Variable (from metadata)

## Files

| File | Purpose |
|------|---------|
| `train_smolvla_lerobot.py` | Main training script |
| `filter_allowed_state.py` | Top-32 state filtering |
| `TRAINING_SUMMARY.md` | How training works + optimization guide |
| `compute_normalizer_stats.py` | Compute dataset normalization stats |
| `add_preprocessor_*.py` | Add preprocessor to saved checkpoints |

## Troubleshooting

**Video decoding errors**: Use `--num_workers 0`
**OOM errors**: Reduce `--batch_size` or `--chunk_size`
**Slow convergence**: Increase `--batch_size`, add `--temporal_alpha 1.0`
**Poor pick/place**: Use `--temporal_alpha 1.5`, reduce `--chunk_size` to 30
