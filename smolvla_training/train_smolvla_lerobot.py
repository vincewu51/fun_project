#!/usr/bin/env python3
"""
Train SmolVLA on BEHAVIOR Challenge dataset using LeRobot's native implementation.
This properly loads videos, handles language instructions, and uses the full VLA architecture.
"""

import argparse
from pathlib import Path
from datetime import datetime
import time
import shutil

import torch
import wandb

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# Import state filtering for BEHAVIOR standard track compliance
from filter_allowed_state import filter_state, filter_state_top32, get_allowed_indices, get_top32_indices


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA on BEHAVIOR Challenge dataset")

    # Dataset args
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to LeRobot v3.0 dataset")
    parser.add_argument("--repo_id", type=str, default=None,
                       help="Repository ID (for HF hub)")

    # Training args
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")

    # SmolVLA specific args
    parser.add_argument("--chunk_size", type=int, default=50,
                       help="Action chunk size (SmolVLA default: 50)")
    parser.add_argument("--pretrained_model", type=str, default="lerobot/smolvla_base",
                       help="Pretrained SmolVLA model to finetune from")
    parser.add_argument("--freeze_vision", action="store_true",
                       help="Freeze vision encoder during finetuning")

    # State filtering args (for BEHAVIOR standard track)
    parser.add_argument("--filter_state", action="store_true",
                       help="Filter state to 217 allowed dimensions")

    # Video loading args
    parser.add_argument("--video_tolerance", type=float, default=0.05,
                       help="Tolerance in seconds for video frame timestamp matching (default: 0.05)")

    # Other args
    parser.add_argument("--output_dir", type=str, default="./outputs/train/smolvla_behavior",
                       help="Directory to save checkpoints")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=10,
                       help="Keep only the last N checkpoints (default: 10, set to 0 to keep all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use for training (default: 0)")
    parser.add_argument("--wandb_project", type=str, default="behavior-smolvla",
                       help="Wandb project name")
    parser.add_argument("--wandb_disable", action="store_true",
                       help="Disable wandb logging")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if not args.wandb_disable:
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )

    print(f"\n{'='*80}")
    print(f"Training SmolVLA on BEHAVIOR Challenge Dataset")
    print(f"{'='*80}\n")

    # Device setup with GPU selection
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Device: {device} (GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device(args.device)
        print(f"Device: {device}")

    # Load dataset metadata
    print(f"\nLoading dataset metadata from {args.dataset_path}...")
    dataset_path_str = args.repo_id if args.repo_id else str(Path(args.dataset_path).expanduser())
    dataset_metadata = LeRobotDatasetMetadata(dataset_path_str)

    print(f"  FPS: {dataset_metadata.fps}")
    print(f"  Total episodes: {dataset_metadata.total_episodes}")
    print(f"  Total frames: {dataset_metadata.total_frames}")

    # Get features and prepare for SmolVLA
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}

    # Filter input features to ONLY include RGB cameras (exclude depth and segmentation)
    all_input_features = {key: ft for key, ft in features.items() if key not in output_features}
    input_features = {}
    for key, ft in all_input_features.items():
        # Only include RGB images and state features
        if "image" in key:
            if "rgb" in key:  # Only RGB cameras
                input_features[key] = ft
        else:
            # Include all non-image features (state, etc.)
            input_features[key] = ft

    print(f"\nInput features (RGB only):")
    for key, ft in input_features.items():
        print(f"  {key}: {ft.shape} ({ft.type})")

    print(f"\nOutput features:")
    for key, ft in output_features.items():
        print(f"  {key}: {ft.shape} ({ft.type})")

    # Configure SmolVLA
    print(f"\nConfiguring SmolVLA...")

    # Determine state dimension from input features
    state_dim = None
    for key, ft in input_features.items():
        if "state" in key and ft.shape and len(ft.shape) > 0:
            if state_dim is None:
                state_dim = ft.shape[0]
            else:
                state_dim = max(state_dim, ft.shape[0])

    # SmolVLA pretrained model uses max_state_dim=32
    # We MUST always use 32 to be compatible with pretrained weights
    # Therefore, filtering to top-32 is REQUIRED (not optional)
    max_state_dim = 32
    max_action_dim = 32

    if not args.filter_state:
        print("\n⚠️  WARNING: --filter_state not set, but it's REQUIRED for SmolVLA!")
        print("   Pretrained SmolVLA expects max_state_dim=32.")
        print("   Automatically enabling state filtering to top-32 dimensions.\n")
        args.filter_state = True

    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        freeze_vision_encoder=args.freeze_vision,
        optimizer_lr=args.lr,
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
    )

    print(f"  Chunk size: {cfg.chunk_size}")
    print(f"  Freeze vision encoder: {cfg.freeze_vision_encoder}")
    print(f"  Learning rate: {cfg.optimizer_lr}")

    # Initialize SmolVLA policy
    print(f"\nInitializing SmolVLA from {args.pretrained_model}...")
    try:
        # Try loading pretrained model
        policy = SmolVLAPolicy.from_pretrained(args.pretrained_model, config=cfg)
        print(f"  ✓ Loaded pretrained model")
    except Exception as e:
        print(f"  ⚠ Could not load pretrained model: {e}")
        print(f"  Creating policy from scratch...")
        policy = SmolVLAPolicy(cfg)

    policy.train()
    policy.to(device)

    # Create preprocessor and postprocessor
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Configure delta_timestamps for SmolVLA
    # SmolVLA uses current observation and predicts chunk_size future actions
    delta_timestamps = {}

    # Observations: just current frame - ONLY RGB images
    for key in input_features.keys():
        # Only include RGB images and state, skip depth and segmentation
        if "image" in key:
            if "rgb" in key:  # Only RGB cameras
                delta_timestamps[key] = [0.0]  # Current frame only
        elif "state" in key:
            delta_timestamps[key] = [0.0]  # Current frame only

    # Actions: current + future actions for chunk
    fps = dataset_metadata.fps
    delta_timestamps["action"] = [i / fps for i in range(args.chunk_size)]

    print(f"\nDelta timestamps configuration (RGB only):")
    for key, timestamps in delta_timestamps.items():
        print(f"  {key}: {len(timestamps)} frames ({timestamps[0]:.3f}s to {timestamps[-1]:.3f}s)")

    # Load dataset
    print(f"\nLoading dataset...")
    print(f"  Video frame tolerance: {args.video_tolerance}s")
    dataset = LeRobotDataset(
        dataset_path_str,
        delta_timestamps=delta_timestamps,
        tolerance_s=args.video_tolerance,
    )

    print(f"  Total frames available for training: {len(dataset)}")

    # Check a sample
    print(f"\nChecking dataset sample...")
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    print(f"\nDataLoader created:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num batches: {len(dataloader)}")
    print(f"  Num workers: {args.num_workers}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        betas=cfg.optimizer_betas,
        eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )

    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")

    if args.max_steps is None:
        print("ERROR: --max_steps must be specified")
        return

    step = 0
    done = False
    start_time = time.time()
    step_start_time = time.time()

    while not done:
        dataloader_iter = iter(dataloader)
        while True:
            # Try to load batch with error handling for corrupted frames
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            except (AssertionError, RuntimeError) as e:
                if "violate the tolerance" in str(e) or "video" in str(e).lower():
                    print(f"\n⚠️  Skipping corrupted batch at step {step}: {str(e)[:100]}...")
                    continue
                else:
                    raise

            # Preprocess batch
            batch = preprocessor(batch)

            # Filter state to top-32 dimensions if requested
            if args.filter_state and "observation.state" in batch:
                # batch["observation.state"] shape: [batch_size, n_obs_steps, 256]
                original_shape = batch["observation.state"].shape
                batch_size = original_shape[0]

                # Apply top-32 filtering to each observation in the batch
                filtered_states = []
                for i in range(batch_size):
                    # Get all observation steps for this batch item
                    obs_steps = batch["observation.state"][i]  # [n_obs_steps, 256]
                    filtered_obs = []
                    for obs in obs_steps:
                        filtered_obs.append(filter_state_top32(obs))
                    filtered_states.append(torch.stack(filtered_obs))

                batch["observation.state"] = torch.stack(filtered_states)
                if step == 0:
                    print(f"  Filtered state: {original_shape} → {batch['observation.state'].shape}")

            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            output = policy.forward(batch)
            # SmolVLA forward returns (loss, pred_actions) tuple
            if isinstance(output, tuple):
                loss = output[0]
            else:
                loss = output["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if cfg.optimizer_grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    cfg.optimizer_grad_clip_norm
                )

            optimizer.step()

            # Logging
            if step % 10 == 0:
                # Calculate timing info
                current_time = time.time()
                elapsed = current_time - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                sec_per_step = elapsed / (step + 1) if step > 0 else 0

                # Estimate remaining time
                remaining_steps = args.max_steps - step
                eta_seconds = remaining_steps * sec_per_step if step > 0 else 0
                eta_hours = eta_seconds / 3600

                # Format timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(f"[{timestamp}] Step {step}/{args.max_steps} | "
                      f"Loss: {loss.item():.4f} | "
                      f"{sec_per_step:.2f}s/step | "
                      f"ETA: {eta_hours:.1f}h")

                if not args.wandb_disable:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/step": step,
                        "train/sec_per_step": sec_per_step,
                        "train/steps_per_sec": steps_per_sec,
                    })

            # Save checkpoint
            if args.save_steps is not None and step > 0 and step % args.save_steps == 0:
                checkpoint_path = output_dir / f"checkpoint_step_{step}"
                policy.save_pretrained(checkpoint_path)
                print(f"  ✓ Saved checkpoint to {checkpoint_path}")

                # Delete old checkpoints if keeping only last N
                if args.keep_last_n_checkpoints > 0:
                    # Get all checkpoint directories (excluding final_checkpoint)
                    checkpoints = sorted(
                        [d for d in output_dir.glob("checkpoint_step_*") if d.is_dir()],
                        key=lambda x: int(x.name.split("_")[-1])
                    )

                    # Delete oldest checkpoints if we have more than N
                    if len(checkpoints) > args.keep_last_n_checkpoints:
                        for old_ckpt in checkpoints[:-args.keep_last_n_checkpoints]:
                            shutil.rmtree(old_ckpt)
                            print(f"  🗑️  Deleted old checkpoint: {old_ckpt.name}")

            step += 1

            # Check if done
            if step >= args.max_steps:
                done = True
                break

    # Save final checkpoint
    final_checkpoint_path = output_dir / "final_checkpoint"
    policy.save_pretrained(final_checkpoint_path)
    print(f"\n✓ Saved final checkpoint to {final_checkpoint_path}")

    if not args.wandb_disable:
        wandb.finish()

    print(f"\n{'='*80}")
    print(f"Training Completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
