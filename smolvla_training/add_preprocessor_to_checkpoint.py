#!/usr/bin/env python3
"""
Add missing preprocessor/postprocessor files to an existing SmolVLA checkpoint.

This script:
1. Loads the training dataset to extract normalization stats
2. Loads the checkpoint config to get input/output features
3. Creates preprocessor and postprocessor pipelines with the stats
4. Saves them to the checkpoint directory

Usage:
    python add_preprocessor_to_checkpoint.py \
        --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
        --checkpoint_path ~/workspace/smolvla_training/outputs/train/smolvla_behavior/checkpoint_step_4200
"""

import argparse
import sys
from pathlib import Path

# Add local lerobot to path
sys.path.insert(0, str(Path.home() / "workspace/lerobot/src"))

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser(description="Add preprocessor/postprocessor to existing checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to training dataset")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint directory")

    args = parser.parse_args()

    # Expand paths
    dataset_path = Path(args.dataset_path).expanduser()
    checkpoint_path = Path(args.checkpoint_path).expanduser()

    print(f"\n{'='*80}")
    print(f"Adding Preprocessor/Postprocessor to Checkpoint")
    print(f"{'='*80}\n")
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint: {checkpoint_path}")

    # Verify checkpoint exists and has config
    if not checkpoint_path.exists():
        print(f"\n❌ Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        print(f"\n❌ Error: No config.json found in checkpoint: {checkpoint_path}")
        sys.exit(1)

    model_file = checkpoint_path / "model.safetensors"
    if not model_file.exists():
        print(f"\n❌ Error: No model.safetensors found in checkpoint: {checkpoint_path}")
        sys.exit(1)

    print(f"\n✓ Checkpoint files found:")
    print(f"  - config.json")
    print(f"  - model.safetensors")

    # Load dataset metadata to get stats
    print(f"\n📊 Loading dataset metadata...")
    try:
        dataset_metadata = LeRobotDatasetMetadata(str(dataset_path))
        print(f"  ✓ Loaded dataset metadata")
        print(f"    - Total episodes: {dataset_metadata.total_episodes}")
        print(f"    - Total frames: {dataset_metadata.total_frames}")
        print(f"    - FPS: {dataset_metadata.fps}")
    except Exception as e:
        print(f"\n❌ Error loading dataset metadata: {e}")
        sys.exit(1)

    # Load checkpoint config
    print(f"\n⚙️  Loading checkpoint config...")
    try:
        config = SmolVLAConfig.from_pretrained(str(checkpoint_path))
        print(f"  ✓ Loaded checkpoint config")
        print(f"    - Input features: {len(config.input_features)}")
        print(f"    - Output features: {len(config.output_features)}")
        print(f"    - Max state dim: {config.max_state_dim}")
        print(f"    - Max action dim: {config.max_action_dim}")
        print(f"    - Chunk size: {config.chunk_size}")
    except Exception as e:
        print(f"\n❌ Error loading checkpoint config: {e}")
        sys.exit(1)

    # Create preprocessor and postprocessor with dataset stats
    print(f"\n🔧 Creating preprocessor and postprocessor...")
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            dataset_stats=dataset_metadata.stats
        )
        print(f"  ✓ Created preprocessor with {len(preprocessor.steps)} steps")
        print(f"  ✓ Created postprocessor with {len(postprocessor.steps)} steps")

        # Print preprocessor steps
        print(f"\n  Preprocessor steps:")
        for i, step in enumerate(preprocessor.steps):
            step_name = step.__class__.__name__
            print(f"    {i+1}. {step_name}")

        print(f"\n  Postprocessor steps:")
        for i, step in enumerate(postprocessor.steps):
            step_name = step.__class__.__name__
            print(f"    {i+1}. {step_name}")

    except Exception as e:
        print(f"\n❌ Error creating preprocessor/postprocessor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Check if preprocessor files already exist
    existing_processor_files = list(checkpoint_path.glob("policy_*.json"))
    if existing_processor_files:
        print(f"\n⚠️  Warning: Found existing processor files:")
        for f in existing_processor_files:
            print(f"    - {f.name}")
        response = input("\n  Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("\n  Aborted.")
            sys.exit(0)

    # Save preprocessor and postprocessor to checkpoint
    print(f"\n💾 Saving preprocessor and postprocessor to checkpoint...")
    try:
        preprocessor.save_pretrained(
            save_directory=checkpoint_path,
            config_filename="policy_preprocessor.json"
        )
        print(f"  ✓ Saved preprocessor")

        postprocessor.save_pretrained(
            save_directory=checkpoint_path,
            config_filename="policy_postprocessor.json"
        )
        print(f"  ✓ Saved postprocessor")

    except Exception as e:
        print(f"\n❌ Error saving preprocessor/postprocessor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Verify saved files
    print(f"\n✅ Verification: Checking saved files...")
    saved_files = sorted(checkpoint_path.glob("policy_*"))
    if saved_files:
        print(f"  ✓ Found {len(saved_files)} processor files:")
        for f in saved_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    - {f.name} ({size_mb:.2f} MB)")
    else:
        print(f"  ⚠️  Warning: No processor files found after saving!")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"✅ Successfully added preprocessor/postprocessor to checkpoint!")
    print(f"{'='*80}\n")
    print(f"Checkpoint is now ready for evaluation with:")
    print(f"  python src/lerobot/scripts/serve_smolvla_websocket.py \\")
    print(f"    --pretrained_name_or_path={checkpoint_path}")
    print()


if __name__ == "__main__":
    main()
