#!/usr/bin/env python3
"""
Add missing preprocessor/postprocessor files to an existing SmolVLA checkpoint.

This simplified version directly loads the dataset stats and creates the processor
without importing the full lerobot codebase.

Usage:
    python add_preprocessor_simple.py \
        --dataset_path ~/workspace/training_data/2025-challenge-demos-task0006 \
        --checkpoint_path ~/workspace/smolvla_training/outputs/train/smolvla_behavior/checkpoint_step_4200
"""

import argparse
import json
import sys
from pathlib import Path

import torch


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
    print(f"Adding Preprocessor/Postprocessor to Checkpoint (Simple Version)")
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

    # Load dataset stats from JSON
    print(f"\n📊 Loading dataset stats...")
    stats_json_file = dataset_path / "meta" / "stats.json"
    if not stats_json_file.exists():
        print(f"\n❌ Error: Stats file not found: {stats_json_file}")
        sys.exit(1)

    with open(stats_json_file) as f:
        stats_json = json.load(f)
    print(f"  ✓ Loaded stats from JSON")

    # Convert JSON stats to tensors
    stats = {}
    for key, value_dict in stats_json.items():
        if isinstance(value_dict, dict):
            # Stats are typically in format: {"min": [...], "max": [...], "mean": [...], "std": [...]}
            for stat_name, stat_value in value_dict.items():
                tensor_key = f"{key}.{stat_name}"
                stats[tensor_key] = torch.tensor(stat_value, dtype=torch.float32)
        else:
            # Direct value
            stats[key] = torch.tensor(value_dict, dtype=torch.float32)

    print(f"  ✓ Converted to {len(stats)} stat tensors")

    # Print some stats keys
    print(f"\n  Stats keys (first 10):")
    for i, key in enumerate(sorted(stats.keys())):
        if i >= 10:
            print(f"    ... and {len(stats) - 10} more")
            break
        print(f"    - {key}: {stats[key].shape}")

    # Load checkpoint config
    print(f"\n⚙️  Loading checkpoint config...")
    with open(config_file) as f:
        config_dict = json.load(f)

    print(f"  ✓ Loaded checkpoint config")
    print(f"    - Model type: {config_dict.get('model_type', 'unknown')}")

    # Get input and output features from config
    input_features = config_dict.get("input_features", {})
    output_features = config_dict.get("output_features", {})

    print(f"    - Input features: {len(input_features)}")
    for key in input_features:
        print(f"      • {key}")
    print(f"    - Output features: {len(output_features)}")
    for key in output_features:
        print(f"      • {key}")

    # Create preprocessor configuration
    # Based on lerobot.policies.smolvla.processor_smolvla.make_smolvla_pre_post_processors
    print(f"\n🔧 Creating preprocessor configuration...")

    preprocessor_config = {
        "name": "policy_preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {
                    "rename_map": {}  # Empty rename map
                }
            },
            {
                "registry_name": "device_processor",
                "config": {
                    "device": "cpu"  # Will be moved to device by policy
                }
            },
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "features": {**input_features, **output_features},
                    "norm_map": config_dict.get("normalization_mapping", {}),
                    "stats": None  # Will load from state file
                }
            }
        ]
    }

    # Create postprocessor configuration
    postprocessor_config = {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {
                    "features": output_features,
                    "norm_map": config_dict.get("normalization_mapping", {}),
                    "stats": None  # Will load from state file
                }
            }
        ]
    }

    print(f"  ✓ Created preprocessor config with {len(preprocessor_config['steps'])} steps")
    print(f"  ✓ Created postprocessor config with {len(postprocessor_config['steps'])} steps")

    # Save preprocessor config
    preprocessor_json = checkpoint_path / "policy_preprocessor.json"
    with open(preprocessor_json, 'w') as f:
        json.dump(preprocessor_config, f, indent=2)
    print(f"\n💾 Saved: {preprocessor_json.name}")

    # Save postprocessor config
    postprocessor_json = checkpoint_path / "policy_postprocessor.json"
    with open(postprocessor_json, 'w') as f:
        json.dump(postprocessor_config, f, indent=2)
    print(f"💾 Saved: {postprocessor_json.name}")

    # Save normalizer stats (for preprocessor step 5 - normalizer_processor)
    from safetensors.torch import save_file

    normalizer_stats_file = checkpoint_path / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    save_file(stats, str(normalizer_stats_file))
    print(f"💾 Saved: {normalizer_stats_file.name}")

    # Save unnormalizer stats (for postprocessor step 0 - unnormalizer_processor)
    unnormalizer_stats_file = checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    save_file(stats, str(unnormalizer_stats_file))
    print(f"💾 Saved: {unnormalizer_stats_file.name}")

    # Verify saved files
    print(f"\n✅ Verification: Checking saved files...")
    saved_files = sorted(checkpoint_path.glob("policy_*"))
    if saved_files:
        print(f"  ✓ Found {len(saved_files)} processor files:")
        for f in saved_files:
            size_mb = f.stat().st_size / 1024 / 1024
            if size_mb < 0.01:
                size_str = f"{f.stat().st_size / 1024:.2f} KB"
            else:
                size_str = f"{size_mb:.2f} MB"
            print(f"    - {f.name} ({size_str})")
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
