#!/usr/bin/env python3
"""
Compute and save normalization statistics for SmolVLA checkpoint.
This creates the missing policy_preprocessor_step_5_normalizer_processor.safetensors file.
"""

import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from filter_allowed_state import get_top32_indices


def main():
    parser = argparse.ArgumentParser(description="Compute normalization stats for SmolVLA")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to LeRobot dataset used for training")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Path to checkpoint directory where stats will be saved")
    parser.add_argument("--use_32_features", action="store_true",
                       help="Filter observation.state to top 32 features instead of all allowed features")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()

    print(f"Loading dataset metadata from: {dataset_path}")
    metadata = LeRobotDatasetMetadata(str(dataset_path))

    print(f"\nDataset stats available for features:")
    for key in metadata.stats.keys():
        print(f"  - {key}")

    # Prepare stats in the format expected by the policy preprocessor
    # Format: {feature_name}.{stat_type} -> tensor
    stats_dict = {}

    # Process each feature's stats
    for feature_name, feature_stats in metadata.stats.items():
        # Handle observation.state filtering
        if feature_name == "observation.state" and args.use_32_features:
            print(f"\nFiltering {feature_name} from 256 to 32 dimensions...")
            top32_indices = get_top32_indices()

            for stat_type, stat_value in feature_stats.items():
                if isinstance(stat_value, torch.Tensor):
                    # Filter to top 32 indices
                    filtered_value = stat_value[top32_indices]
                    stats_dict[f"{feature_name}.{stat_type}"] = filtered_value
                    print(f"  {stat_type}: {stat_value.shape} -> {filtered_value.shape}")
        else:
            # Keep all other features as-is
            for stat_type, stat_value in feature_stats.items():
                if isinstance(stat_value, torch.Tensor):
                    stats_dict[f"{feature_name}.{stat_type}"] = stat_value

    # Save stats file
    output_path = checkpoint_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    print(f"\nSaving normalization stats to: {output_path}")
    save_file(stats_dict, str(output_path))

    print(f"\n✓ Stats file created successfully!")
    print(f"  File: {output_path}")
    print(f"  Total tensors: {len(stats_dict)}")

    # Verify what was saved
    print(f"\nSaved stats for features:")
    feature_names = set(key.rsplit('.', 1)[0] for key in stats_dict.keys())
    for feature_name in sorted(feature_names):
        stat_types = [key.split('.')[-1] for key in stats_dict.keys() if key.startswith(feature_name + '.')]
        shape = stats_dict[f"{feature_name}.mean"].shape if f"{feature_name}.mean" in stats_dict else "N/A"
        print(f"  {feature_name}: {stat_types} (shape: {shape})")


if __name__ == "__main__":
    main()
