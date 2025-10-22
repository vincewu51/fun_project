#!/usr/bin/env python3
"""
Simplified script to combine multiple LeRobot datasets.

This script uses LeRobot's native API to properly combine datasets
while maintaining correct episode and chunk indexing.

Usage:
    python combine_lerobot_simple.py
"""

import logging
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def combine_datasets_properly(
    source_datasets: list[Path],
    output_repo_id: str,
    output_root: Path,
    task_name: str = "pickup candy with right arm"
):
    """
    Combine multiple LeRobot datasets using the proper API.

    Args:
        source_datasets: List of paths to source dataset directories
        output_repo_id: Repo ID for the combined dataset
        output_root: Root directory for the combined dataset
        task_name: Task description for all episodes
    """

    logger.info(f"Combining {len(source_datasets)} datasets...")

    # Load first dataset to get structure
    logger.info(f"Loading first dataset: {source_datasets[0]}")
    first_dataset = LeRobotDataset(
        repo_id=source_datasets[0].name,
        root=source_datasets[0].parent,
    )

    # Create new empty dataset with same structure
    logger.info(f"Creating output dataset: {output_repo_id}")
    combined_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=first_dataset.fps,
        root=output_root,
        features=first_dataset.meta.features,
        robot_type=first_dataset.meta.robot_type,
        use_videos=len(first_dataset.meta.video_keys) > 0,
    )

    # Process each source dataset
    total_episodes_added = 0

    for dataset_idx, dataset_path in enumerate(source_datasets):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset {dataset_idx + 1}/{len(source_datasets)}: {dataset_path.name}")
        logger.info(f"{'='*60}")

        # Load source dataset
        source_dataset = LeRobotDataset(
            repo_id=dataset_path.name,
            root=dataset_path.parent,
        )

        logger.info(f"  Episodes: {source_dataset.meta.total_episodes}")
        logger.info(f"  Frames: {source_dataset.meta.total_frames}")

        # Copy each episode
        for ep_idx in range(source_dataset.meta.total_episodes):
            logger.info(f"\n  Copying episode {ep_idx + 1}/{source_dataset.meta.total_episodes}...")

            # Get episode info
            ep_data = source_dataset.meta.episodes[ep_idx]
            ep_length = int(ep_data["length"])
            ep_start_idx = int(ep_data["dataset_from_index"])
            ep_end_idx = int(ep_data["dataset_to_index"])

            # Create episode buffer for new dataset
            combined_dataset.episode_buffer = combined_dataset.create_episode_buffer(
                episode_index=total_episodes_added
            )

            # Copy frames one by one
            for frame_idx in range(ep_start_idx, ep_end_idx):
                # Get frame from source
                frame_data = source_dataset.hf_dataset[frame_idx]

                # Build frame dict
                frame = {}
                frame["task"] = task_name

                # Add all features
                for key in source_dataset.features.keys():
                    if key in ["index", "episode_index", "task_index"]:
                        continue  # These are auto-generated

                    value = frame_data[key]

                    # Convert tensors to numpy
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()

                    frame[key] = value

                # Add timestamp
                if "timestamp" in frame_data:
                    timestamp = frame_data["timestamp"]
                    if isinstance(timestamp, torch.Tensor):
                        timestamp = timestamp.item()
                    frame["timestamp"] = timestamp

                # Add frame to combined dataset
                combined_dataset.add_frame(frame)

            # Save the complete episode
            logger.info(f"  Saving episode with {ep_length} frames...")
            combined_dataset.save_episode()

            total_episodes_added += 1
            logger.info(f"  ✓ Episode saved (total episodes: {total_episodes_added})")

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ COMBINATION COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Total episodes: {total_episodes_added}")
    logger.info(f"Total frames: {combined_dataset.meta.total_frames}")
    logger.info(f"Output location: {combined_dataset.root}")

    return combined_dataset


def upload_to_hub(dataset_root: Path, repo_id: str, push_videos: bool = True):
    """
    Upload the combined dataset to HuggingFace Hub.

    Args:
        dataset_root: Path to the combined dataset root
        repo_id: HuggingFace repository ID
        push_videos: Whether to upload videos (can be large)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"UPLOADING TO HUGGINGFACE HUB")
    logger.info(f"{'='*60}")

    # Load dataset
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root.parent)

    # Push to hub
    logger.info(f"Uploading to: {repo_id}")
    logger.info(f"Push videos: {push_videos}")

    dataset.push_to_hub(
        push_videos=push_videos,
        tags=["xlerobot", "manipulation", "robotics"],
        license="apache-2.0",
    )

    logger.info(f"\n✓ Successfully uploaded!")
    logger.info(f"View at: https://huggingface.co/datasets/{repo_id}")


def main():
    """Main entry point."""

    # ========== CONFIGURATION ==========
    # UPDATE THESE VALUES FOR YOUR SETUP!

    # Base directory containing your datasets
    BASE_DIR = Path("xlerobot-data")

    # List of dataset directory names to combine
    DATASET_NAMES = [
        "pickup_candy_rightarm001",
        "pickup_candy_rightarm002",
        # Add more as needed:
        # "pickup_candy_rightarm003",
    ]

    # Output configuration
    OUTPUT_REPO_ID = "your-username/pickup-candy-rightarm-xlerobot"
    OUTPUT_ROOT = Path("combined_datasets") / OUTPUT_REPO_ID.split("/")[-1]
    TASK_NAME = "pickup candy with right arm"

    # Upload configuration
    PUSH_VIDEOS = True  # Set False to skip videos (faster upload)

    # ===================================

    # Validate inputs
    dataset_paths = [BASE_DIR / name for name in DATASET_NAMES]

    for path in dataset_paths:
        if not path.exists():
            logger.error(f"Dataset not found: {path}")
            logger.error("Please check BASE_DIR and DATASET_NAMES")
            return

        # Verify it's a LeRobot dataset
        if not (path / "meta" / "info.json").exists():
            logger.error(f"Not a valid LeRobot dataset: {path}")
            logger.error("Missing meta/info.json")
            return

    logger.info("All source datasets found ✓")

    # Combine datasets
    combined_dataset = combine_datasets_properly(
        source_datasets=dataset_paths,
        output_repo_id=OUTPUT_REPO_ID,
        output_root=OUTPUT_ROOT,
        task_name=TASK_NAME
    )

    # Ask before uploading
    print(f"\n{'='*60}")
    print("Dataset combination complete!")
    print(f"{'='*60}")
    print(f"Output directory: {combined_dataset.root}")
    print(f"Repository ID: {OUTPUT_REPO_ID}")
    print(f"Push videos: {PUSH_VIDEOS}")
    print()

    response = input("Upload to HuggingFace Hub? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        upload_to_hub(
            dataset_root=combined_dataset.root,
            repo_id=OUTPUT_REPO_ID,
            push_videos=PUSH_VIDEOS
        )
    else:
        logger.info("\nSkipping upload.")
        logger.info("You can upload later with:")
        logger.info(f"  huggingface-cli upload {OUTPUT_REPO_ID} {combined_dataset.root} . --repo-type=dataset")


if __name__ == "__main__":
    main()
