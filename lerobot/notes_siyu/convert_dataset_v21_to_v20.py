"""
This script converts a LeRobot dataset from codebase version 2.1 back to 2.0. It reverses the
changes introduced in ``convert_dataset_v20_to_v21.py`` by aggregating the per-episode statistics
into the legacy ``meta/stats.json`` file and updating the dataset metadata accordingly.

The script always downloads a fresh copy of the dataset metadata from the Hub into a user-provided
output directory before applying the conversion locally, and it never pushes updates back to the
Hub.

Typical usage:

```bash
python src/lerobot/datasets/v30/convert_dataset_v21_to_v20.py \
    --repo-id=siyulw2025/xlerobot-data \
    --output-dir=/tmp/lerobot_datasets \
    --delete-episodes-stats
```
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

# Add parent directories to path for imports
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import cast_stats_to_numpy, load_info, write_info, write_stats
from lerobot.utils.utils import init_logging

# Version constants
V20 = "v2.0"
V21 = "v2.1"

# Path constants - try to import from utils, fallback to hardcoded values
try:
    from lerobot.datasets.utils import EPISODES_STATS_PATH
except ImportError:
    try:
        from lerobot.datasets.utils import LEGACY_EPISODES_STATS_PATH as EPISODES_STATS_PATH
    except ImportError:
        EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"


def _load_episodes_stats_from_file(root: Path) -> dict[int, dict[str, Any]]:
    """Load per-episode stats from ``meta/episodes_stats.jsonl``.

    This is a lightweight fallback when ``LeRobotDatasetMetadata`` does not expose
    ``episodes_stats`` directly (e.g. on newer codebase versions).

    Args:
        root: The root directory of the dataset.

    Returns:
        A dictionary mapping episode indices to their statistics.

    Raises:
        FileNotFoundError: If the episodes_stats file cannot be found.
        ValueError: If the episodes_stats file is empty.
    """
    path = root / EPISODES_STATS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot locate '{EPISODES_STATS_PATH}' under '{root}'. The dataset must contain per-episode stats."
        )

    episodes_stats: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line in {EPISODES_STATS_PATH}: {line[:100]}... Error: {e}")
                continue

            ep_index = int(record["episode_index"])
            episodes_stats[ep_index] = cast_stats_to_numpy(record["stats"])

    if not episodes_stats:
        raise ValueError(f"'{EPISODES_STATS_PATH}' is empty or contains no valid records; cannot build legacy stats.json.")

    return episodes_stats


def _resolve_episodes_stats(root: Path) -> dict[int, dict[str, Any]]:
    """Attempt to load episode statistics using available methods.

    First tries to use the load_episodes_stats helper if available in older package versions,
    then falls back to loading directly from the file.

    Args:
        root: The root directory of the dataset.

    Returns:
        A dictionary mapping episode indices to their statistics.
    """
    # Try to rely on helper available in older package versions.
    try:
        from lerobot.datasets.utils import load_episodes_stats as _load_episodes_stats
    except ImportError:
        _load_episodes_stats = None

    if _load_episodes_stats is not None:
        try:
            loaded = _load_episodes_stats(root)
            if loaded:
                return loaded
        except (FileNotFoundError, OSError) as e:
            logging.debug(f"Could not load episodes stats using load_episodes_stats: {e}")

    return _load_episodes_stats_from_file(root)


def _remove_episode_stats_files(root: Path) -> None:
    """Remove the per-episode statistics file or directory.

    Args:
        root: The root directory of the dataset.
    """
    path = root / EPISODES_STATS_PATH
    if path.is_file():
        logging.info(f"Removing per-episode stats file: {path}")
        path.unlink()
    elif path.is_dir():
        logging.info(f"Removing per-episode stats directory: {path}")
        shutil.rmtree(path)
    else:
        logging.debug(f"No per-episode stats file found at {path}")


def _prepare_dataset_root(repo_id: str, output_dir: str) -> Path:
    """Prepare the output directory for the converted dataset.

    Creates the output directory if it doesn't exist and removes any existing
    dataset directory with the same name.

    Args:
        repo_id: The HuggingFace repository ID.
        output_dir: The base output directory.

    Returns:
        The path to the dataset root directory.
    """
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_root = output_path / repo_id.replace("/", "-")
    if dataset_root.exists():
        logging.info(f"Removing existing directory {dataset_root}")
        shutil.rmtree(dataset_root)

    return dataset_root


def convert_dataset(
    repo_id: str,
    output_dir: str,
    delete_episodes_stats: bool = False,
    download_media: bool = False,
) -> None:
    """Convert a LeRobot dataset from v2.1 to v2.0 format.

    Downloads the dataset from HuggingFace Hub, aggregates per-episode statistics
    into a single stats.json file, and updates the codebase version in info.json.

    Args:
        repo_id: The HuggingFace repository ID (e.g., "siyulw2025/xlerobot-data").
        output_dir: The directory where the converted dataset will be saved.
        delete_episodes_stats: If True, remove the per-episode stats file after conversion.
        download_media: If True, also download data files, videos, and images.

    Raises:
        ValueError: If the dataset is not v2.1 format.
        RuntimeError: If no per-episode stats are found.
    """
    dataset_root = _prepare_dataset_root(repo_id, output_dir)

    # Define what files to download
    allow_patterns = ["meta/**"]
    if download_media:
        allow_patterns.extend(["data/**", "videos/**", "images/**"])

    logging.info(f"Downloading dataset {repo_id} from HuggingFace Hub...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_root),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    # Verify the dataset version
    info = load_info(dataset_root)
    info_version = info.get("codebase_version")
    if info_version != V21:
        raise ValueError(
            f"Dataset '{repo_id}' is marked as '{info_version}', expected '{V21}'. "
            "This script only handles conversions from v2.1 to v2.0."
        )

    # Load per-episode statistics
    logging.info("Loading per-episode statistics...")
    episodes_stats = _resolve_episodes_stats(dataset_root)
    if not episodes_stats:
        raise RuntimeError("No per-episode stats found; cannot reconstruct legacy stats.json")

    # Aggregate statistics across all episodes
    logging.info(f"Aggregating statistics from {len(episodes_stats)} episodes...")
    aggregated_stats = aggregate_stats(list(episodes_stats.values()))

    # Write the aggregated stats to meta/stats.json
    logging.info("Writing aggregated statistics to meta/stats.json...")
    write_stats(aggregated_stats, dataset_root)

    # Update the codebase version
    info["codebase_version"] = V20
    write_info(info, dataset_root)
    logging.info(f"Updated codebase version from {V21} to {V20}")

    # Optionally remove per-episode stats file
    if delete_episodes_stats:
        _remove_episode_stats_files(dataset_root)

    logging.info(f"Conversion completed successfully. Output available at {dataset_root}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot dataset from v2.1 to v2.0 format."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face (e.g. `siyulw2025/xlerobot-data`).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Directory where the dataset will be downloaded and converted. Existing data in "
            "`<output-dir>/<repo-id>` is removed before download."
        ),
    )
    parser.add_argument(
        "--delete-episodes-stats",
        action="store_true",
        help="Remove the per-episode stats file after generating the legacy stats.json.",
    )
    parser.add_argument(
        "--download-media",
        action="store_true",
        help="Also download episode data (parquet), videos, and images alongside metadata.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    init_logging()
    args = parse_args()
    convert_dataset(**vars(args))
