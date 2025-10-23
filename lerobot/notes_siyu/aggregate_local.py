#!/usr/bin/env python3
from pathlib import Path
from aggregate import aggregate_datasets

# List of datasets in your xlerobot-data folder
repo_ids = [
    "xlerobot-candybar-001",
    "xlerobot-candybar-002",
    "xlerobot-candybar-003",
    "xlerobot-candybar-006",
    "xlerobot-candybar-007-remove-fourth",
    "xlerobot-candybar-rightarm-001",
    "xlerobot-candybar-rightarm-002",
    "xlerobot-candybar-basket-001",
    "xlerobot-candybar-basket-002",
    "xlerobot-candybar-basket-003",
    "xlerobot-candybar-basket-004",
    "xlerobot-candybar-basket-005-removelastone",
    "xlerobot-candybar-leftarm",
    "xlerobot-candybar-leftarm-desk2",
    "xlerobot-candybar-rightarm-desk2",
    "xlerobot-figbar-basket-001",
    "xlerobot-figbar-basket-002",
    "xlerobot-figbar-basket-003",
    "xlerobot-figbar-leftarm",
    "xlerobot-figbar-leftarm-desk2",
    "xlerobot-figbar-leftarm-desk2-001",
    "xlerobot-figbar-rightarm-desk2",
]

# All of them now live under the same directory
root_base = Path("/Users/siyulw/workspace/xlerobot-data")
roots = [root_base / rid for rid in repo_ids]

aggregate_datasets(
    repo_ids=repo_ids,
    roots=roots,
    aggr_repo_id="siyulw2025/xlerobot-data",  # final Hugging Face dataset name
    aggr_root=root_base / "AGGREGATED",       # local output directory
)
