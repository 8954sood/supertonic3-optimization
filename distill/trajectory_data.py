"""Resolve published trajectory entries relative to the explicitly selected dataset."""

from pathlib import Path


def trajectory_path(root, row):
    split = row["split"]
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported trajectory split: {split}")
    return Path(root) / split / Path(row["file"]).name
