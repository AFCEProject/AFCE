"""Download the public DexJoCo training inputs from Hugging Face.

The standard 11-task LeRobot dataset and the already-converted 44-D π0.5 base
checkpoint live in separate public repositories.  Download them concurrently
into a staging root so a partially transferred runtime directory is never
mistaken for a complete training input.
"""

from __future__ import annotations

import concurrent.futures
import pathlib

from huggingface_hub import snapshot_download
import tyro


def _download_dataset(output_root: pathlib.Path, max_workers: int) -> str:
    return snapshot_download(
        repo_id="DexJoCo/DexJoCo-Datasets-LeRobot",
        repo_type="dataset",
        allow_patterns="dexjoco_lerobot_datasets/**",
        local_dir=output_root / "dataset",
        max_workers=max_workers,
    )


def _download_checkpoint(output_root: pathlib.Path, max_workers: int) -> str:
    return snapshot_download(
        repo_id="DexJoCo/DexJoCo-Pi05",
        allow_patterns="pi05_base_action_dim_44/**",
        local_dir=output_root / "model",
        max_workers=max_workers,
    )


def main(output_root: pathlib.Path, max_workers: int = 8) -> None:
    if max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers}")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            "dataset": executor.submit(_download_dataset, output_root, max_workers),
            "checkpoint": executor.submit(_download_checkpoint, output_root, max_workers),
        }
        for name, future in futures.items():
            print(f"{name}: {future.result()}", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
