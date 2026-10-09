"""I/O for DINO reuse and AFCE v2.1 window caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from effect_afce_v21.constants import DEFAULT_AFCE_CACHE_ROOT, DEFAULT_DINO_CACHE_ROOT


class DinoCacheStore:
    def __init__(self, root: str | Path = DEFAULT_DINO_CACHE_ROOT):
        self.root = Path(root)

    def episode_dir(self, task: str, episode_index: int) -> Path:
        return self.root / task / f"episode_{episode_index:06d}"

    def load_dino(self, task: str, episode_index: int) -> np.ndarray:
        with np.load(self.episode_dir(task, episode_index) / "dino_front_features.npz") as z:
            return np.asarray(z["features"], dtype=np.float32)

    def load_robot_mask(self, task: str, episode_index: int) -> np.ndarray | None:
        path = self.episode_dir(task, episode_index) / "robot_mask.npz"
        if not path.exists():
            return None
        with np.load(path) as z:
            return np.asarray(z["mask"], dtype=np.float32)

    def list_episodes(self, task: str) -> list[int]:
        task_dir = self.root / task
        if not task_dir.exists():
            return []
        return [
            int(p.name.split("_")[1])
            for p in sorted(task_dir.glob("episode_*"))
            if (p / "dino_front_features.npz").exists()
        ]


class AFCECacheStore:
    def __init__(self, root: str | Path = DEFAULT_AFCE_CACHE_ROOT):
        self.root = Path(root)

    def task_dir(self, task: str) -> Path:
        return self.root / task

    def episode_dir(self, task: str, episode_index: int) -> Path:
        return self.task_dir(task) / f"episode_{episode_index:06d}"

    def window_path(self, task: str, episode_index: int) -> Path:
        return self.episode_dir(task, episode_index) / "afce_windows.npz"

    def exists(self, task: str, episode_index: int) -> bool:
        return self.window_path(task, episode_index).exists()

    def save_episode(self, task: str, episode_index: int, **arrays) -> Path:
        d = self.episode_dir(task, episode_index)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "afce_windows.npz"
        np.savez_compressed(path, **arrays)
        return path

    def save_json(self, task: str, name: str, payload: dict) -> None:
        d = self.task_dir(task)
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(json.dumps(payload, indent=2, sort_keys=True))
