"""Windowed DexJoCo samples for offline cache / visualization (not the OpenPI loader)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from effect_vla.constants import ACTION_HORIZON, ANCHOR_FRACTIONS
from effect_vla.data.lerobot_io import (
    episode_front_video_path,
    iter_episode_rows,
    load_episode_parquet,
    read_video_frames,
    slice_episode_frames,
)


class DexJoCoWindowDataset:
    """Yields one current-frame window per (episode, t) for cache construction."""

    def __init__(
        self,
        task_root: str | Path,
        task: str,
        *,
        action_horizon: int = ACTION_HORIZON,
        anchor_fractions: tuple[float, ...] = ANCHOR_FRACTIONS,
    ):
        self.task_root = Path(task_root)
        self.task = task
        self.action_horizon = int(action_horizon)
        self.anchor_fractions = tuple(anchor_fractions)
        self.episodes = iter_episode_rows(self.task_root)
        self.offsets = [max(1, int(round(f * self.action_horizon))) for f in self.anchor_fractions]
        self._lerobot_ds = None

    def episode_indices(self) -> list[int]:
        return [int(r.get("episode_index", r.get("index", i))) for i, r in enumerate(self.episodes)]

    def load_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        lowdim = load_episode_parquet(self.task_root, episode_index)
        rgb = self._load_rgb(episode_index, expected_len=lowdim["length"])
        t = min(lowdim["state"].shape[0], rgb.shape[0])
        return {
            "rgb": rgb[:t],
            "state": lowdim["state"][:t],
            "length": t,
            "episode_index": episode_index,
        }

    def _load_rgb(self, episode_index: int, expected_len: int) -> np.ndarray:
        try:
            return self._load_rgb_lerobot(episode_index, expected_len)
        except Exception as exc:
            print(f"[warn] LeRobot video path failed ({exc}); falling back to raw mp4")
        video_path = episode_front_video_path(self.task_root, self.task, episode_index)
        frames = read_video_frames(video_path)
        return slice_episode_frames(frames, episode_index, self.episodes)

    def _load_rgb_lerobot(self, episode_index: int, expected_len: int) -> np.ndarray:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from effect_vla.data.lerobot_io import effect_camera_key

        if self._lerobot_ds is None:
            self._lerobot_ds = LeRobotDataset("local_repo", root=str(self.task_root))
        key = effect_camera_key(self.task)
        row = next(
            r
            for r in self.episodes
            if int(r.get("episode_index", r.get("index", -1))) == episode_index
        )
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        frames = []
        for i in range(start, end):
            sample = self._lerobot_ds[i]
            img = sample[key]
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            if np.issubdtype(arr.dtype, np.floating):
                if arr.max() <= 1.5:
                    arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            frames.append(arr)
        out = np.stack(frames, axis=0)
        if expected_len and out.shape[0] != expected_len:
            print(f"[warn] RGB length {out.shape[0]} != state length {expected_len} for episode {episode_index}")
        return out
