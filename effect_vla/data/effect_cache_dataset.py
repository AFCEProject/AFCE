"""Map a LeRobot (episode, frame) sample onto cached E* / G*."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class EffectCacheStore:
    """On-disk cache: ``<root>/<task>/episode_XXXXXX/{dino,mask,effect,grounding}.npz``."""

    def __init__(self, cache_root: str | Path, task: str):
        self.root = Path(cache_root) / task
        self.task = task
        self._index: dict[int, dict[str, np.lib.npyio.NpzFile | np.ndarray]] = {}

    def episode_dir(self, episode_index: int) -> Path:
        return self.root / f"episode_{int(episode_index):06d}"

    def exists(self, episode_index: int) -> bool:
        d = self.episode_dir(episode_index)
        return (d / "effect_target.npz").exists() and (d / "grounding_target.npz").exists()

    def save_episode(
        self,
        episode_index: int,
        *,
        dino: np.ndarray | None = None,
        robot_mask: np.ndarray | None = None,
        effect: np.ndarray,
        grounding: np.ndarray,
        change: np.ndarray | None = None,
        keep: np.ndarray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        d = self.episode_dir(episode_index)
        d.mkdir(parents=True, exist_ok=True)
        if dino is not None:
            np.savez_compressed(d / "dino_front_features.npz", features=np.asarray(dino, dtype=np.float32))
        if robot_mask is not None:
            np.savez_compressed(d / "robot_mask.npz", mask=np.asarray(robot_mask, dtype=np.float32))
        payload = {"effect": np.asarray(effect, dtype=np.float32)}
        if change is not None:
            payload["change"] = np.asarray(change, dtype=np.float32)
        if keep is not None:
            payload["keep"] = np.asarray(keep)
        if extra:
            for k, v in extra.items():
                payload[k] = v
        np.savez_compressed(d / "effect_target.npz", **payload)
        np.savez_compressed(d / "grounding_target.npz", target=np.asarray(grounding, dtype=np.float32))
        # Also write the per-anchor names from the plan for debugging.
        if effect.ndim == 3 and effect.shape[1] >= 2:
            np.savez_compressed(d / "effect_target_anchor1.npz", effect=effect[:, 0])
            np.savez_compressed(d / "effect_target_anchor2.npz", effect=effect[:, 1])
        return d

    def load_effect(self, episode_index: int) -> np.ndarray:
        path = self.episode_dir(episode_index) / "effect_target.npz"
        with np.load(path) as z:
            return np.asarray(z["effect"], dtype=np.float32)

    def load_grounding(self, episode_index: int) -> np.ndarray:
        path = self.episode_dir(episode_index) / "grounding_target.npz"
        with np.load(path) as z:
            return np.asarray(z["target"], dtype=np.float32)

    def load_change(self, episode_index: int) -> np.ndarray | None:
        path = self.episode_dir(episode_index) / "effect_target.npz"
        with np.load(path) as z:
            if "change" not in z.files:
                return None
            return np.asarray(z["change"], dtype=np.float32)

    def get(self, episode_index: int, frame_index: int) -> dict[str, np.ndarray]:
        e = self.load_effect(episode_index)
        g = self.load_grounding(episode_index)
        t = int(np.clip(frame_index, 0, e.shape[0] - 1))
        return {
            "effect_target": e[t],
            "grounding_target": g[t],
        }


class EffectCacheDataset:
    """Torch-style dataset wrapper. Adds ``effect_target`` / ``grounding_target``.

    Compatible with OpenPI's ``create_torch_dataset``: wrap the LeRobot
    dataset *before* RepackTransform so those keys can be remapped.
    """

    def __init__(self, dataset, store: EffectCacheStore, *, required: bool = True):
        self.dataset = dataset
        self.store = store
        self.required = required

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        ep = int(np.asarray(sample["episode_index"]).reshape(-1)[0])
        frame = int(np.asarray(sample["frame_index"]).reshape(-1)[0])
        if not self.store.exists(ep):
            if self.required:
                raise FileNotFoundError(
                    f"Effect cache missing for {self.store.task} episode {ep}. "
                    "Run `python -m effect_vla.scripts.cache_effect_targets`."
                )
            return sample
        cached = self.store.get(ep, frame)
        sample = dict(sample)
        sample["effect_target"] = cached["effect_target"]
        sample["grounding_target"] = cached["grounding_target"]
        return sample
