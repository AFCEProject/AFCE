"""Offline cache: frozen DINOv3 features, optional robot mask, E*, G*.

Training never runs DINOv3. Example:

    PYTHONPATH=/path/to/Dexjoco:$PYTHONPATH \\
      python -m effect_vla.scripts.cache_effect_targets \\
        --task water_plant --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from effect_vla.constants import (
    ACTION_HORIZON,
    ALL_TASKS,
    DEFAULT_CACHE_ROOT,
    DEFAULT_DINO_CKPT,
    DEFAULT_LEROBOT_ROOT,
    EFFECT_DIM,
    PROJECTION_SEED,
)
from effect_vla.data.dexjoco_window_dataset import DexJoCoWindowDataset
from effect_vla.data.effect_cache_dataset import EffectCacheStore
from effect_vla.data.grounding_targets import grounding_targets_episode
from effect_vla.effect.effect_target import effect_targets_episode, frozen_projection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", action="append", dest="tasks", help="Task id. Repeatable. Default: all 11.")
    p.add_argument("--lerobot-root", type=Path, default=Path(DEFAULT_LEROBOT_ROOT))
    p.add_argument("--cache-root", type=Path, default=Path(DEFAULT_CACHE_ROOT))
    p.add_argument("--dino-ckpt", type=Path, default=Path(DEFAULT_DINO_CKPT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--start-episode", type=int, default=0, help="Skip episodes with index < this.")
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all remaining after --start-episode.")
    p.add_argument("--generic-future", action="store_true", help="Ablation 2: skip correspondence/mask.")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    return p.parse_args()


def cache_task(args: argparse.Namespace, task: str) -> None:
    task_root = args.lerobot_root / task
    if not task_root.exists():
        raise FileNotFoundError(f"LeRobot task root not found: {task_root}")
    store = EffectCacheStore(args.cache_root, task)
    window = DexJoCoWindowDataset(task_root, task, action_horizon=args.action_horizon)

    from effect_vla.effect.dino_extractor import FrozenDINOv3

    dino = FrozenDINOv3(args.dino_ckpt, device=args.device)
    projection = frozen_projection(dino.hidden, EFFECT_DIM, PROJECTION_SEED)
    meta = {
        "task": task,
        "dino_ckpt": str(args.dino_ckpt),
        "hidden": int(dino.hidden),
        "effect_dim": EFFECT_DIM,
        "action_horizon": args.action_horizon,
        "generic_future": bool(args.generic_future),
        "projection_seed": PROJECTION_SEED,
        "patch_grid": int(dino.patch_grid()),
    }
    args.cache_root.joinpath(task).mkdir(parents=True, exist_ok=True)
    (args.cache_root / task / "meta.json").write_text(json.dumps(meta, indent=2))
    np.save(args.cache_root / task / "projection.npy", projection)

    indices = [ep for ep in window.episode_indices() if ep >= args.start_episode]
    if args.max_episodes > 0:
        indices = indices[: args.max_episodes]
    for ep in indices:
        if args.skip_existing and store.exists(ep):
            print(f"[{task}] skip episode {ep}")
            continue
        print(f"[{task}] episode {ep}")
        packed = window.load_episode(ep)
        features = dino.encode_numpy(packed["rgb"], batch_size=args.batch_size)
        t = min(features.shape[0], packed["state"].shape[0])
        features = features[:t]
        state = packed["state"][:t]
        # No simulator qpos in LeRobot → zero mask (no suppression) unless a
        # precomputed robot_mask.npz is already sitting in the episode folder.
        mask_path = store.episode_dir(ep) / "robot_mask.npz"
        robot_mask = None
        if mask_path.exists():
            with np.load(mask_path) as z:
                robot_mask = np.asarray(z["mask"], dtype=np.float32)[:t]
        effect_pack = effect_targets_episode(
            features,
            robot_mask,
            projection,
            action_horizon=args.action_horizon,
            generic_future=args.generic_future,
        )
        grounding = grounding_targets_episode(state, action_horizon=args.action_horizon)
        store.save_episode(
            ep,
            dino=features,
            robot_mask=robot_mask if robot_mask is not None else np.zeros(features.shape[:2], np.float32),
            effect=effect_pack["effect"],
            grounding=grounding,
            change=effect_pack["change"],
            keep=effect_pack["keep"],
            extra={"generic_future": np.asarray(args.generic_future)},
        )


def main() -> None:
    args = parse_args()
    tasks = tuple(args.tasks) if args.tasks else ALL_TASKS
    for task in tasks:
        cache_task(args, task)


if __name__ == "__main__":
    main()
