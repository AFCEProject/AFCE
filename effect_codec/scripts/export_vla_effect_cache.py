"""Encode AFCE v2.1 E* and write EffectCacheStore layout for π0.5 FM(E).

Writes under ``<out-root>/<task>/episode_XXXXXX/``:
  - effect_target.npz  with effect ∈ R^(T, H=30, 256)
  - grounding_target.npz  zeros (T, H, 42) so EffectCacheStore.exists() passes
    (Pi0EffectPolicy AE-FM path does not use grounding loss).

Example:
  PYTHONPATH=$PWD python -m effect_codec.scripts.export_vla_effect_cache \\
    --ckpt outputs/effect_codec/runs/afce_pilot/best.pt \\
    --task water_plant --task pick_bucket --task hammer_nail \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from effect_codec.constants import (
    ACTION_HORIZON,
    DEFAULT_AFCE_CACHE_ROOT,
    DEFAULT_PILOT_TASKS,
    EFFECT_DIM,
)
from effect_codec.model.codec import AFCECodec
from effect_vla.data.effect_cache_dataset import EffectCacheStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--task", action="append", dest="tasks", default=None)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_AFCE_CACHE_ROOT)
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Default: <cache-root.parent>/vla_cache",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--geo-dim", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def encode_episode(
    model: AFCECodec,
    windows: dict[str, np.ndarray],
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    """Return E* ∈ R^(T, H, 256) indexed by window start frame t."""
    t_idx = np.asarray(windows["t"], dtype=np.int64)
    n = int(t_idx.shape[0])
    if n == 0:
        raise ValueError("empty windows")
    t_max = int(t_idx.max())
    effect = np.zeros((t_max + 1, ACTION_HORIZON, EFFECT_DIM), dtype=np.float32)
    filled = np.zeros((t_max + 1,), dtype=np.bool_)

    keys = [
        "action",
        "state_t",
        "root_feat",
        "tip_feat",
        "joint_feat",
        "W_raw",
        "W_valid",
    ]
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        batch = {
            k: torch.from_numpy(np.asarray(windows[k][sl], dtype=np.float32)).to(device)
            for k in keys
        }
        e = model.encode(batch)["E"].detach().float().cpu().numpy()
        for i, ti in enumerate(t_idx[sl]):
            effect[int(ti)] = e[i]
            filled[int(ti)] = True
    # Forward-fill any missing frames (should be rare if windows are dense).
    last = None
    for ti in range(effect.shape[0]):
        if filled[ti]:
            last = effect[ti]
        elif last is not None:
            effect[ti] = last
    return effect


def export_task(
    model: AFCECodec,
    task: str,
    *,
    cache_root: Path,
    out_root: Path,
    device: torch.device,
    batch_size: int,
    skip_existing: bool,
    geo_dim: int,
) -> dict:
    store = EffectCacheStore(out_root, task)
    ep_dirs = sorted((cache_root / task).glob("episode_*"))
    done = 0
    skipped = 0
    for ep_dir in ep_dirs:
        ep = int(ep_dir.name.split("_")[1])
        win_path = ep_dir / "afce_windows.npz"
        if not win_path.exists():
            continue
        if skip_existing and store.exists(ep):
            skipped += 1
            continue
        with np.load(win_path) as z:
            windows = {k: z[k] for k in z.files}
        effect = encode_episode(model, windows, device, batch_size=batch_size)
        grounding = np.zeros((effect.shape[0], ACTION_HORIZON, geo_dim), dtype=np.float32)
        store.save_episode(ep, effect=effect, grounding=grounding)
        done += 1
        if done % 10 == 0 or done == 1:
            print(f"[{task}] exported {done} eps (last ep={ep} T={effect.shape[0]})")
    meta = {
        "task": task,
        "schema": "afce_vla_effect_cache",
        "effect_shape": [None, ACTION_HORIZON, EFFECT_DIM],
        "n_exported": done,
        "n_skipped": skipped,
    }
    out_root.joinpath(task).mkdir(parents=True, exist_ok=True)
    (out_root / task / "meta_effect_codec.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    args = parse_args()
    tasks = tuple(args.tasks) if args.tasks else DEFAULT_PILOT_TASKS
    out_root = args.out_root or (Path(args.cache_root).parent / "vla_cache")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = AFCECodec().to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    print(f"export → {out_root} device={device} tasks={tasks}")
    summaries = []
    for task in tasks:
        summaries.append(
            export_task(
                model,
                task,
                cache_root=Path(args.cache_root),
                out_root=out_root,
                device=device,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
                geo_dim=args.geo_dim,
            )
        )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
