"""Overlay high-weight Effect patches on ~50 front-camera frames.

Confirms that mass concentrates on task objects / interaction regions.
No quantitative analysis in v1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from effect_vla.constants import DEFAULT_CACHE_ROOT, DEFAULT_LEROBOT_ROOT
from effect_vla.data.dexjoco_window_dataset import DexJoCoWindowDataset
from effect_vla.data.effect_cache_dataset import EffectCacheStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--lerobot-root", type=Path, default=Path(DEFAULT_LEROBOT_ROOT))
    p.add_argument("--cache-root", type=Path, default=Path(DEFAULT_CACHE_ROOT))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--anchor", type=int, default=1, choices=(0, 1), help="0 = 0.5H, 1 = H")
    return p.parse_args()


def overlay_keep(rgb: np.ndarray, keep: np.ndarray, change: np.ndarray) -> np.ndarray:
    """rgb (H,W,3) uint8, keep/change (N,) on a square grid."""
    n = int(keep.shape[0])
    side = int(np.sqrt(n))
    if side * side != n:
        raise ValueError(f"Patch count {n} is not square")
    h, w = rgb.shape[:2]
    ph, pw = h // side, w // side
    heat = change.reshape(side, side)
    heat = heat / np.maximum(heat.max(), 1e-6)
    keep_g = keep.reshape(side, side)
    vis = rgb.astype(np.float32).copy()
    for i in range(side):
        for j in range(side):
            y0, x0 = i * ph, j * pw
            y1, x1 = min(h, y0 + ph), min(w, x0 + pw)
            if keep_g[i, j]:
                vis[y0:y1, x0:x1, 0] = np.clip(vis[y0:y1, x0:x1, 0] * 0.45 + 255 * heat[i, j] * 0.55, 0, 255)
                vis[y0:y1, x0:x1, 1] *= 0.45
                vis[y0:y1, x0:x1, 2] *= 0.45
    return vis.astype(np.uint8)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.cache_root / args.task / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    store = EffectCacheStore(args.cache_root, args.task)
    window = DexJoCoWindowDataset(args.lerobot_root / args.task, args.task)
    saved = 0
    for ep in window.episode_indices():
        if not store.exists(ep):
            continue
        packed = window.load_episode(ep)
        change = store.load_change(ep)
        with np.load(store.episode_dir(ep) / "effect_target.npz") as z:
            keep = np.asarray(z["keep"])
        t_len = min(packed["rgb"].shape[0], keep.shape[0])
        step = max(1, t_len // max(1, args.n - saved))
        for t in range(0, t_len, step):
            vis = overlay_keep(packed["rgb"][t], keep[t, args.anchor], change[t, args.anchor])
            path = out_dir / f"ep{ep:06d}_t{t:04d}_a{args.anchor}.png"
            _save_png(path, vis)
            saved += 1
            if saved >= args.n:
                print(f"Wrote {saved} overlays to {out_dir}")
                return
    print(f"Wrote {saved} overlays to {out_dir}")


def _save_png(path: Path, rgb: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
        return
    except ImportError:
        pass
    import imageio.v3 as iio

    iio.imwrite(path, rgb)


if __name__ == "__main__":
    main()
