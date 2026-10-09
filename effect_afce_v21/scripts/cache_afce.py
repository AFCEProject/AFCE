"""Build AFCE v2.1 caches: per-step ΔS + sparse world regions (no DINO re-run).

Example:
  PYTHONPATH=$PWD python -m effect_afce_v21.scripts.cache_afce \\
    --task water_plant --task pick_bucket --task hammer_nail --device cuda:0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from effect_afce_v21.constants import (
    ACTION_HORIZON,
    DEFAULT_AFCE_CACHE_ROOT,
    DEFAULT_DINO_CACHE_ROOT,
    DEFAULT_LEROBOT_ROOT,
    DEFAULT_PILOT_TASKS,
    NUM_WORLD_INTERVALS,
    SCHEMA_VERSION,
)
from effect_afce_v21.data.cache_io import AFCECacheStore, DinoCacheStore
from effect_afce_v21.data.robot_tokens import robot_horizon_delta_s
from effect_afce_v21.data.world_tokens import build_interval_regions, make_pca_proj, world_interval_bounds
from effect_coupled_v4.constants import MATCH_RADIUS_IMAGE_FRACTION, NUM_SEGMENTS
from effect_coupled_v4.data.env_descriptor import local_candidate_matrix, patch_centers
from effect_coupled_v4.data.window_index import enumerate_windows
from effect_vla.data.lerobot_io import load_episode_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", action="append", dest="tasks")
    p.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    p.add_argument("--dino-cache-root", type=Path, default=DEFAULT_DINO_CACHE_ROOT)
    p.add_argument("--afce-cache-root", type=Path, default=DEFAULT_AFCE_CACHE_ROOT)
    p.add_argument("--horizon", type=int, default=ACTION_HORIZON)
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    return p.parse_args()


def _load_actions(task_root: Path, episode_index: int) -> np.ndarray:
    import pandas as pd

    for pf in sorted((task_root / "data").rglob("*.parquet")):
        df = pd.read_parquet(pf, columns=["action", "episode_index"])
        sub = df[df["episode_index"] == episode_index]
        if sub.empty:
            continue
        return np.stack([np.asarray(v, dtype=np.float32) for v in sub["action"]], axis=0)
    raise KeyError(episode_index)


def cache_episode(
    *,
    task: str,
    ep: int,
    task_root: Path,
    dino: DinoCacheStore,
    store: AFCECacheStore,
    centers: np.ndarray,
    cand_mat: np.ndarray,
    cand_valid: np.ndarray,
    pca_w: np.ndarray,
    pca_b: np.ndarray,
    horizon: int,
    device: str,
) -> int:
    feats = dino.load_dino(task, ep)
    mask = dino.load_robot_mask(task, ep)
    states = np.asarray(load_episode_parquet(task_root, ep)["state"], dtype=np.float32)
    actions = _load_actions(task_root, ep)
    # Need state[t+H] and action[t:t+H); same rule as v4 window_index.
    ep_len = min(feats.shape[0], states.shape[0], actions.shape[0])
    feats = feats[:ep_len]
    states = states[:ep_len]
    actions = actions[:ep_len]
    if mask is not None:
        mask = mask[:ep_len]

    windows = enumerate_windows(
        ep_len, episode_index=ep, task=task, horizon=horizon, k=NUM_SEGMENTS, full_horizon_only=True
    )
    # Per-step ΔS needs state[t+H]; filter defensively.
    windows = [w for w in windows if w.t + horizon < states.shape[0] and w.t + horizon <= actions.shape[0]]
    if not windows:
        return 0

    n_win = len(windows)
    kw = NUM_WORLD_INTERVALS
    A = np.zeros((n_win, horizon, actions.shape[-1]), np.float32)
    state_t = np.zeros((n_win, states.shape[-1]), np.float32)
    t0 = np.zeros((n_win,), np.int32)
    root_feat = np.zeros((n_win, horizon, 9), np.float32)
    tip_feat = np.zeros((n_win, horizon, 4, 3), np.float32)
    joint_feat = np.zeros((n_win, horizon, 16), np.float32)
    delta_s = np.zeros((n_win, horizon, 37), np.float32)
    W_raw = np.zeros((n_win, kw, 8, 72), np.float32)
    Y_W = np.zeros((n_win, kw, 8, 66), np.float32)
    W_valid = np.zeros((n_win, kw, 8), np.float32)
    W_conf = np.zeros((n_win, kw, 8), np.float32)
    W_patch_idx = np.zeros((n_win, kw, 8), np.int32)

    # Match cache keyed by absolute (s,e); times patched per window
    match_cache: dict[tuple[int, int], dict] = {}

    for wi, w in enumerate(windows):
        t0[wi] = w.t
        state_t[wi] = states[w.t]
        A[wi] = actions[w.t : w.t + horizon]
        rob = robot_horizon_delta_s(states, w.t, horizon)
        root_feat[wi] = rob["root_feat"]
        tip_feat[wi] = rob["tip_feat"]
        joint_feat[wi] = rob["joint_feat"]
        delta_s[wi] = rob["delta_s"]

        for ki, (s, e) in enumerate(world_interval_bounds(w.t, horizon, kw)):
            key = (int(s), int(e))
            if key not in match_cache:
                mask_s = None if mask is None else mask[s]
                pack = build_interval_regions(
                    feats[s],
                    feats[e],
                    mask_s,
                    centers=centers,
                    cand_mat=cand_mat,
                    cand_valid=cand_valid,
                    pca_w=pca_w,
                    pca_b=pca_b,
                    t_start=0.0,
                    t_end=1.0,
                    device=device,
                )
                match_cache[key] = pack
            pack = match_cache[key]
            raw = pack["W_raw"].copy()
            ts = (s - w.t) / float(max(horizon, 1))
            te = (e - w.t) / float(max(horizon, 1))
            raw[:, -2] = ts
            raw[:, -1] = te
            W_raw[wi, ki] = raw
            Y_W[wi, ki] = pack["Y_W"]
            W_valid[wi, ki] = pack["W_valid"]
            W_conf[wi, ki] = pack["W_conf"]
            W_patch_idx[wi, ki] = pack["patch_idx"]

    store.save_episode(
        task,
        ep,
        action=A,
        state_t=state_t,
        t=t0,
        root_feat=root_feat,
        tip_feat=tip_feat,
        joint_feat=joint_feat,
        delta_s=delta_s,
        W_raw=W_raw,
        Y_W=Y_W,
        W_valid=W_valid,
        W_conf=W_conf,
        W_patch_idx=W_patch_idx,
    )
    return n_win


def shard_episodes(eps: list[int], num_shards: int, shard_id: int) -> list[int]:
    return [e for i, e in enumerate(sorted(eps)) if i % num_shards == shard_id]


def main() -> None:
    args = parse_args()
    tasks = tuple(args.tasks) if args.tasks else DEFAULT_PILOT_TASKS
    dino = DinoCacheStore(args.dino_cache_root)
    store = AFCECacheStore(args.afce_cache_root)
    centers = patch_centers()
    cand_mat, cand_valid = local_candidate_matrix(centers, radius_frac=MATCH_RADIUS_IMAGE_FRACTION)
    pca_w, pca_b = make_pca_proj(0)
    args.afce_cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.afce_cache_root / "pca_proj.npz", weight=pca_w, bias=pca_b)

    for task in tasks:
        task_root = Path(args.lerobot_root) / task
        eps_all = dino.list_episodes(task)
        if args.max_episodes > 0:
            eps_all = eps_all[: args.max_episodes]
        if not eps_all:
            raise RuntimeError(f"No DINO episodes for {task}")
        if args.shard_id == 0:
            eps_sorted = sorted(eps_all)
            n_val = max(1, int(round(len(eps_sorted) * args.val_ratio))) if len(eps_sorted) > 1 else 0
            store.save_json(
                task,
                "split_source_episodes.json",
                {
                    "train": eps_sorted[:-n_val] if n_val else eps_sorted,
                    "val": eps_sorted[-n_val:] if n_val else [],
                    "schema": SCHEMA_VERSION,
                },
            )
            store.save_json(
                task,
                "afce_meta.json",
                {"task": task, "horizon": args.horizon, "K_W": NUM_WORLD_INTERVALS, "schema": SCHEMA_VERSION},
            )
        eps = shard_episodes(eps_all, args.num_shards, args.shard_id)
        print(f"[{task}] shard {args.shard_id}/{args.num_shards} eps={len(eps)} device={args.device}")
        total, t_start = 0, time.time()
        for i, ep in enumerate(eps):
            if args.skip_existing and store.exists(task, ep):
                print(f"[{task} s{args.shard_id}] skip {ep}")
                continue
            t0 = time.time()
            n = cache_episode(
                task=task,
                ep=ep,
                task_root=task_root,
                dino=dino,
                store=store,
                centers=centers,
                cand_mat=cand_mat,
                cand_valid=cand_valid,
                pca_w=pca_w,
                pca_b=pca_b,
                horizon=args.horizon,
                device=args.device,
            )
            total += n
            print(f"[{task} s{args.shard_id}] ep {ep} windows={n} sec={time.time()-t0:.1f} ({i+1}/{len(eps)})")
        print(f"[{task} s{args.shard_id}] done windows={total} elapsed={time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
