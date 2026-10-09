"""AFCE v2.1 full-episode GT-E sim rollout (R1 chunk / R2 continuous).

Teacher-forced E* = Enc(A,ΔS,ΔV) from AFCE cache; A_hat = D_A(E*, S).
Modes:
  open_loop_stitch — decode all chunks offline with recorded S_t, then execute
  r2_online        — after each H-step chunk, re-decode next E* with sim S_t

Reuses recovered state0 from prior GT succeed search (same as v4 gate).

Example:
  PYTHONPATH=$PWD/dexjoco:$PWD python -m effect_codec.scripts.eval_full_ep_gt_e_replay \\
    --codec-ckpt outputs/effect_codec/runs/afce_pilot/best.pt \\
    --episodes 0,1,2,3,4 --mode r2_online --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from effect_codec.constants import (
    ACTION_HORIZON,
    DEFAULT_AFCE_CACHE_ROOT,
    DEFAULT_AFCE_RUN_ROOT,
    DEFAULT_LEROBOT_ROOT,
)
from effect_codec.model.codec import AFCECodec
from effect_codec.scripts.train_afce import AFCEWindowDataset, collate
from effect_coupled_v4.eval.action_convert import action_rotvec22_to_quat23
from effect_coupled_v4.scripts.eval_full_ep_codec_e_replay import (
    _append_obs_frames,
    _write_episode_videos,
    replay_full,
)
from effect_coupled_v4.scripts.eval_full_ep_gt_replay import load_episode_actions_states
from dexjoco.tasks.mappings import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import restore_initial_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="water_plant")
    p.add_argument(
        "--codec-ckpt",
        type=Path,
        default=DEFAULT_AFCE_RUN_ROOT / "afce_pilot" / "best.pt",
    )
    p.add_argument("--cache-root", type=Path, default=DEFAULT_AFCE_CACHE_ROOT)
    p.add_argument(
        "--state0-json",
        type=Path,
        default=DEFAULT_AFCE_RUN_ROOT.parent
        / "effect_coupled_v4"
        / "sim_replay"
        / "full_ep_gt_water_plant_5ep.json",
    )
    p.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    p.add_argument("--episodes", default="0,1,2,3,4")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mode",
        choices=("open_loop_stitch", "r2_online"),
        default="r2_online",
        help="r2_online uses sim S_t for each chunk decode (plan §31)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_AFCE_RUN_ROOT
        / "sim_replay"
        / "full_ep_gt_e_water_plant_5ep.json",
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_AFCE_RUN_ROOT
        / "sim_replay"
        / "videos_gt_e_water_plant_5ep",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--continue-after-success", action="store_true")
    return p.parse_args()


def build_window_index(cache_root: Path, task: str) -> dict[tuple[int, int], tuple[str, int]]:
    """Map (episode, t0) -> (split, dataset_index) using cached t field."""
    index: dict[tuple[int, int], tuple[str, int]] = {}
    for split in ("train", "val"):
        ds = AFCEWindowDataset(cache_root, (task,), split=split)
        for i, (_task, ep, wi) in enumerate(ds.samples):
            # Resolve absolute t from episode cache
            data = ds._load(task, int(ep))
            t0 = int(data["t"][wi])
            index[(int(ep), t0)] = (split, i)
    return index


@torch.no_grad()
def encode_and_decode(
    model: AFCECodec,
    sample: dict,
    device: torch.device,
    *,
    state_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    batch = collate([sample])
    batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    if state_override is not None:
        st = torch.as_tensor(state_override, dtype=batch["state_t"].dtype, device=device)
        batch["state_t"] = st.view(1, -1)
    out = model(batch)
    a_hat = out["action_pred"][0].detach().cpu().numpy().astype(np.float64)
    a_gt = sample["action"].numpy().astype(np.float64)
    return a_hat, a_gt


def stitch_open_loop(
    *,
    model: AFCECodec,
    datasets: dict[str, AFCEWindowDataset],
    index: dict[tuple[int, int], tuple[str, int]],
    episode: int,
    episode_len: int,
    horizon: int,
    device: torch.device,
    gt_actions22: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    chunks: list[np.ndarray] = []
    chunk_meta: list[dict[str, Any]] = []
    offline_maes: list[float] = []
    t = 0
    n_gt_tail = 0
    while t < episode_len:
        key = (episode, t)
        if key in index and t + horizon <= episode_len:
            split, di = index[key]
            a_hat, a_gt = encode_and_decode(model, datasets[split][di], device)
            chunks.append(a_hat)
            offline_maes.append(float(np.mean(np.abs(a_hat - a_gt))))
            chunk_meta.append({"t_exec": t, "source": "gt_e", "split": split, "mae": offline_maes[-1]})
            t += horizon
            continue
        valid_ts = [tw for (ep, tw) in index if ep == episode and tw <= t < tw + horizon]
        if valid_ts:
            t_win = max(valid_ts)
            split, di = index[(episode, t_win)]
            a_hat, a_gt = encode_and_decode(model, datasets[split][di], device)
            offset = t - t_win
            take = min(horizon - offset, episode_len - t)
            chunks.append(a_hat[offset : offset + take])
            offline_maes.append(float(np.mean(np.abs(a_hat - a_gt))))
            chunk_meta.append(
                {
                    "t_exec": t,
                    "t_win": t_win,
                    "source": "gt_e_overlap_tail",
                    "split": split,
                    "mae": offline_maes[-1],
                }
            )
            t += take
            continue
        take = episode_len - t
        chunks.append(np.asarray(gt_actions22[t : t + take], dtype=np.float64))
        n_gt_tail += take
        chunk_meta.append({"t_exec": t, "source": "gt_tail_fill", "take": take})
        t += take
    actions = np.concatenate(chunks, axis=0)
    assert actions.shape[0] == episode_len
    return actions, {
        "chunk_meta": chunk_meta,
        "mean_offline_mae": float(np.mean(offline_maes)) if offline_maes else None,
        "n_chunks": len(chunk_meta),
        "n_gt_tail": n_gt_tail,
    }


def replay_r2_online(
    *,
    env,
    config,
    task: str,
    state0: np.ndarray,
    model: AFCECodec,
    datasets: dict[str, AFCEWindowDataset],
    index: dict[tuple[int, int], tuple[str, int]],
    episode: int,
    episode_len: int,
    horizon: int,
    device: torch.device,
    gt_actions22: np.ndarray,
    continue_after_success: bool = False,
    record_video: bool = False,
) -> dict[str, Any]:
    """Plan §31: each chunk uses E_GT from cache + current sim S_t."""
    obs, _ = env.reset()
    obs = restore_initial_state(env, task, config, state0)
    succeed = False
    done = False
    steps = 0
    info_last: dict[str, Any] = {}
    frames: dict[str, list[np.ndarray]] = {"front": [], "wrist": []}
    if record_video:
        _append_obs_frames(obs, frames)

    chunk_meta: list[dict[str, Any]] = []
    offline_maes: list[float] = []
    t = 0
    while t < episode_len:
        key = (episode, t)
        if key in index and t + horizon <= episode_len:
            split, di = index[key]
            sample = datasets[split][di]
            s_t = np.asarray(obs["state"], dtype=np.float32).ravel()[:23]
            a_hat, a_gt = encode_and_decode(model, sample, device, state_override=s_t)
            take = horizon
            source = "gt_e_r2"
        else:
            valid_ts = [tw for (ep, tw) in index if ep == episode and tw <= t < tw + horizon]
            if valid_ts:
                t_win = max(valid_ts)
                split, di = index[(episode, t_win)]
                sample = datasets[split][di]
                s_t = np.asarray(obs["state"], dtype=np.float32).ravel()[:23]
                a_hat_full, a_gt = encode_and_decode(model, sample, device, state_override=s_t)
                offset = t - t_win
                take = min(horizon - offset, episode_len - t)
                a_hat = a_hat_full[offset : offset + take]
                source = "gt_e_r2_overlap"
            else:
                take = episode_len - t
                a_hat = np.asarray(gt_actions22[t : t + take], dtype=np.float64)
                a_gt = a_hat
                source = "gt_tail_fill"
                split = None

        if source != "gt_tail_fill":
            offline_maes.append(float(np.mean(np.abs(a_hat[: min(len(a_hat), len(a_gt))] - a_gt[: len(a_hat)]))))
        chunk_meta.append({"t_exec": t, "take": take, "source": source, "split": split})

        a23 = action_rotvec22_to_quat23(a_hat)
        for a in a23:
            obs, _rew, done, _trunc, info = env.step(np.asarray(a, dtype=np.float64))
            info_last = dict(info) if isinstance(info, dict) else {}
            succeed = bool(info_last.get("succeed", False)) or succeed
            steps += 1
            if record_video:
                _append_obs_frames(obs, frames)
            if not continue_after_success and (succeed or done):
                return {
                    "succeed": bool(succeed),
                    "done": bool(done),
                    "steps_executed": steps,
                    "horizon": episode_len,
                    "frames": frames if record_video else None,
                    "chunk_meta": chunk_meta,
                    "mean_offline_mae": float(np.mean(offline_maes)) if offline_maes else None,
                    "stopped_early": True,
                    "last_info": {
                        k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                        for k, v in info_last.items()
                        if k in ("succeed", "grasp_penalty")
                    },
                }
        t += take

    return {
        "succeed": bool(succeed),
        "done": bool(done),
        "steps_executed": steps,
        "horizon": episode_len,
        "frames": frames if record_video else None,
        "chunk_meta": chunk_meta,
        "mean_offline_mae": float(np.mean(offline_maes)) if offline_maes else None,
        "stopped_early": False,
        "last_info": {
            k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
            for k, v in info_last.items()
            if k in ("succeed", "grasp_penalty")
        },
    }


def main() -> None:
    args = parse_args()
    ep_ids = [int(x) for x in args.episodes.split(",") if x.strip() != ""]
    state0_blob = json.loads(args.state0_json.read_text())
    state0_by_ep = {
        int(r["episode_index"]): np.asarray(r["state0_full"], dtype=np.float64)
        for r in state0_blob["results"]
    }
    for ep in ep_ids:
        if ep not in state0_by_ep:
            raise SystemExit(f"No recovered state0 for ep {ep} in {args.state0_json}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} ckpt={args.codec_ckpt} mode={args.mode}", flush=True)

    datasets = {
        split: AFCEWindowDataset(args.cache_root, (args.task,), split=split)
        for split in ("train", "val")
    }
    print("building window index...", flush=True)
    index = build_window_index(args.cache_root, args.task)
    print(f"indexed windows for task={args.task}: {len(index)}", flush=True)

    model = AFCECodec().to(device)
    blob = torch.load(args.codec_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"] if "model" in blob else blob, strict=True)
    model.eval()
    ckpt_meta = {k: blob[k] for k in ("step", "val") if k in blob}
    horizon = ACTION_HORIZON
    print(f"loaded codec meta={ckpt_meta} H={horizon}", flush=True)

    config = CONFIG_MAPPING[args.task]()
    save_video = not args.no_video
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array" if save_video else "none",
        randomize=False,
        randomize_dynamics=False,
        seed=args.seed,
    )
    if save_video:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    t_all = time.time()
    try:
        for ep in ep_ids:
            actions22, _states23, meta = load_episode_actions_states(
                args.lerobot_root, args.task, ep
            )
            T = int(actions22.shape[0])
            state0 = state0_by_ep[ep]
            gt_quat = action_rotvec22_to_quat23(actions22)

            gt_out = replay_full(
                env=env,
                config=config,
                task=args.task,
                state0=state0,
                actions_quat23=gt_quat,
                continue_after_success=args.continue_after_success,
                record_video=save_video,
            )

            if args.mode == "open_loop_stitch":
                a_hat22, stitch_info = stitch_open_loop(
                    model=model,
                    datasets=datasets,
                    index=index,
                    episode=ep,
                    episode_len=T,
                    horizon=horizon,
                    device=device,
                    gt_actions22=actions22,
                )
                hat_quat = action_rotvec22_to_quat23(a_hat22)
                e_out = replay_full(
                    env=env,
                    config=config,
                    task=args.task,
                    state0=state0,
                    actions_quat23=hat_quat,
                    continue_after_success=args.continue_after_success,
                    record_video=save_video,
                )
                e_out["mean_offline_mae"] = stitch_info.get("mean_offline_mae")
                e_out["chunk_meta"] = stitch_info.get("chunk_meta")
                offline_mae = float(np.mean(np.abs(a_hat22 - actions22)))
            else:
                e_out = replay_r2_online(
                    env=env,
                    config=config,
                    task=args.task,
                    state0=state0,
                    model=model,
                    datasets=datasets,
                    index=index,
                    episode=ep,
                    episode_len=T,
                    horizon=horizon,
                    device=device,
                    gt_actions22=actions22,
                    continue_after_success=args.continue_after_success,
                    record_video=save_video,
                )
                offline_mae = e_out.get("mean_offline_mae")

            videos = {}
            if save_video:
                videos["gt"] = _write_episode_videos(
                    args.video_dir, ep, "gt", gt_out["frames"], args.fps, succeed=gt_out["succeed"]
                )
                videos["gt_e"] = _write_episode_videos(
                    args.video_dir, ep, "gt_e", e_out["frames"], args.fps, succeed=e_out["succeed"]
                )

            row = {
                "task": args.task,
                "episode_index": ep,
                "episode_length": T,
                "mode": args.mode,
                "offline_action_mae": offline_mae,
                "gt_succeed": gt_out["succeed"],
                "gt_steps": gt_out["steps_executed"],
                "gt_e_succeed": e_out["succeed"],
                "gt_e_steps": e_out["steps_executed"],
                "gt_e_stopped_early": e_out.get("stopped_early"),
                "n_chunks": len(e_out.get("chunk_meta") or []),
                "videos": videos,
            }
            rows.append(row)
            print(
                f"[ep{ep}] mae={offline_mae} gt_suc={gt_out['succeed']} "
                f"gtE_suc={e_out['succeed']}({e_out['steps_executed']}/{T}) "
                f"chunks={row['n_chunks']}",
                flush=True,
            )
    finally:
        env.close()

    n_gt = sum(1 for r in rows if r["gt_succeed"])
    n_e = sum(1 for r in rows if r["gt_e_succeed"])
    summary = {
        "task": args.task,
        "mode": args.mode,
        "codec_ckpt": str(args.codec_ckpt),
        "ckpt_meta": {k: (float(v) if hasattr(v, "item") else v) for k, v in ckpt_meta.items()}
        if isinstance(ckpt_meta, dict)
        else ckpt_meta,
        "state0_source": str(args.state0_json),
        "episodes": ep_ids,
        "video_dir": str(args.video_dir) if save_video else None,
        "n_gt_success": n_gt,
        "n_gt_e_success": n_e,
        "n_total": len(rows),
        "gt_success_rate": float(n_gt) / max(1, len(rows)),
        "gt_e_success_rate": float(n_e) / max(1, len(rows)),
        "elapsed_sec": time.time() - t_all,
        "note": (
            "AFCE v2.1 GT-E rollout: E*=Enc(A,ΔS,ΔV) from cache; "
            "D_A(E*,S). r2_online feeds sim S_t each chunk."
        ),
        "results": rows,
    }
    # JSON-safe ckpt_meta
    if "val" in summary["ckpt_meta"] and isinstance(summary["ckpt_meta"]["val"], dict):
        summary["ckpt_meta"]["val"] = {
            k: float(v) for k, v in summary["ckpt_meta"]["val"].items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "mode",
                    "n_gt_success",
                    "n_gt_e_success",
                    "n_total",
                    "gt_success_rate",
                    "gt_e_success_rate",
                    "elapsed_sec",
                )
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
