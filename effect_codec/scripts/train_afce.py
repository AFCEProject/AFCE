"""Train AFCE v2.1 Unified Information-Preserving Effect Codec.

Example:
  PYTHONPATH=$PWD python -m effect_codec.scripts.train_afce \\
    --task water_plant --task pick_bucket --task hammer_nail \\
    --device cuda:0 --run-name afce_pilot
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from effect_codec.constants import (
    DEFAULT_AFCE_CACHE_ROOT,
    DEFAULT_AFCE_RUN_ROOT,
    LAMBDA_RAMP_STEPS,
    MAX_STEPS_DEFAULT,
)
from effect_codec.model.codec import AFCECodec, afce_losses, schedule_lambdas


class AFCEWindowDataset(Dataset):
    def __init__(self, cache_root: Path, tasks: tuple[str, ...], split: str = "train"):
        self.samples: list[tuple[str, int, int]] = []
        self.cache_root = Path(cache_root)
        self._episode_cache: dict[tuple[str, int], dict] = {}
        for task in tasks:
            split_path = self.cache_root / task / "split_source_episodes.json"
            if split_path.exists():
                splits = json.loads(split_path.read_text())
                eps = splits.get(split, splits.get("train", []))
            else:
                eps = [
                    int(p.name.split("_")[1])
                    for p in sorted((self.cache_root / task).glob("episode_*"))
                    if (p / "afce_windows.npz").exists()
                ]
            for ep in eps:
                path = self.cache_root / task / f"episode_{int(ep):06d}" / "afce_windows.npz"
                if not path.exists():
                    continue
                with np.load(path) as z:
                    n = int(z["t"].shape[0])
                for wi in range(n):
                    self.samples.append((task, int(ep), wi))

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, task: str, ep: int) -> dict:
        key = (task, ep)
        if key in self._episode_cache:
            return self._episode_cache[key]
        path = self.cache_root / task / f"episode_{ep:06d}" / "afce_windows.npz"
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        if len(self._episode_cache) > 16:
            self._episode_cache.pop(next(iter(self._episode_cache)))
        self._episode_cache[key] = data
        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        task, ep, wi = self.samples[idx]
        data = self._load(task, ep)
        out = {}
        for k, v in data.items():
            if k == "t" and v.ndim == 1:
                out[k] = torch.tensor(int(v[wi]), dtype=torch.int32)
                continue
            arr = v[wi]
            if np.issubdtype(arr.dtype, np.integer):
                out[k] = torch.from_numpy(np.asarray(arr, dtype=np.int64))
            else:
                out[k] = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        return out


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


@torch.no_grad()
def eval_loss(model, loader, device, *, lambda_s: float, lambda_v: float, max_batches: int = 20) -> dict[str, float]:
    model.eval()
    keys = ("loss", "loss_action", "loss_s", "loss_v", "action_pos_l2")
    totals = {k: 0.0 for k in keys}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch)
        losses = afce_losses(out, batch, lambda_s=lambda_s, lambda_v=lambda_v)
        for k in totals:
            totals[k] += float(losses[k].detach().cpu())
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", action="append", dest="tasks", required=True)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_AFCE_CACHE_ROOT)
    p.add_argument("--run-root", type=Path, default=DEFAULT_AFCE_RUN_ROOT)
    p.add_argument("--run-name", default="afce_pilot")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_false", dest="bf16")
    p.add_argument("--lambda-ramp-steps", type=int, default=LAMBDA_RAMP_STEPS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = tuple(args.tasks)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    train_ds = AFCEWindowDataset(args.cache_root, tasks, "train")
    val_ds = AFCEWindowDataset(args.cache_root, tasks, "val")
    if len(train_ds) == 0:
        raise RuntimeError(f"empty train set under {args.cache_root}")
    print(f"train windows={len(train_ds)} val windows={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(2, args.num_workers // 2),
        collate_fn=collate, pin_memory=device.type == "cuda",
    )

    model = AFCECodec().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_bf16 = bool(args.bf16 and device.type == "cuda")
    log_path = run_dir / "train_log.jsonl"
    best_val = float("inf")
    best_pos = float("inf")
    step = 0
    t0 = time.time()
    model.train()
    it = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        lam_s, lam_v = schedule_lambdas(step, ramp=args.lambda_ramp_steps)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(batch)
            losses = afce_losses(out, batch, lambda_s=lam_s, lambda_v=lam_v)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % args.log_every == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(losses["loss"].detach().cpu()),
                "L_A": float(losses["loss_action"].detach().cpu()),
                "L_S": float(losses["loss_s"].detach().cpu()),
                "L_V": float(losses["loss_v"].detach().cpu()),
                "lambda_s": lam_s,
                "lambda_v": lam_v,
                "pos_l2": float(losses["action_pos_l2"].detach().cpu()),
                "sec": time.time() - t0,
            }
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"step={step} loss={row['loss']:.4f} LA={row['L_A']:.4f} "
                f"LS={row['L_S']:.4f} LV={row['L_V']:.4f} "
                f"lam={lam_s:.3f} pos={row['pos_l2']*100:.2f}cm"
            )

        if step % args.eval_every == 0 or step == args.max_steps:
            val = eval_loss(model, val_loader, device, lambda_s=lam_s, lambda_v=lam_v)
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "split": "val", **val}) + "\n")
            print(f"[val] step={step} loss={val['loss']:.4f} LA={val['loss_action']:.4f} pos={val['action_pos_l2']*100:.2f}cm")
            ckpt = {"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "args": vars(args), "val": val}
            torch.save(ckpt, run_dir / "last.pt")
            # Prefer action fidelity for checkpoint (§29)
            if val["action_pos_l2"] < best_pos or (
                abs(val["action_pos_l2"] - best_pos) < 1e-6 and val["loss"] < best_val
            ):
                best_pos = val["action_pos_l2"]
                best_val = val["loss"]
                torch.save(ckpt, run_dir / "best.pt")
                print(f"  saved best.pt pos={best_pos*100:.2f}cm")

        if step % args.ckpt_every == 0:
            torch.save({"step": step, "model": model.state_dict()}, run_dir / f"ckpt_{step:06d}.pt")

    print(f"done steps={step} elapsed={time.time()-t0:.1f}s run_dir={run_dir}")


if __name__ == "__main__":
    main()
