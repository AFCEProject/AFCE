"""Offline Go/No-Go + modality interventions for AFCE v2.1.

Checks:
  1. L_A(shuffle E) / L_A(correct) > 1.2
  2. L_A(zero E) >> L_A(correct)
  3. drop-V → L_V worse; drop-S → L_S worse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from effect_codec.constants import DEFAULT_AFCE_CACHE_ROOT
from effect_codec.model.codec import AFCECodec, afce_losses, world_loss
from effect_codec.scripts.train_afce import AFCEWindowDataset, collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--task", action="append", dest="tasks", required=True)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_AFCE_CACHE_ROOT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=40)
    return p.parse_args()


@torch.no_grad()
def mean_metrics(model, loader, device, *, max_batches: int, mode: str) -> dict[str, float]:
    model.eval()
    totals = {"L_A": 0.0, "L_S": 0.0, "L_V": 0.0, "pos_l2": 0.0}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if mode == "correct":
            out = model(batch)
        elif mode == "drop_a":
            out = model(batch, drop_a=True)
        elif mode == "drop_s":
            out = model(batch, drop_s=True)
        elif mode == "drop_v":
            out = model(batch, drop_v=True)
        elif mode == "shuffle_e":
            # Intervene on E for ALL decoders (D_A / D_S / D_V), not only D_A.
            out = model(batch)
            e = out["E"][torch.randperm(out["E"].shape[0], device=out["E"].device)]
            a_hat = model.d_a(e, batch["state_t"])
            s_hat = model.d_s(e)
            v_hat = model.d_v(e, batch["W_raw"])
            totals["L_A"] += float(F.smooth_l1_loss(a_hat, batch["action"]).cpu())
            totals["L_S"] += float(F.smooth_l1_loss(s_hat, batch["delta_s"]).cpu())
            totals["L_V"] += float(world_loss(v_hat, batch["Y_W"], batch["W_valid"]).cpu())
            totals["pos_l2"] += float(torch.linalg.norm(a_hat[..., :3] - batch["action"][..., :3], dim=-1).mean().cpu())
            n += 1
            continue
        elif mode == "zero_e":
            out = model(batch)
            e0 = torch.zeros_like(out["E"])
            a_hat = model.d_a(e0, batch["state_t"])
            s_hat = model.d_s(e0)
            v_hat = model.d_v(e0, batch["W_raw"])
            totals["L_A"] += float(F.smooth_l1_loss(a_hat, batch["action"]).cpu())
            totals["L_S"] += float(F.smooth_l1_loss(s_hat, batch["delta_s"]).cpu())
            totals["L_V"] += float(world_loss(v_hat, batch["Y_W"], batch["W_valid"]).cpu())
            totals["pos_l2"] += float(torch.linalg.norm(a_hat[..., :3] - batch["action"][..., :3], dim=-1).mean().cpu())
            n += 1
            continue
        else:
            raise ValueError(mode)
        losses = afce_losses(out, batch)
        totals["L_A"] += float(losses["loss_action"].cpu())
        totals["L_S"] += float(losses["loss_s"].cpu())
        totals["L_V"] += float(losses["loss_v"].cpu())
        totals["pos_l2"] += float(losses["action_pos_l2"].cpu())
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = tuple(args.tasks)
    ds = AFCEWindowDataset(args.cache_root, tasks, "val")
    if len(ds) == 0:
        ds = AFCEWindowDataset(args.cache_root, tasks, "train")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = AFCECodec().to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    modes = ["correct", "shuffle_e", "zero_e", "drop_a", "drop_s", "drop_v"]
    metrics = {m: mean_metrics(model, loader, device, max_batches=args.max_batches, mode=m) for m in modes}
    c = metrics["correct"]
    shuffle_ratio = metrics["shuffle_e"]["L_A"] / max(c["L_A"], 1e-8)
    zero_ratio = metrics["zero_e"]["L_A"] / max(c["L_A"], 1e-8)
    # Relative gates (absolute deltas are meaningless when L_S/L_V ≈ 0)
    rel = {
        "shuffle_LA": shuffle_ratio,
        "zero_LA": zero_ratio,
        "shuffle_LS": metrics["shuffle_e"]["L_S"] / max(c["L_S"], 1e-12),
        "zero_LS": metrics["zero_e"]["L_S"] / max(c["L_S"], 1e-12),
        "shuffle_LV": metrics["shuffle_e"]["L_V"] / max(c["L_V"], 1e-12),
        "zero_LV": metrics["zero_e"]["L_V"] / max(c["L_V"], 1e-12),
        "drop_a_LA": metrics["drop_a"]["L_A"] / max(c["L_A"], 1e-12),
        "drop_s_LS": metrics["drop_s"]["L_S"] / max(c["L_S"], 1e-12),
        "drop_v_LV": metrics["drop_v"]["L_V"] / max(c["L_V"], 1e-12),
    }
    gates = {
        "shuffle_LA_gt_1_2": {"value": rel["shuffle_LA"], "pass": rel["shuffle_LA"] > 1.2},
        "zero_LA_gt_2": {"value": rel["zero_LA"], "pass": rel["zero_LA"] > 2.0},
        "zero_LS_gt_2": {"value": rel["zero_LS"], "pass": rel["zero_LS"] > 2.0},
        "zero_LV_gt_2": {"value": rel["zero_LV"], "pass": rel["zero_LV"] > 2.0},
        "drop_s_LS_gt_2": {"value": rel["drop_s_LS"], "pass": rel["drop_s_LS"] > 2.0},
        "drop_v_LV_gt_2": {"value": rel["drop_v_LV"], "pass": rel["drop_v_LV"] > 2.0},
    }
    report = {
        "ckpt": str(args.ckpt),
        "tasks": list(tasks),
        "metrics": metrics,
        "relative": rel,
        "gates": gates,
        "go": all(g["pass"] for g in gates.values()),
        "fix_note": (
            "shuffle/zero intervene on E for D_A, D_S, and D_V "
            "(fixed: previously only D_A was intervened)."
        ),
    }
    out = args.out or (Path(args.ckpt).parent / "go_nogo_report.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2))
    print(f"GO={report['go']} wrote {out}")


if __name__ == "__main__":
    main()
