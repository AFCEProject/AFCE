"""Print / optionally launch the existing DexJoCo π0.5 rollout protocol.

Full vs Baseline-Continue must use the same env seeds and episode count as
the frozen baseline. This wrapper does not invent a new evaluator.

    python -m effect_vla.eval.dexjoco_rollout --task water_plant --port 8000
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from effect_vla.constants import DEXJOCO_ROOT, DEFAULT_PILOT_TASKS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", action="append", dest="tasks")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--config-dir", type=Path, default=Path(DEXJOCO_ROOT) / "configs" / "rand_obj")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", action="store_true", help="Actually call dexjoco-openpi-eval.")
    p.add_argument("--rand-full", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = tuple(args.tasks) if args.tasks else (DEFAULT_PILOT_TASKS if args.pilot else ())
    if not tasks:
        raise SystemExit("Pass --task or --pilot")
    cfg_dir = Path(DEXJOCO_ROOT) / ("configs/rand_full" if args.rand_full else "configs/rand_obj")
    for task in tasks:
        cfg = cfg_dir / f"{task}.yaml"
        cmd = [
            "dexjoco-openpi-eval",
            f"--config={cfg}",
            f"--seed={args.seed}",
            f"--port={args.port}",
        ]
        if args.rand_full:
            cmd.append("--rand-full")
        print(" ".join(cmd))
        if args.run:
            subprocess.check_call(cmd, cwd=DEXJOCO_ROOT)


if __name__ == "__main__":
    main()
