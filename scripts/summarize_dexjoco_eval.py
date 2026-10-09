"""Validate and summarize a formal 11-task, three-seed DexJoCo evaluation."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics


TASKS = (
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
MARKER_RE = re.compile(r"success_rate_(\d+)_(\d+)\.txt$")


def load_results(root: pathlib.Path, episodes: int) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {}
    missing: list[str] = []
    for task in TASKS:
        rates: list[float] = []
        for seed in range(3):
            markers = sorted((root / task / f"seed{seed}").glob("success_rate_*_*.txt"))
            valid = []
            for marker in markers:
                match = MARKER_RE.fullmatch(marker.name)
                if match and int(match.group(2)) == episodes:
                    valid.append((marker, int(match.group(1))))
            if len(valid) != 1:
                missing.append(f"{task}/seed{seed}: expected one {episodes}-episode marker, got {markers}")
                continue
            rates.append(100.0 * valid[0][1] / episodes)
        if len(rates) == 3:
            results[task] = rates
    if missing:
        raise RuntimeError("Incomplete formal evaluation:\n" + "\n".join(missing))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    results = load_results(root, args.episodes)
    task_summary = {
        task: {
            "seed_rates": rates,
            "mean": statistics.fmean(rates),
            "population_std": statistics.pstdev(rates),
        }
        for task, rates in results.items()
    }
    seed_means = [statistics.fmean(results[task][seed] for task in TASKS) for seed in range(3)]
    summary = {
        "label": args.label,
        "episodes_per_task_seed": args.episodes,
        "total_episodes": len(TASKS) * 3 * args.episodes,
        "tasks": task_summary,
        "seed_means": seed_means,
        "overall_mean": statistics.fmean(seed_means),
        "overall_population_std": statistics.pstdev(seed_means),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {args.label}",
        "",
        "| Task | Seed 0 | Seed 1 | Seed 2 | Mean ± population std |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = task_summary[task]
        rates = item["seed_rates"]
        lines.append(
            f"| {task} | {rates[0]:.1f}% | {rates[1]:.1f}% | {rates[2]:.1f}% | "
            f"{item['mean']:.1f}% ± {item['population_std']:.1f}% |"
        )
    lines.append(
        f"| **11-task average** | **{seed_means[0]:.1f}%** | **{seed_means[1]:.1f}%** | "
        f"**{seed_means[2]:.1f}%** | **{summary['overall_mean']:.1f}% ± "
        f"{summary['overall_population_std']:.1f}%** |"
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"FORMAL_SUMMARY_DONE {root}")


if __name__ == "__main__":
    main()
