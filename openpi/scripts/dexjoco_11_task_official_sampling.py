"""Train an eleven-task DexJoCo policy with official-style frame sampling.

The official converter creates one multi-task LeRobot dataset by appending all
retained frames.  Uniform sampling from that dataset is equivalent to the
``balance="proportional"`` multi-source view used here.  The launcher keeps the
existing per-task roots in place while matching that sampling distribution.
"""

from __future__ import annotations

import dataclasses
import importlib
from importlib import metadata as importlib_metadata
import json
import pathlib

import tyro

import dexjoco_11_task_balanced as _balanced


FLAT_CONFIG_NAME = "dexjoco_11_task_official_sampling"
STRUCTURED_CONFIG_NAME = "dexjoco_11_task_structured_official_sampling"


def build_config(args: _balanced.Args):
    """Build the existing eleven-task config with merged-frame weighting."""

    config, init_params = _balanced.build_config(args)
    data = dataclasses.replace(config.data, balance="proportional")
    config = dataclasses.replace(
        config,
        name=STRUCTURED_CONFIG_NAME if args.structured_hand_state else FLAT_CONFIG_NAME,
        data=data,
    )
    return config, init_params


def _summary(config, init_params: pathlib.Path, operation: _balanced.Operation) -> dict:
    summary = _balanced._summary(config, init_params, operation)  # noqa: SLF001
    data = config.data.create(config.assets_dirs, config.model)
    total_frames = []
    for source in data.sources:
        info = json.loads((pathlib.Path(source.root) / "meta/info.json").read_text(encoding="utf-8"))
        total_frames.append((pathlib.Path(source.root).name, int(info["total_frames"])))
    frame_sum = sum(frame_count for _, frame_count in total_frames)
    summary["data"].update(
        {
            "balance": "proportional",
            "task_weight": "retained_frames / all_retained_frames",
            "frame_counts": dict(total_frames),
            "frame_weights": {
                task: frame_count / frame_sum for task, frame_count in total_frames
            },
            "total_frames": frame_sum,
        }
    )
    return summary


def main(args: _balanced.Args) -> None:
    config, init_params = build_config(args)
    print(json.dumps(_summary(config, init_params, args.operation), indent=2, sort_keys=True))
    if args.operation == "inspect":
        return

    _balanced._validate_sources(config)  # noqa: SLF001
    if args.operation == "norm-stats":
        stats_module = importlib.import_module("compute_proportional_dexjoco_11_norm_stats")
        norm_stats, task_samples = stats_module.compute_proportional_stats(
            args.data_root,
            structured_hand_state=args.structured_hand_state,
        )
        from openpi.shared import normalize  # noqa: PLC0415

        output_dir = _balanced._norm_stats_path(config).parent  # noqa: SLF001
        normalize.save(output_dir, norm_stats)
        total_frames = sum(sample.frame_count for sample in task_samples)
        for sample in task_samples:
            print(
                f"validated {sample.task}: episodes={sample.episode_count}, "
                f"frames={sample.frame_count}, frame_weight={sample.frame_count / total_frames:.8f}"
            )
        print(f"wrote frame-proportional normalization stats to {output_dir / 'norm_stats.json'}")
        return

    try:
        lerobot_version = importlib_metadata.version("lerobot")
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeError("LeRobot is not installed in the training environment.") from error
    if lerobot_version != _balanced.EXPECTED_LEROBOT_VERSION:
        raise RuntimeError(
            f"DexJoCo training requires lerobot=={_balanced.EXPECTED_LEROBOT_VERSION}, "
            f"got {lerobot_version}."
        )
    norm_stats_path = _balanced._norm_stats_path(config)  # noqa: SLF001
    if not norm_stats_path.is_file():
        raise FileNotFoundError(
            f"Normalization stats do not exist: {norm_stats_path}. Run --operation norm-stats first."
        )
    train = importlib.import_module("train")
    train.main(config)


if __name__ == "__main__":
    main(tyro.cli(_balanced.Args))

