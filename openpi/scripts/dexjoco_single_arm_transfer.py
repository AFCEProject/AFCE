"""Train a task-balanced DexJoCo single-arm π0.5 model, then Click/Pinch specialists.

Run this launcher from ``openpi/``. ``--data-root`` for the shared stage is the
parent of the six official LeRobot task directories. The loader opens them in
place, samples each task equally, and lazily maps Click's ``ego_right`` camera and
the other tasks' ``front`` camera to one canonical base view. It does not merge or
copy the datasets.

The shared stage uses only two RGB views, language, 23-D state, and 22-D action:

    python scripts/dexjoco_single_arm_transfer.py \
      --operation norm-stats --stage shared \
      --data-root /datasets/dexjoco_lerobot_datasets \
      --init-params-path /checkpoints/pi05_base/params \
      --assets-base-dir /outputs/assets

    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/dexjoco_single_arm_transfer.py \
      --operation train --stage shared \
      --data-root /datasets/dexjoco_lerobot_datasets \
      --init-params-path /checkpoints/pi05_base/params \
      --assets-base-dir /outputs/assets \
      --checkpoint-base-dir /outputs/checkpoints

The final 60,000-update checkpoint is stored at
``<checkpoint_base_dir>/single_arm_shared/<exp_name>/59999``. Specialists load
that checkpoint's ``params`` with a fresh optimizer and keep the shared model's
normalization coordinates, so the transferred function is continuous at step 0:

    CUDA_VISIBLE_DEVICES=0,1 python scripts/dexjoco_single_arm_transfer.py \
      --operation train --stage click_mouse \
      --data-root /datasets/dexjoco_lerobot_datasets/click_mouse \
      --init-params-path /outputs/checkpoints/single_arm_shared/EXP/59999/params \
      --checkpoint-base-dir /outputs/checkpoints

    CUDA_VISIBLE_DEVICES=2,3 python scripts/dexjoco_single_arm_transfer.py \
      --operation train --stage pinch_tongs \
      --data-root /datasets/dexjoco_lerobot_datasets/pinch_tongs \
      --init-params-path /outputs/checkpoints/single_arm_shared/EXP/59999/params \
      --checkpoint-base-dir /outputs/checkpoints

Specialists default to 10,000 updates, a 1e-5 peak learning rate, a 1,000-update
warmup, and cosine decay to 1e-6. Their checkpoints contain the inherited shared
normalization assets, so the existing ``serve_policy.py`` path remains valid.
No force, taxel, contact, or other tactile field is read by this pipeline.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC
from datetime import datetime
import importlib
from importlib import metadata as importlib_metadata
import json
import pathlib
from typing import Literal

import tyro

from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

Operation = Literal["inspect", "norm-stats", "train"]
Stage = Literal["shared", "click_mouse", "pinch_tongs"]

SHARED_CONFIG_NAME = "single_arm_shared"
SHARED_TEMPLATE_NAME = "pinch_tongs"
SHARED_DEFAULT_STEPS = 60_000
SHARED_DEFAULT_KEEP_PERIOD = 20_000
SPECIALIST_DEFAULT_STEPS = 10_000
SPECIALIST_DEFAULT_WARMUP_STEPS = 1_000
SPECIALIST_DEFAULT_PEAK_LR = 1e-5
SPECIALIST_DEFAULT_DECAY_LR = 1e-6
SPECIALIST_DEFAULT_SAVE_INTERVAL = 5_000
SPECIALIST_DEFAULT_KEEP_PERIOD = 5_000
EXPECTED_LEROBOT_VERSION = "0.4.4"
SINGLE_ARM_TASK_BASE_KEYS = {
    "click_mouse": "observation.images.ego_right",
    "fold_glasses": "observation.images.front",
    "hammer_nail": "observation.images.front",
    "pick_bucket": "observation.images.front",
    "pinch_tongs": "observation.images.front",
    "water_plant": "observation.images.front",
}


@dataclasses.dataclass(frozen=True)
class Args:
    """Arguments for the shared-to-specialist training pipeline."""

    operation: Operation = "inspect"
    stage: Stage = "shared"

    # Shared: parent containing all six task roots. Specialist: the selected single-task root.
    data_root: pathlib.Path | None = None
    # Shared: pi0.5 base params (optional if config.yaml points to a valid local path).
    # Specialist: required shared checkpoint path ending in /params.
    init_params_path: pathlib.Path | None = None

    exp_name: str | None = None
    checkpoint_base_dir: pathlib.Path | None = None
    assets_base_dir: pathlib.Path | None = None
    project_name: str = "dexjoco-pi05-single-arm-transfer"

    batch_size: int | None = None
    num_workers: int | None = None
    num_train_steps: int | None = None
    save_interval: int | None = None
    keep_period: int | None = None
    fsdp_devices: int | None = None
    video_backend: str = "pyav"

    # None chooses the stage default. All four schedule fields are independently overridable.
    warmup_steps: int | None = None
    peak_lr: float | None = None
    decay_steps: int | None = None
    decay_lr: float | None = None

    wandb_enabled: bool = True


def _positive_int(value: int, field_name: str) -> int:
    if value <= 0:
        raise ValueError(f"--{field_name.replace('_', '-')} must be positive, got {value}.")
    return value


def _nonnegative_int(value: int, field_name: str) -> int:
    if value < 0:
        raise ValueError(f"--{field_name.replace('_', '-')} must be non-negative, got {value}.")
    return value


def _positive_float(value: float, field_name: str) -> float:
    if value <= 0:
        raise ValueError(f"--{field_name.replace('_', '-')} must be positive, got {value}.")
    return value


def _resolve_directory(path: pathlib.Path, *, description: str, must_exist: bool) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    if must_exist and not resolved.is_dir():
        raise FileNotFoundError(f"{description} directory does not exist: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(f"{description} path is not a directory: {resolved}")
    return resolved


def _resolve_params_path(path: pathlib.Path) -> pathlib.Path:
    resolved = _resolve_directory(path, description="Checkpoint params", must_exist=True)
    if resolved.name != "params":
        raise ValueError(f"--init-params-path must point to a checkpoint's /params directory, got: {resolved}")
    return resolved


def _default_exp_name(stage: Stage) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}_from_shared_{timestamp}" if stage != "shared" else f"shared_{timestamp}"


def _validate_exp_name(exp_name: str) -> None:
    if not exp_name or pathlib.PurePath(exp_name).name != exp_name:
        raise ValueError(f"--exp-name must be one non-empty path component, got: {exp_name!r}")


def _model_signature(config: _config.TrainConfig) -> tuple:
    model = config.model
    return (
        type(model),
        getattr(model, "pi05", None),
        model.action_dim,
        model.action_horizon,
        getattr(model, "paligemma_variant", None),
        getattr(model, "action_expert_variant", None),
        model.max_token_len,
    )


def _normalization_path(config: _config.TrainConfig) -> pathlib.Path:
    data = config.data
    asset_id = data.assets.asset_id or data.repo_id
    if asset_id is None:
        raise ValueError(f"The {config.name} data config has no normalization asset id.")
    assets_root = pathlib.Path(data.assets.assets_dir).resolve() if data.assets.assets_dir else config.assets_dirs
    return assets_root / asset_id / "norm_stats.json"


def _shared_sources(data_parent: pathlib.Path) -> tuple[_config.LeRobotDatasetSource, ...]:
    return tuple(
        _config.LeRobotDatasetSource(
            root=(data_parent / task).resolve(),
            base_image_key=base_image_key,
        )
        for task, base_image_key in SINGLE_ARM_TASK_BASE_KEYS.items()
    )


def _build_schedule(args: Args, *, num_train_steps: int, template_schedule) -> _optimizer.CosineDecaySchedule:
    if args.stage == "shared":
        default_warmup = template_schedule.warmup_steps
        default_peak = template_schedule.peak_lr
        default_decay_steps = template_schedule.decay_steps
        # The repository's single-arm config intentionally stays at its peak LR after warmup.
        default_decay_lr = args.peak_lr if args.peak_lr is not None else template_schedule.decay_lr
    else:
        default_warmup = SPECIALIST_DEFAULT_WARMUP_STEPS
        default_peak = SPECIALIST_DEFAULT_PEAK_LR
        default_decay_steps = num_train_steps
        default_decay_lr = SPECIALIST_DEFAULT_DECAY_LR

    warmup_steps = args.warmup_steps if args.warmup_steps is not None else default_warmup
    peak_lr = args.peak_lr if args.peak_lr is not None else default_peak
    decay_steps = args.decay_steps if args.decay_steps is not None else default_decay_steps
    if args.decay_lr is not None:
        decay_lr = args.decay_lr
    elif args.stage == "shared":
        decay_lr = default_decay_lr
    elif args.peak_lr is not None:
        decay_lr = args.peak_lr / 10
    else:
        decay_lr = default_decay_lr

    _positive_int(warmup_steps, "warmup_steps")
    _positive_int(decay_steps, "decay_steps")
    _positive_float(peak_lr, "peak_lr")
    _positive_float(decay_lr, "decay_lr")
    if decay_steps <= warmup_steps:
        raise ValueError(f"--decay-steps ({decay_steps}) must be greater than --warmup-steps ({warmup_steps}).")

    return _optimizer.CosineDecaySchedule(
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        decay_steps=decay_steps,
        decay_lr=decay_lr,
    )


def build_config(args: Args) -> tuple[_config.TrainConfig, pathlib.Path]:
    """Build a same-architecture shared or specialist config and resolve its source params."""

    template_name = SHARED_TEMPLATE_NAME if args.stage == "shared" else args.stage
    template = _config.get_config(template_name)
    shared_template = _config.get_config(SHARED_TEMPLATE_NAME)
    if not isinstance(template.data, _config.SingleArmDataConfig):
        raise TypeError(f"Expected a SingleArmDataConfig template, got {type(template.data).__name__}.")
    if not isinstance(template.lr_schedule, _optimizer.CosineDecaySchedule):
        raise TypeError(f"Expected a CosineDecaySchedule template, got {type(template.lr_schedule).__name__}.")
    if _model_signature(template) != _model_signature(shared_template):
        raise ValueError(
            f"{template_name} no longer matches the shared pi0.5 architecture. "
            "Refusing checkpoint transfer because it could reinitialize or drop parameters."
        )
    if not args.video_backend:
        raise ValueError("--video-backend must be a non-empty LeRobot backend name.")
    if args.operation == "norm-stats" and args.stage != "shared":
        raise ValueError(
            "Specialists inherit the shared checkpoint's normalization coordinates; "
            "--operation norm-stats is only valid for --stage shared."
        )

    if args.stage == "shared":
        if args.data_root is None:
            raise ValueError("--data-root is required for --stage shared.")
        data_root = _resolve_directory(
            args.data_root,
            description="Single-arm LeRobot task parent",
            must_exist=args.operation != "inspect",
        )
        data = _config.SingleArmMultiTaskDataConfig(
            repo_id=template.data.repo_id,
            assets=template.data.assets,
            base_config=template.data.base_config,
            sources=_shared_sources(data_root),
            balance="task",
            video_backend=args.video_backend,
        )
        config_name = SHARED_CONFIG_NAME
        default_steps = SHARED_DEFAULT_STEPS
        default_save_interval = template.save_interval
        default_keep_period = SHARED_DEFAULT_KEEP_PERIOD
        source_path = args.init_params_path
        if source_path is None:
            if not isinstance(template.weight_loader, weight_loaders.CheckpointWeightLoader):
                raise TypeError(
                    "Shared template does not use CheckpointWeightLoader; pass --init-params-path explicitly."
                )
            source_path = pathlib.Path(template.weight_loader.params_path)
    else:
        if args.init_params_path is None:
            raise ValueError(
                f"--init-params-path is required for --stage {args.stage}; "
                "point it at the shared checkpoint's <step>/params directory."
            )
        data = template.data
        if args.data_root is not None:
            data = dataclasses.replace(
                data,
                root=_resolve_directory(
                    args.data_root,
                    description=f"{args.stage} LeRobot dataset",
                    must_exist=args.operation != "inspect",
                ),
            )
        config_name = args.stage
        default_steps = SPECIALIST_DEFAULT_STEPS
        default_save_interval = SPECIALIST_DEFAULT_SAVE_INTERVAL
        default_keep_period = SPECIALIST_DEFAULT_KEEP_PERIOD
        source_path = args.init_params_path

    source_params = _resolve_params_path(source_path)
    if args.stage != "shared":
        if data.root is None:
            raise ValueError(f"The {args.stage} data config has no dataset root; pass --data-root explicitly.")
        shared_assets = source_params.parent / "assets"
        data = dataclasses.replace(
            data,
            root=pathlib.Path(data.root).expanduser().resolve(),
            video_backend=args.video_backend,
            assets=_config.AssetsConfig(
                assets_dir=str(shared_assets),
                asset_id=data.repo_id,
            ),
        )

    num_train_steps = args.num_train_steps if args.num_train_steps is not None else default_steps
    save_interval = args.save_interval if args.save_interval is not None else default_save_interval
    keep_period = args.keep_period if args.keep_period is not None else default_keep_period
    batch_size = args.batch_size if args.batch_size is not None else template.batch_size
    num_workers = args.num_workers if args.num_workers is not None else template.num_workers
    fsdp_devices = args.fsdp_devices if args.fsdp_devices is not None else template.fsdp_devices
    _positive_int(num_train_steps, "num_train_steps")
    _positive_int(save_interval, "save_interval")
    _positive_int(keep_period, "keep_period")
    _positive_int(batch_size, "batch_size")
    _nonnegative_int(num_workers, "num_workers")
    _positive_int(fsdp_devices, "fsdp_devices")

    exp_name = args.exp_name or _default_exp_name(args.stage)
    _validate_exp_name(exp_name)

    schedule = _build_schedule(args, num_train_steps=num_train_steps, template_schedule=template.lr_schedule)
    checkpoint_base_dir = _resolve_directory(
        args.checkpoint_base_dir or pathlib.Path(template.checkpoint_base_dir),
        description="Checkpoint base",
        must_exist=False,
    )
    assets_base_dir = _resolve_directory(
        args.assets_base_dir or pathlib.Path(template.assets_base_dir),
        description="Assets base",
        must_exist=False,
    )

    config = dataclasses.replace(
        template,
        name=config_name,
        exp_name=exp_name,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(source_params)),
        project_name=args.project_name,
        checkpoint_base_dir=str(checkpoint_base_dir),
        assets_base_dir=str(assets_base_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        save_interval=save_interval,
        keep_period=keep_period,
        fsdp_devices=fsdp_devices,
        lr_schedule=schedule,
        overwrite=False,
        resume=False,
        wandb_enabled=args.wandb_enabled,
    )
    return config, source_params


def _resolved_config(args: Args, config: _config.TrainConfig, source_params: pathlib.Path) -> dict:
    data = config.data
    model = config.model
    schedule = config.lr_schedule
    norm_stats_path = _normalization_path(config)
    sources = getattr(data, "sources", ())
    if sources:
        roots = [str(source.root) for source in sources]
        base_image_keys = {pathlib.Path(source.root).name: source.base_image_key for source in sources}
    else:
        roots = [str(data.root)]
        base_image_keys = {
            args.stage: getattr(data, "base_img_name", None) or "observation.images.front",
        }
    return {
        "operation": args.operation,
        "stage": args.stage,
        "no_tactile": True,
        "parameter_transfer": {
            "source_params": str(source_params),
            "source_exists": source_params.is_dir(),
            "loader": type(config.weight_loader).__name__,
            "inherits": "all same-shape checkpoint params, including learned LoRA weights",
            "optimizer": "fresh optimizer state (resume=False)",
        },
        "model": {
            "type": type(model).__name__,
            "pi05": getattr(model, "pi05", None),
            "action_dim": model.action_dim,
            "action_horizon": model.action_horizon,
            "paligemma_variant": getattr(model, "paligemma_variant", None),
            "action_expert_variant": getattr(model, "action_expert_variant", None),
            "max_token_len": model.max_token_len,
            "freeze_filter": repr(config.freeze_filter),
        },
        "data": {
            "config_type": type(data).__name__,
            "roots": roots,
            "repo_id": data.repo_id,
            "balance": getattr(data, "balance", "proportional"),
            "video_backend": getattr(data, "video_backend", None),
            "selected_inputs": ["base_rgb", "wrist_rgb", "state", "action", "language"],
            "base_image_keys": base_image_keys,
            "wrist_image_key": "observation.images.wrist",
            "expected_state_dim": 23,
            "expected_action_dim": 22,
            "tactile_keys": [],
        },
        "normalization": {
            "assets_dir": str(norm_stats_path.parent.parent),
            "norm_stats_path": str(norm_stats_path),
            "norm_stats_exist": norm_stats_path.is_file(),
            "scope": "six-task equal mixture" if args.stage == "shared" else "inherited six-task shared",
            "policy_loads_stats_from_specialist_checkpoint": args.stage != "shared",
        },
        "training": {
            "config_name": config.name,
            "exp_name": config.exp_name,
            "project_name": config.project_name,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "num_train_steps": config.num_train_steps,
            "save_interval": config.save_interval,
            "keep_period": config.keep_period,
            "warmup_steps": schedule.warmup_steps,
            "peak_lr": schedule.peak_lr,
            "decay_steps": schedule.decay_steps,
            "decay_lr": schedule.decay_lr,
            "ema_decay": config.ema_decay,
            "optimizer": type(config.optimizer).__name__,
            "gradient_clip_norm": getattr(config.optimizer, "clip_gradient_norm", None),
            "seed": config.seed,
            "fsdp_devices": config.fsdp_devices,
            "wandb_enabled": config.wandb_enabled,
            "resume": config.resume,
            "overwrite": config.overwrite,
        },
        "output": {
            "checkpoint_base_dir": config.checkpoint_base_dir,
            "checkpoint_dir": str(config.checkpoint_dir),
            "final_checkpoint_step": config.num_train_steps - 1,
        },
    }


def _validate_runtime_inputs(args: Args, config: _config.TrainConfig) -> None:
    data = config.data
    sources = getattr(data, "sources", ())
    if sources:
        roots = [pathlib.Path(source.root) for source in sources]
    else:
        if data.root is None:
            raise ValueError(f"The {args.stage} data config has no dataset root.")
        roots = [pathlib.Path(data.root)]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"LeRobot dataset directory does not exist: {root}")

    if not sources:
        info_path = roots[0] / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot metadata does not exist: {info_path}")
        info = json.loads(info_path.read_text())
        features = info.get("features", {})
        base_image_key = getattr(data, "base_img_name", None) or "observation.images.front"
        required_features = {
            base_image_key,
            "observation.images.wrist",
            "observation.state",
            "action",
        }
        missing_features = sorted(required_features - set(features))
        if missing_features:
            raise ValueError(f"{roots[0]} is missing required specialist features: {missing_features}")
        if info.get("total_tasks") != 1:
            raise ValueError(f"Specialist root must contain exactly one task: {roots[0]}")

    if args.operation == "train":
        try:
            lerobot_version = importlib_metadata.version("lerobot")
        except importlib_metadata.PackageNotFoundError as error:
            raise RuntimeError("LeRobot is not installed in the training environment.") from error
        if lerobot_version != EXPECTED_LEROBOT_VERSION:
            raise RuntimeError(
                f"DexJoCo LeRobot v3 training is pinned to lerobot=={EXPECTED_LEROBOT_VERSION}, "
                f"but the active environment provides {lerobot_version}."
            )
        norm_stats_path = _normalization_path(config)
        if not norm_stats_path.is_file():
            raise FileNotFoundError(
                f"Normalization stats do not exist: {norm_stats_path}. "
                "Run the shared stage once with --operation norm-stats before training."
            )


def main(args: Args) -> None:
    config, source_params = build_config(args)
    print(json.dumps(_resolved_config(args, config, source_params), indent=2, sort_keys=True))
    if args.operation == "inspect":
        return

    _validate_runtime_inputs(args, config)
    if args.operation == "norm-stats":
        if args.data_root is None:
            raise AssertionError("Shared norm-stat computation requires --data-root.")
        compute_balanced_stats = importlib.import_module("compute_balanced_lerobot_norm_stats")
        norm_stats, task_samples = compute_balanced_stats.compute_balanced_stats(args.data_root)
        from openpi.shared import normalize  # noqa: PLC0415

        output_dir = _normalization_path(config).parent
        normalize.save(output_dir, norm_stats)
        for sample in task_samples:
            print(
                f"validated {sample.task}: episodes={sample.episode_count}, "
                f"frames={sample.frame_count}, task_weight=1/6"
            )
        print(f"wrote task-balanced normalization stats to {output_dir / 'norm_stats.json'}")
    elif args.operation == "train":
        train = importlib.import_module("train")
        train.main(config)
    else:
        raise AssertionError(f"Unhandled operation: {args.operation}")


if __name__ == "__main__":
    main(tyro.cli(Args))
