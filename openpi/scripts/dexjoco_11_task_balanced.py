"""Train one task-balanced π0.5 LoRA policy on a DexJoCo task mixture.

Default is all eleven tasks with an equal 1/11 mixture (official multi-task
baseline). Pass ``--tasks`` for a subset; each selected task gets weight 1/N.

Example (Effect pilot, 3 single-arm tasks):

    python scripts/dexjoco_11_task_balanced.py --operation norm-stats \\
      --data-root ../datasets/dexjoco_lerobot_datasets \\
      --init-params-path ../checkpoints/pi05_base_action_dim_44/params \\
      --tasks water_plant pick_bucket hammer_nail

    python scripts/dexjoco_11_task_balanced.py --operation train \\
      --data-root ../datasets/dexjoco_lerobot_datasets \\
      --init-params-path ../checkpoints/pi05_base_action_dim_44/params \\
      --tasks water_plant pick_bucket hammer_nail \\
      --exp-name pilot3_task_balanced_v1 --num-train-steps 30000
"""

from __future__ import annotations

import dataclasses
import importlib
from importlib import metadata as importlib_metadata
import json
import pathlib
from typing import Literal

import tyro

from openpi.training import config as _config
from openpi.training import weight_loaders

Operation = Literal["inspect", "norm-stats", "train"]

CONFIG_NAME = "dexjoco_11_task_balanced"
STRUCTURED_CONFIG_NAME = "dexjoco_11_task_structured"
TEMPLATE_NAME = "multi_task"
DEFAULT_STEPS = 60_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 16
DEFAULT_SAVE_INTERVAL = 10_000
DEFAULT_KEEP_PERIOD = 20_000
EXPECTED_LEROBOT_VERSION = "0.4.4"

# Effect-plan pilot: clear object/world effects, all single-arm.
DEFAULT_PILOT_TASKS = ("water_plant", "pick_bucket", "hammer_nail")

TASK_IMAGE_KEYS = {
    "bimanual_assembly": (
        "observation.images.ego",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    ),
    "bimanual_hanoi": (
        "observation.images.ego",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    ),
    "bimanual_microwave_cook": (
        "observation.images.ego",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    ),
    "bimanual_photograph": (
        "observation.images.ego",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    ),
    "bimanual_unlock_ipad": (
        "observation.images.ego",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    ),
    "click_mouse": (
        "observation.images.ego_right",
        "observation.images.wrist",
        "observation.images.wrist",
    ),
    "fold_glasses": (
        "observation.images.front",
        "observation.images.wrist",
        "observation.images.wrist",
    ),
    "hammer_nail": (
        "observation.images.front",
        "observation.images.wrist",
        "observation.images.wrist",
    ),
    "pick_bucket": (
        "observation.images.front",
        "observation.images.wrist",
        "observation.images.wrist",
    ),
    "pinch_tongs": (
        "observation.images.front",
        "observation.images.wrist",
        "observation.images.wrist",
    ),
    "water_plant": (
        "observation.images.front",
        "observation.images.wrist",
        "observation.images.wrist",
    ),
}

BIMANUAL_TASKS = {
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
}


@dataclasses.dataclass(frozen=True)
class Args:
    operation: Operation = "inspect"
    data_root: pathlib.Path | None = None
    init_params_path: pathlib.Path | None = None
    # Empty → all 11. Pass e.g. water_plant pick_bucket hammer_nail for pilot.
    tasks: tuple[str, ...] = ()
    # Shortcut: same as --tasks water_plant pick_bucket hammer_nail
    pilot3: bool = False
    exp_name: str = "all11_task_balanced_v1"
    checkpoint_base_dir: pathlib.Path = pathlib.Path("checkpoints")
    assets_base_dir: pathlib.Path = pathlib.Path("assets")
    project_name: str = "dexjoco-pi05-task-balanced"
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    num_train_steps: int = DEFAULT_STEPS
    save_interval: int = DEFAULT_SAVE_INTERVAL
    keep_period: int = DEFAULT_KEEP_PERIOD
    fsdp_devices: int = 1
    video_backend: str = "pyav"
    wandb_enabled: bool = False
    structured_hand_state: bool = False
    resume: bool = False


def _selected_tasks(args: Args) -> tuple[str, ...]:
    if args.pilot3 and args.tasks:
        raise ValueError("Pass either --pilot3 or --tasks, not both.")
    if args.pilot3:
        return DEFAULT_PILOT_TASKS
    if not args.tasks:
        return tuple(TASK_IMAGE_KEYS)
    unknown = [t for t in args.tasks if t not in TASK_IMAGE_KEYS]
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}. Known: {sorted(TASK_IMAGE_KEYS)}")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError(f"--tasks must be unique, got {args.tasks}")
    return tuple(args.tasks)


def _config_name(tasks: tuple[str, ...], *, structured_hand_state: bool) -> str:
    all_tasks = tuple(TASK_IMAGE_KEYS)
    if tasks == all_tasks:
        return STRUCTURED_CONFIG_NAME if structured_hand_state else CONFIG_NAME
    if tasks == DEFAULT_PILOT_TASKS:
        base = "dexjoco_pilot3_task_balanced"
    else:
        slug = "__".join(tasks)
        base = f"dexjoco_{len(tasks)}_task_balanced__{slug}"
    return f"{base}_structured" if structured_hand_state else base


def _resolve_directory(path: pathlib.Path, *, description: str, must_exist: bool) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    if must_exist and not resolved.is_dir():
        raise FileNotFoundError(f"{description} directory does not exist: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(f"{description} path is not a directory: {resolved}")
    return resolved


def _sources(data_root: pathlib.Path, tasks: tuple[str, ...]) -> tuple[_config.LeRobotDatasetSource, ...]:
    return tuple(
        _config.LeRobotDatasetSource(
            root=data_root / task,
            base_image_key=TASK_IMAGE_KEYS[task][0],
            wrist_left_image_key=TASK_IMAGE_KEYS[task][1],
            wrist_right_image_key=TASK_IMAGE_KEYS[task][2],
        )
        for task in tasks
    )


def _norm_stats_path(config: _config.TrainConfig) -> pathlib.Path:
    data = config.data
    asset_id = data.assets.asset_id or data.repo_id
    if asset_id is None:
        raise ValueError("The multi-task data config has no normalization asset id.")
    assets_root = pathlib.Path(data.assets.assets_dir).resolve() if data.assets.assets_dir else config.assets_dirs
    return assets_root / asset_id / "norm_stats.json"


def build_config(args: Args) -> tuple[_config.TrainConfig, pathlib.Path, tuple[str, ...]]:
    if args.data_root is None:
        raise ValueError("--data-root is required.")
    if args.init_params_path is None:
        raise ValueError("--init-params-path must point to pi05_base_action_dim_44/params.")
    if not args.exp_name or pathlib.PurePath(args.exp_name).name != args.exp_name:
        raise ValueError("--exp-name must be one non-empty path component.")
    for value, name in (
        (args.batch_size, "batch-size"),
        (args.num_train_steps, "num-train-steps"),
        (args.save_interval, "save-interval"),
        (args.keep_period, "keep-period"),
        (args.fsdp_devices, "fsdp-devices"),
    ):
        if value <= 0:
            raise ValueError(f"--{name} must be positive, got {value}.")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be non-negative, got {args.num_workers}.")
    if not args.video_backend:
        raise ValueError("--video-backend must be non-empty.")

    tasks = _selected_tasks(args)
    data_root = _resolve_directory(
        args.data_root,
        description="DexJoCo LeRobot task parent",
        must_exist=args.operation != "inspect",
    )
    init_params = _resolve_directory(
        args.init_params_path,
        description="Action-dim-44 base checkpoint params",
        must_exist=True,
    )
    if init_params.name != "params":
        raise ValueError("--init-params-path must end in /params.")

    template = _config.get_config(TEMPLATE_NAME)
    if not isinstance(template.data, _config.DualArmDataConfig):
        raise TypeError(f"Expected DualArmDataConfig template, got {type(template.data).__name__}.")
    if template.model.action_dim != 44 or template.model.action_horizon != 30:
        raise ValueError(
            "Official multi_task template must retain action_dim=44 and action_horizon=30; "
            f"got {template.model.action_dim} and {template.model.action_horizon}."
        )

    model = dataclasses.replace(
        template.model,
        structured_hand_state=args.structured_hand_state,
    )
    data = _config.DexJoCoMultiTaskDataConfig(
        repo_id=template.data.repo_id,
        assets=template.data.assets,
        base_config=template.data.base_config,
        sources=_sources(data_root, tasks),
        balance="task",
        video_backend=args.video_backend,
        target_state_dim=46,
        target_action_dim=44,
        structured_hand_state=args.structured_hand_state,
    )
    config = dataclasses.replace(
        template,
        name=_config_name(tasks, structured_hand_state=args.structured_hand_state),
        exp_name=args.exp_name,
        model=model,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(init_params),
            missing_regex=(
                ".*(lora|structured_hand_encoder).*"
                if args.structured_hand_state
                else ".*lora.*"
            ),
        ),
        checkpoint_base_dir=str(
            _resolve_directory(
                args.checkpoint_base_dir,
                description="Checkpoint base",
                must_exist=False,
            )
        ),
        assets_base_dir=str(
            _resolve_directory(
                args.assets_base_dir,
                description="Assets base",
                must_exist=False,
            )
        ),
        project_name=args.project_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        fsdp_devices=args.fsdp_devices,
        overwrite=False,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
    )
    return config, init_params, tasks


def _validate_sources(config: _config.TrainConfig) -> None:
    for source in config.data.create(config.assets_dirs, config.model).sources:
        root = pathlib.Path(source.root)
        info_path = root / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.get("features", {})
        required = {
            source.base_image_key,
            source.wrist_left_image_key,
            source.wrist_right_image_key,
            "observation.state",
            "action",
        }
        missing = sorted(required - set(features))
        if missing:
            raise ValueError(f"{root} is missing required features: {missing}")
        task = root.name
        expected_state_dim, expected_action_dim = (
            (46, 44) if task in BIMANUAL_TASKS else (23, 22)
        )
        if features["observation.state"].get("shape") != [expected_state_dim]:
            raise ValueError(f"{task}: unexpected state feature {features['observation.state']!r}")
        if features["action"].get("shape") != [expected_action_dim]:
            raise ValueError(f"{task}: unexpected action feature {features['action']!r}")
        if info.get("total_tasks") != 1:
            raise ValueError(f"{task}: expected one language task, got {info!r}")


def _summary(
    config: _config.TrainConfig, init_params: pathlib.Path, operation: Operation, tasks: tuple[str, ...]
) -> dict:
    data = config.data.create(config.assets_dirs, config.model)
    schedule = config.lr_schedule
    n = len(tasks)
    return {
        "operation": operation,
        "no_tactile": True,
        "structured_hand_state": getattr(config.model, "structured_hand_state", False),
        "model": {
            "pi05": getattr(config.model, "pi05", None),
            "action_dim": config.model.action_dim,
            "action_horizon": config.model.action_horizon,
            "paligemma_variant": getattr(config.model, "paligemma_variant", None),
            "action_expert_variant": getattr(config.model, "action_expert_variant", None),
        },
        "data": {
            "tasks": list(tasks),
            "task_count": n,
            "balance": data.balance,
            "task_weight": f"1/{n}",
            "state_dim": data.source_target_state_dim,
            "action_dim": data.source_target_action_dim,
            "views": ["base", "wrist_left", "wrist_right"],
            "norm_stats": str(_norm_stats_path(config)),
        },
        "training": {
            "config_name": config.name,
            "steps": config.num_train_steps,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "save_interval": config.save_interval,
            "keep_period": config.keep_period,
            "fsdp_devices": config.fsdp_devices,
            "warmup_steps": getattr(schedule, "warmup_steps", None),
            "peak_lr": getattr(schedule, "peak_lr", None),
            "decay_steps": getattr(schedule, "decay_steps", None),
            "decay_lr": getattr(schedule, "decay_lr", None),
            "wandb_enabled": config.wandb_enabled,
            "resume": config.resume,
            "init_params": str(init_params),
        },
        "output": {
            "checkpoint_dir": str(config.checkpoint_dir),
            "final_checkpoint_step": config.num_train_steps - 1,
        },
    }


def main(args: Args) -> None:
    config, init_params, tasks = build_config(args)
    print(json.dumps(_summary(config, init_params, args.operation, tasks), indent=2, sort_keys=True))
    if args.operation == "inspect":
        return

    _validate_sources(config)
    if args.operation == "norm-stats":
        stats_module = importlib.import_module("compute_balanced_dexjoco_11_norm_stats")
        norm_stats, task_samples = stats_module.compute_balanced_stats(
            args.data_root,
            structured_hand_state=args.structured_hand_state,
            tasks=tasks,
        )
        from openpi.shared import normalize  # noqa: PLC0415

        output_dir = _norm_stats_path(config).parent
        normalize.save(output_dir, norm_stats)
        n = len(task_samples)
        for sample in task_samples:
            print(
                f"validated {sample.task}: episodes={sample.episode_count}, "
                f"frames={sample.frame_count}, task_weight=1/{n}"
            )
        print(f"wrote task-balanced normalization stats to {output_dir / 'norm_stats.json'}")
        return

    try:
        lerobot_version = importlib_metadata.version("lerobot")
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeError("LeRobot is not installed in the training environment.") from error
    if lerobot_version != EXPECTED_LEROBOT_VERSION:
        raise RuntimeError(
            f"DexJoCo training requires lerobot=={EXPECTED_LEROBOT_VERSION}, got {lerobot_version}."
        )
    norm_stats_path = _norm_stats_path(config)
    if not norm_stats_path.is_file():
        raise FileNotFoundError(
            f"Normalization stats do not exist: {norm_stats_path}. Run --operation norm-stats first."
        )
    train = importlib.import_module("train")
    train.main(config)


if __name__ == "__main__":
    main(tyro.cli(Args))
