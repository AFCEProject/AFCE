from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import yaml

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

with open("config.yaml") as f:
    config = yaml.safe_load(f)

WANDB_ENABLED = config["wandb_enabled"]
PRETRAINED_MODEL_PATH = config["pretrained_model_path"]
PRETRAINED_MODEL_ACTION_DIM_44_PATH = config["pretrained_model_action_dim_44_path"]
DATASET_ROOT = Path(config["dataset_root"])
RAND_FULL_DATASET_ROOT = Path(config["rand_full_dataset_root"])
CKPTS_ROOT = Path(config["ckpts_root"])
RAND_FULL_CKPTS_ROOT = Path(config["rand_full_ckpts_root"])
BATCH_SIZE = config["batch_size"]
SINGLE_ARM_STEPS = config["single_arm_steps"]
DUAL_ARM_STEPS = config["dual_arm_steps"]
EFFECT_CACHE_ROOT = Path(config.get("effect_cache_root", "../outputs/effect_vla/cache"))
EFFECT_INIT_CHECKPOINT_ROOT = Path(config.get("effect_init_checkpoint_root", config["ckpts_root"]))
EFFECT_EXTRA_STEPS = int(config.get("effect_extra_steps", 13000))
EFFECT_BOOTSTRAP_STEPS = int(config.get("effect_bootstrap_steps", 3000))
EFFECT_PILOT_TASKS = tuple(config.get("effect_pilot_tasks", ["water_plant", "pick_bucket", "hammer_nail"]))


@dataclass
class DexJoCoConfig:
    name: str
    checkpoint_base_dir: str
    data_root: Path
    single_arm: bool
    base_img_name: str | None = None
    wrist_left_img_name: str | None = None
    wrist_right_img_name: str | None = None


TrainConfigs: list[DexJoCoConfig] = [
    # rand_obj datasets
    DexJoCoConfig(
        name="bimanual_assembly",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/bimanual_assembly"),
        single_arm=False,
    ),
    DexJoCoConfig(
        name="bimanual_microwave_cook",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/bimanual_microwave_cook"),
        single_arm=False,
    ),
    DexJoCoConfig(
        name="bimanual_unlock_ipad",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/bimanual_unlock_ipad"),
        single_arm=False,
    ),
    DexJoCoConfig(
        name="bimanual_hanoi",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/bimanual_hanoi"),
        single_arm=False,
    ),
    DexJoCoConfig(
        name="bimanual_photograph",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/bimanual_photograph"),
        single_arm=False,
    ),
    DexJoCoConfig(
        name="fold_glasses",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/fold_glasses"),
        single_arm=True,
    ),
    DexJoCoConfig(
        name="pick_bucket",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/pick_bucket"),
        single_arm=True,
    ),
    DexJoCoConfig(
        name="water_plant",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/water_plant"),
        single_arm=True,
    ),
    DexJoCoConfig(
        name="click_mouse",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/click_mouse"),
        single_arm=True,
        base_img_name="observation.images.ego_right",
    ),
    DexJoCoConfig(
        name="hammer_nail",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/hammer_nail"),
        single_arm=True,
    ),
    DexJoCoConfig(
        name="pinch_tongs",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/pinch_tongs"),
        single_arm=True,
    ),

    # rand_full datasets
    DexJoCoConfig(
        name="bimanual_assembly_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/bimanual_assembly"),
        single_arm=False,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="bimanual_microwave_cook_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/bimanual_microwave_cook"),
        single_arm=False,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="bimanual_unlock_ipad_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/bimanual_unlock_ipad"),
        single_arm=False,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="bimanual_hanoi_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/bimanual_hanoi"),
        single_arm=False,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="bimanual_photograph_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/bimanual_photograph"),
        single_arm=False,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="fold_glasses_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/fold_glasses"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="pick_bucket_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/pick_bucket"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="water_plant_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/water_plant"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="click_mouse_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/click_mouse"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="hammer_nail_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/hammer_nail"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="pinch_tongs_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/pinch_tongs"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    DexJoCoConfig(
        name="multi_task",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path("path/to/multi_task/dataset"),
        single_arm=False,
        base_img_name="observation.images.base",
        wrist_left_img_name="observation.images.wrist1",
        wrist_right_img_name="observation.images.wrist2",
    ),
]


def make_single_arm_config(cfg: DexJoCoConfig):
    # import in function to avoid circular import
    from .config import DataConfig  # noqa: PLC0415
    from .config import SingleArmDataConfig  # noqa: PLC0415
    from .config import TrainConfig  # noqa: PLC0415

    return TrainConfig(
        name=cfg.name,
        exp_name=cfg.name + datetime.now().strftime("%Y%m%d"),  # noqa: DTZ005
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=30,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=250,
        ),
        data=SingleArmDataConfig(
            root=cfg.data_root,
            repo_id="local_repo",
            base_img_name=cfg.base_img_name,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=BATCH_SIZE,
        num_workers=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=30,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=250,
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_MODEL_PATH),
        num_train_steps=SINGLE_ARM_STEPS,
        save_interval=10000,
        wandb_enabled=WANDB_ENABLED,
        checkpoint_base_dir=cfg.checkpoint_base_dir,
    )


def make_dual_arm_config(cfg: DexJoCoConfig):
    # import in function to avoid circular import
    from .config import DataConfig  # noqa: PLC0415
    from .config import DualArmDataConfig  # noqa: PLC0415
    from .config import TrainConfig  # noqa: PLC0415

    return TrainConfig(
        name=cfg.name,
        exp_name=cfg.name + datetime.now().strftime("%Y%m%d"),  # noqa: DTZ005
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=44,
            action_horizon=30,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=250,
        ),
        data=DualArmDataConfig(
            root=cfg.data_root,
            repo_id="local_repo",
            base_img_name=cfg.base_img_name,
            wrist_left_img_name=cfg.wrist_left_img_name,
            wrist_right_img_name=cfg.wrist_right_img_name,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=BATCH_SIZE,
        num_workers=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=44,
            action_horizon=30,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=250,
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_MODEL_ACTION_DIM_44_PATH),
        num_train_steps=DUAL_ARM_STEPS,
        save_interval=10000,
        wandb_enabled=WANDB_ENABLED,
        checkpoint_base_dir=cfg.checkpoint_base_dir,
    )


def resolve_trained_params(task_name: str) -> str:
    """Locate a trained DexJoCo π0.5 params dir for continue / effect init.

    Search order:
      1. ``effect_init_checkpoints[task]`` in config.yaml if present
      2. ``<effect_init_checkpoint_root>/<task>/params``
      3. newest ``<effect_init_checkpoint_root>/<task>/**/params``
    """
    mapping = config.get("effect_init_checkpoints") or {}
    if task_name in mapping:
        return str(Path(mapping[task_name]))
    root = EFFECT_INIT_CHECKPOINT_ROOT / task_name
    direct = root / "params"
    if direct.exists():
        return str(direct)
    candidates = sorted(root.glob("**/params"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if candidates:
        return str(candidates[-1])
    return str(direct)


def _with_effect_model(base_model: pi0_config.Pi0Config, **overrides) -> pi0_config.Pi0Config:
    defaults = dict(
        effect_grounding=True,
        lambda_effect=0.2,
        lambda_grounding=0.1,
        condition_effect=True,
        use_grounding=True,
    )
    defaults.update(overrides)
    return replace(base_model, **defaults)


def make_continue_config(train_cfg, cfg: DexJoCoConfig):
    return replace(
        train_cfg,
        name=f"{cfg.name}_continue",
        exp_name=cfg.name + "_continue" + datetime.now().strftime("%Y%m%d"),  # noqa: DTZ005
        weight_loader=weight_loaders.CheckpointWeightLoader(
            resolve_trained_params(cfg.name), missing_regex=".*"
        ),
        num_train_steps=EFFECT_EXTRA_STEPS,
        save_interval=2000,
    )


def make_effect_config(train_cfg, cfg: DexJoCoConfig, *, bootstrap: bool):
    model = _with_effect_model(train_cfg.model)
    data = replace(train_cfg.data, effect_cache_dir=EFFECT_CACHE_ROOT / cfg.name)
    suffix = "_effect_v1_bootstrap" if bootstrap else "_effect_v1"
    return replace(
        train_cfg,
        name=f"{cfg.name}{suffix}",
        exp_name=cfg.name + suffix + datetime.now().strftime("%Y%m%d"),  # noqa: DTZ005
        model=model,
        data=data,
        freeze_filter=model.get_bootstrap_freeze_filter() if bootstrap else model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            resolve_trained_params(cfg.name), missing_regex=".*"
        ),
        num_train_steps=EFFECT_BOOTSTRAP_STEPS if bootstrap else EFFECT_EXTRA_STEPS,
        save_interval=1000 if bootstrap else 2000,
    )


def make_effect_only_config(train_cfg, cfg: DexJoCoConfig):
    """Ablation 1: Effect → Action, no grounding loss / tokens."""
    model = _with_effect_model(train_cfg.model, use_grounding=False, lambda_grounding=0.0)
    data = replace(train_cfg.data, effect_cache_dir=EFFECT_CACHE_ROOT / cfg.name)
    return replace(
        train_cfg,
        name=f"{cfg.name}_effect_only",
        exp_name=cfg.name + "_effect_only" + datetime.now().strftime("%Y%m%d"),  # noqa: DTZ005
        model=model,
        data=data,
        freeze_filter=model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            resolve_trained_params(cfg.name), missing_regex=".*"
        ),
        num_train_steps=EFFECT_EXTRA_STEPS,
        save_interval=2000,
    )


def make_no_effect_conditioning_config(train_cfg, cfg: DexJoCoConfig):
    """Ablation 3: still train L_E but Action Expert cannot see Ê."""
    model = _with_effect_model(train_cfg.model, condition_effect=False)
    data = replace(train_cfg.data, effect_cache_dir=EFFECT_CACHE_ROOT / cfg.name)
    return replace(
        train_cfg,
        name=f"{cfg.name}_no_effect_cond",
        exp_name=cfg.name + "_no_effect_cond" + datetime.now().strftime("%Y%m%d"),  # noqa: DTZ005
        model=model,
        data=data,
        freeze_filter=model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            resolve_trained_params(cfg.name), missing_regex=".*"
        ),
        num_train_steps=EFFECT_EXTRA_STEPS,
        save_interval=2000,
    )


def get_dexjoco_configs():
    cfgs = []
    for cfg in TrainConfigs:
        if cfg.single_arm:
            train_cfg = make_single_arm_config(cfg)
        else:
            train_cfg = make_dual_arm_config(cfg)
        cfgs.append(train_cfg)
        # Effect / continue variants only for the 11 rand-obj tasks.
        if cfg.name.endswith("_rand_full") or cfg.name == "multi_task":
            continue
        cfgs.append(make_continue_config(train_cfg, cfg))
        cfgs.append(make_effect_config(train_cfg, cfg, bootstrap=True))
        cfgs.append(make_effect_config(train_cfg, cfg, bootstrap=False))
        if cfg.name in EFFECT_PILOT_TASKS:
            cfgs.append(make_effect_only_config(train_cfg, cfg))
            cfgs.append(make_no_effect_conditioning_config(train_cfg, cfg))
    return cfgs
