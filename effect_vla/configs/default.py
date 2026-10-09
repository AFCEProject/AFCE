"""Shared config resolution for cache / train / eval scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from effect_vla.constants import (
    ACTION_HORIZON,
    ALL_TASKS,
    DEFAULT_CACHE_ROOT,
    DEFAULT_DINO_CKPT,
    DEFAULT_LEROBOT_ROOT,
    DEFAULT_PILOT_TASKS,
    DEFAULT_RUN_ROOT,
    DEXJOCO_ROOT,
    EFFECT_DIM,
    LAMBDA_EFFECT,
    LAMBDA_GROUNDING,
)


@dataclass
class EffectVLAConfig:
    lerobot_root: Path = Path(DEFAULT_LEROBOT_ROOT)
    cache_root: Path = Path(DEFAULT_CACHE_ROOT)
    run_root: Path = Path(DEFAULT_RUN_ROOT)
    dino_ckpt: Path = Path(DEFAULT_DINO_CKPT)
    dexjoco_root: Path = Path(DEXJOCO_ROOT)
    action_horizon: int = ACTION_HORIZON
    effect_dim: int = EFFECT_DIM
    lambda_effect: float = LAMBDA_EFFECT
    lambda_grounding: float = LAMBDA_GROUNDING
    tasks: tuple[str, ...] = ALL_TASKS
    pilot_tasks: tuple[str, ...] = DEFAULT_PILOT_TASKS


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def default_config() -> EffectVLAConfig:
    return EffectVLAConfig()
