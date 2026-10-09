"""AFCE v2.1 Final constants — Unified Information-Preserving Effect."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


DEXJOCO_ROOT = _env_path("AFCE_ROOT", _REPO_ROOT)
DEFAULT_LEROBOT_ROOT = _env_path(
    "C01_DATA",
    DEXJOCO_ROOT / "datasets" / "dexjoco_lerobot_datasets",
)
DEFAULT_DINO_CACHE_ROOT = _env_path(
    "AFCE_EFFECT_CACHE",
    DEXJOCO_ROOT / "outputs" / "effect_vla" / "cache",
)
DEFAULT_AFCE_CACHE_ROOT = _env_path(
    "AFCE_CACHE_ROOT",
    DEXJOCO_ROOT / "outputs" / "effect_afce_v21" / "cache",
)
DEFAULT_AFCE_RUN_ROOT = _env_path(
    "AFCE_RUN_ROOT",
    DEXJOCO_ROOT / "outputs" / "effect_afce_v21" / "runs",
)

ACTION_HORIZON = 30  # H; E tokens = H
NUM_WORLD_INTERVALS = 4  # K_W sparse real DINO transitions
BRANCH_DIM = 64
EFFECT_DIM = 256
N_WORLD_REGIONS = 8  # 6 dynamic + 2 reference
N_DYNAMIC_REGIONS = 6
N_REFERENCE_REGIONS = 2
NUM_FINGERTIPS = 4
ACTION_DIM = 22
STATE_DIM = 23
DELTA_S_DIM = 37  # root9 + tip12 + joint16
DINO_DIM = 768
PCA_DIM = 64
W_RAW_DIM = 72  # df64 + disp2 + xy2 + conf + is_dyn + t0 + t1
Y_W_DIM = 66  # df64 + disp2
PATCH_GRID = 14

MATCH_TEMPERATURE = 0.07
ROBOT_SUPPRESSION = 0.9

LOSS_W_S = 0.2
LOSS_W_V = 0.2
LAMBDA_START = 0.05
LAMBDA_RAMP_STEPS = 1000
MAX_STEPS_DEFAULT = 15000

DEFAULT_PILOT_TASKS = ("water_plant", "pick_bucket", "hammer_nail")
SCHEMA_VERSION = "afce_v21_unified_info_preserving"
