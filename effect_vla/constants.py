"""v1 defaults from the DexJoCo Task-Effect Grounded π0.5 plan."""

from __future__ import annotations

import os
from pathlib import Path

# Temporal / action window. Must match the existing DexJoCo π0.5 baseline.
ACTION_HORIZON = 30
EFFECT_HORIZON = ACTION_HORIZON
NUM_ANCHORS = 2
ANCHOR_FRACTIONS = (0.5, 1.0)  # τ = 0.5 H, H

# Effect representation.
EFFECT_DIM = 512
DINO_IMAGE_SIZE = 224
DINO_PATCH_SIZE = 16
MATCH_TEMPERATURE = 0.07
DYNAMIC_TOP_RATIO = 0.20
DYNAMIC_MIN_PATCHES = 8
AGGREGATION_BETA = 8.0
PROJECTION_SEED = 0

# Robot-motion suppression is training-only.
ROBOT_MASK_THRESHOLD = 0.3

# Grounding: DexJoCo Allegro has 4 fingertips per hand.
# The plan writes "5 fingertips" generically; v1 uses the 4 real Allegro tips
# (index, middle, ring, thumb) plus wrist pose. A zero pad is kept so the
# geometry vector stays 3+6+15=24 per hand if a 5th slot is needed later.
NUM_FINGERTIPS = 4
WRIST_POS_DIM = 3
WRIST_ROT6D_DIM = 6
TIP_POS_DIM = NUM_FINGERTIPS * 3
PER_HAND_GEO_DIM = WRIST_POS_DIM + WRIST_ROT6D_DIM + TIP_POS_DIM  # 21
NUM_HANDS = 2
GEO_DIM = NUM_HANDS * PER_HAND_GEO_DIM  # 42
GROUNDING_DIM = 512

# Losses.
LAMBDA_FM = 1.0
LAMBDA_EFFECT = 0.2
LAMBDA_GROUNDING = 0.1
SMOOTH_L1_BETA = 0.05
EFFECT_SMOOTH_L1_WEIGHT = 0.1

# Training schedule (additional steps on top of a trained DexJoCo π0.5 ckpt).
BOOTSTRAP_STEPS = 3000
JOINT_FINETUNE_STEPS = 10000
TOTAL_EXTRA_STEPS = BOOTSTRAP_STEPS + JOINT_FINETUNE_STEPS  # 13K, within 10–15K
PILOT_BOOTSTRAP_STEPS = 3000
PILOT_JOINT_STEPS = 5000
PILOT_TOTAL_STEPS = PILOT_BOOTSTRAP_STEPS + PILOT_JOINT_STEPS

NEW_MODULE_LR = 1.0e-4
ACTION_LR = 2.0e-5
VLM_LR = 1.0e-5
GRAD_CLIP = 1.0

# Cameras. Policy cameras stay on the baseline; Effect uses static/front only.
SINGLE_ARM_EFFECT_CAMERA = "observation.images.front"
BIMANUAL_EFFECT_CAMERA = "observation.images.ego"
CLICK_MOUSE_EFFECT_CAMERA = "observation.images.ego_right"

SINGLE_ARM_TASKS = (
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
BIMANUAL_TASKS = (
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
)
ALL_TASKS = SINGLE_ARM_TASKS + BIMANUAL_TASKS

# First-round 3-task pilot: clear object/world effects, mixed difficulty.
DEFAULT_PILOT_TASKS = ("water_plant", "pick_bucket", "hammer_nail")

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_str(name: str, default: Path) -> str:
    return os.environ.get(name, str(default))


DEXJOCO_ROOT = _env_str("AFCE_ROOT", _REPO_ROOT)
DEFAULT_LEROBOT_ROOT = _env_str(
    "AFCE_DATA",
    Path(DEXJOCO_ROOT) / "datasets" / "dexjoco_lerobot_datasets",
)
DEFAULT_DINO_CKPT = _env_str(
    "AFCE_DINO",
    Path(DEXJOCO_ROOT) / "third_party" / "weights" / "dinov3",
)
DEFAULT_CACHE_ROOT = _env_str(
    "AFCE_EFFECT_CACHE",
    Path(DEXJOCO_ROOT) / "outputs" / "effect_vla" / "cache",
)
DEFAULT_RUN_ROOT = _env_str(
    "AFCE_EFFECT_RUNS",
    Path(DEXJOCO_ROOT) / "outputs" / "effect_vla" / "runs",
)
