"""Robot-motion suppression for Effect targets.

The mask is used only while constructing E*. It never enters the policy
input and is not required at inference.

Preferred source: MuJoCo segmentation of robot/hand geoms, downsampled onto
the DINO patch grid. When simulator state is unavailable the cache writes a
zero mask (no suppression) and records ``mask_source=none``.
"""

from __future__ import annotations

import numpy as np

from effect_vla.constants import ROBOT_MASK_THRESHOLD

_ROBOT_NAME_TOKENS = (
    "panda",
    "allegro",
    "gripper",
    "hand",
    "finger",
    "palm",
    "wrist",
    "forearm",
    "link_",
    "ff_",
    "mf_",
    "rf_",
    "th_",
    "thumb",
    "robot",
)


def is_robot_geom_name(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return any(tok in lowered for tok in _ROBOT_NAME_TOKENS)


def downsample_mask_to_patches(
    mask_hw: np.ndarray,
    patch_grid: int,
    *,
    threshold: float = ROBOT_MASK_THRESHOLD,
) -> np.ndarray:
    """mask_hw: (H, W) in {0,1} or [0,1] → (patch_grid**2,) float32 robot occupancy."""
    mask_hw = np.asarray(mask_hw, dtype=np.float32)
    if mask_hw.ndim != 2:
        raise ValueError(f"Expected HxW mask, got {mask_hw.shape}")
    h, w = mask_hw.shape
    # Average-pool onto the DINO grid.
    ph, pw = h // patch_grid, w // patch_grid
    if ph < 1 or pw < 1:
        resized = _resize_nearest(mask_hw, patch_grid, patch_grid)
        return (resized.reshape(-1) >= threshold).astype(np.float32)
    cropped = mask_hw[: ph * patch_grid, : pw * patch_grid]
    pooled = cropped.reshape(patch_grid, ph, patch_grid, pw).mean(axis=(1, 3))
    return pooled.reshape(-1).astype(np.float32)


def apply_robot_suppression(delta: np.ndarray, robot_mask: np.ndarray) -> np.ndarray:
    """delta: (..., N, D), robot_mask: (..., N) in [0, 1]. Returns (1-M_R) ⊙ delta."""
    keep = (1.0 - np.clip(robot_mask, 0.0, 1.0))[..., None]
    return delta * keep


def _resize_nearest(mask: np.ndarray, nh: int, nw: int) -> np.ndarray:
    ys = (np.linspace(0, mask.shape[0] - 1, nh)).astype(np.int64)
    xs = (np.linspace(0, mask.shape[1] - 1, nw)).astype(np.int64)
    return mask[ys][:, xs]


def robot_geom_ids_from_model(model) -> np.ndarray:
    """Collect MuJoCo geom ids whose names look like robot/hand parts."""
    import mujoco

    ids = []
    for gid in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if is_robot_geom_name(name):
            ids.append(gid)
    return np.asarray(ids, dtype=np.int32)


def render_robot_mask_mujoco(
    model,
    data,
    *,
    camera: str | int,
    height: int,
    width: int,
    robot_geom_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Return (H, W) float32 mask using MuJoCo segmentation rendering."""
    import mujoco

    robot_geom_ids = robot_geom_ids if robot_geom_ids is not None else robot_geom_ids_from_model(model)
    robot_set = set(int(g) for g in robot_geom_ids.tolist())
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera)
        seg = renderer.render()
    finally:
        renderer.close()
    # seg[..., 0] is geom id; background is -1.
    geom_ids = np.asarray(seg[..., 0], dtype=np.int32)
    mask = np.zeros(geom_ids.shape, dtype=np.float32)
    for gid in robot_set:
        mask[geom_ids == gid] = 1.0
    return mask
