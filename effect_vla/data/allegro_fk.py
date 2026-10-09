"""Approximate Allegro-hand FK in the current wrist/palm frame.

DexJoCo state stores 16 Allegro joint angles per hand. Fingertip positions
are reconstructed with the link lengths taken from
``dexjoco/sim/envs/xmls/wonik_allegro/right_hand.xml``. The reconstruction is
used only to form relative geometry targets; it is not a MuJoCo replica.

Joint layout (16, matching DexJoCo tactile order):
    index 0-3, middle 4-7, ring 8-11, thumb 12-15
Each finger is a 4-DoF serial chain. Rotations are about +Y except the
finger-base abduction about +X for the three fingers and a distinct thumb
base pose.
"""

from __future__ import annotations

import numpy as np

from effect_vla.data.geometry import quat_to_rotmat

# Finger base pose in palm frame: (xyz, xyzw quat). From right_hand.xml.
_FF_BASE_POS = np.array([0.0, 0.0435, -0.001542], dtype=np.float32)
_MF_BASE_POS = np.array([0.0, 0.0, 0.0007], dtype=np.float32)
_RF_BASE_POS = np.array([0.0, -0.0435, -0.001542], dtype=np.float32)
_TH_BASE_POS = np.array([-0.0182, 0.019333, -0.045987], dtype=np.float32)

# Identity-like bases; small x-rotations on index/ring from XML (~5 deg).
_FF_BASE_QUAT_WXYZ = np.array([0.999048, -0.0436194, 0.0, 0.0], dtype=np.float32)
_MF_BASE_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_RF_BASE_QUAT_WXYZ = np.array([0.999048, 0.0436194, 0.0, 0.0], dtype=np.float32)
# Thumb is mounted at ~90 deg; a right-handed +Y then +Z approximation.
_TH_BASE_QUAT_WXYZ = np.array([0.477, 0.075, -0.075, 0.872], dtype=np.float32)

# Serial-link translations along +Z of the previous joint frame.
_FINGER_LINKS = np.array([0.0164, 0.0540, 0.0384, 0.0267], dtype=np.float32)
_THUMB_LINKS = np.array([0.0177, 0.0514, 0.0384, 0.0423], dtype=np.float32)


def _rotx(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    z = np.zeros_like(a)
    o = np.ones_like(a)
    row0 = np.stack([o, z, z], axis=-1)
    row1 = np.stack([z, c, -s], axis=-1)
    row2 = np.stack([z, s, c], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def _roty(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    z = np.zeros_like(a)
    o = np.ones_like(a)
    row0 = np.stack([c, z, s], axis=-1)
    row1 = np.stack([z, o, z], axis=-1)
    row2 = np.stack([-s, z, c], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def _chain_fk(
    joints4: np.ndarray,
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    links: np.ndarray,
    *,
    first_axis: str,
) -> np.ndarray:
    """Return fingertip xyz in palm frame. joints4: (..., 4)."""
    r = quat_to_rotmat(base_quat_wxyz)
    t = np.broadcast_to(base_pos, joints4.shape[:-1] + (3,)).copy()
    axes = (first_axis, "y", "y", "y")
    for i, axis in enumerate(axes):
        ang = joints4[..., i]
        delta = _rotx(ang) if axis == "x" else _roty(ang)
        r = np.einsum("...ij,...jk->...ik", r, delta)
        t = t + np.einsum("...ij,...j->...i", r, np.array([0.0, 0.0, float(links[i])], dtype=np.float32))
    return t.astype(np.float32)


def fingertip_positions_palm(joints16: np.ndarray) -> np.ndarray:
    """joints16: (..., 16) -> (..., 4, 3) index/middle/ring/thumb in palm frame."""
    joints16 = np.asarray(joints16, dtype=np.float32)
    ff = _chain_fk(joints16[..., 0:4], _FF_BASE_POS, _FF_BASE_QUAT_WXYZ, _FINGER_LINKS, first_axis="x")
    mf = _chain_fk(joints16[..., 4:8], _MF_BASE_POS, _MF_BASE_QUAT_WXYZ, _FINGER_LINKS, first_axis="x")
    rf = _chain_fk(joints16[..., 8:12], _RF_BASE_POS, _RF_BASE_QUAT_WXYZ, _FINGER_LINKS, first_axis="x")
    th = _chain_fk(joints16[..., 12:16], _TH_BASE_POS, _TH_BASE_QUAT_WXYZ, _THUMB_LINKS, first_axis="y")
    return np.stack([ff, mf, rf, th], axis=-2)


def split_dexjoco_state(state: np.ndarray) -> dict[str, np.ndarray]:
    """Parse 23-D single-arm or 46-D canonical bimanual state.

    23-D: [right_tcp7, right_joints16]
    46-D: [right_tcp7, left_tcp7, right_joints16, left_joints16]
    """
    state = np.asarray(state, dtype=np.float32)
    dim = state.shape[-1]
    if dim == 23:
        return {
            "right_tcp": state[..., :7],
            "left_tcp": None,
            "right_joints": state[..., 7:23],
            "left_joints": None,
            "hand_presence": np.asarray([True, False]),
        }
    if dim == 46:
        return {
            "right_tcp": state[..., :7],
            "left_tcp": state[..., 7:14],
            "right_joints": state[..., 14:30],
            "left_joints": state[..., 30:46],
            "hand_presence": np.asarray([True, True]),
        }
    if dim >= 7 + 16:
        # Padded π0.5 state (e.g. 32-D). Recover the leading 23-D layout.
        return {
            "right_tcp": state[..., :7],
            "left_tcp": None,
            "right_joints": state[..., 7:23],
            "left_joints": None,
            "hand_presence": np.asarray([True, False]),
        }
    raise ValueError(f"Unsupported DexJoCo state dim {dim}")


def world_fingertips(tcp7: np.ndarray, joints16: np.ndarray) -> np.ndarray:
    """(..., 4, 3) fingertip xyz in the world/base frame."""
    from effect_vla.data.geometry import pose7_xyzw_to_rt, transform_points

    r, t = pose7_xyzw_to_rt(tcp7)
    tips_palm = fingertip_positions_palm(joints16)
    return transform_points(r, t, tips_palm)
