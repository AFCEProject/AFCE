"""DexJoCo grounding targets: future wrist + fingertip geometry in the current wrist frame.

G_k* = [Δp_wrist, ΔR_wrist_6D, ΔP_tips^{1:4}] per hand, stacked right then left.

Targets never include GT action or raw joint trajectories. They describe
interaction geometry only.
"""

from __future__ import annotations

import numpy as np

from effect_vla.constants import (
    ANCHOR_FRACTIONS,
    GEO_DIM,
    NUM_ANCHORS,
    NUM_FINGERTIPS,
    PER_HAND_GEO_DIM,
)
from effect_vla.data.allegro_fk import split_dexjoco_state, world_fingertips
from effect_vla.data.geometry import pose7_xyzw_to_rt, relative_pose_in_frame, rotmat_to_rot6d


def _hand_geometry(
    tcp_t: np.ndarray | None,
    joints_t: np.ndarray | None,
    tcp_k: np.ndarray | None,
    joints_k: np.ndarray | None,
) -> np.ndarray:
    geo = np.zeros((PER_HAND_GEO_DIM,), dtype=np.float32)
    if tcp_t is None or tcp_k is None or joints_t is None or joints_k is None:
        return geo
    r_t, t_t = pose7_xyzw_to_rt(tcp_t)
    r_k, t_k = pose7_xyzw_to_rt(tcp_k)
    r_rel, t_rel = relative_pose_in_frame(r_t, t_t, r_k, t_k)
    geo[0:3] = t_rel
    geo[3:9] = rotmat_to_rot6d(r_rel)

    tips_t = world_fingertips(tcp_t, joints_t)
    tips_k = world_fingertips(tcp_k, joints_k)
    # Express both clouds in the current wrist frame, then take the delta.
    r_inv = r_t.T
    tips_t_local = (tips_t - t_t) @ r_inv.T
    tips_k_local = (tips_k - t_t) @ r_inv.T
    geo[9 : 9 + NUM_FINGERTIPS * 3] = (tips_k_local - tips_t_local).reshape(-1)
    return geo


def grounding_vector_at(
    state_t: np.ndarray,
    state_k: np.ndarray,
) -> np.ndarray:
    """Return G* at one temporal anchor. Shape (GEO_DIM,)."""
    cur = split_dexjoco_state(state_t)
    fut = split_dexjoco_state(state_k)
    right = _hand_geometry(
        cur["right_tcp"], cur["right_joints"], fut["right_tcp"], fut["right_joints"]
    )
    left = _hand_geometry(
        cur["left_tcp"], cur["left_joints"], fut["left_tcp"], fut["left_joints"]
    )
    return np.concatenate([right, left], axis=0).astype(np.float32)


def grounding_targets_episode(
    states: np.ndarray,
    *,
    action_horizon: int,
    anchor_fractions: tuple[float, ...] = ANCHOR_FRACTIONS,
) -> np.ndarray:
    """states: (T, D) -> (T, K, GEO_DIM). End-of-episode anchors clamp to T-1."""
    states = np.asarray(states, dtype=np.float32)
    t_len = states.shape[0]
    k = len(anchor_fractions)
    out = np.zeros((t_len, k, GEO_DIM), dtype=np.float32)
    offsets = [max(1, int(round(frac * action_horizon))) for frac in anchor_fractions]
    for t in range(t_len):
        for ki, dt in enumerate(offsets):
            tk = min(t + dt, t_len - 1)
            out[t, ki] = grounding_vector_at(states[t], states[tk])
    return out


def hand_presence_mask(state: np.ndarray) -> np.ndarray:
    """(2,) bool [right, left]."""
    parsed = split_dexjoco_state(state)
    return np.asarray(parsed["hand_presence"], dtype=np.bool_)
