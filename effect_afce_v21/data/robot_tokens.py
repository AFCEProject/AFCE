"""Per-timestep robot ΔS: R(S_i, S_{i+1}) in consecutive relative frame."""

from __future__ import annotations

import numpy as np

from effect_afce_v21.constants import ACTION_HORIZON, DELTA_S_DIM, NUM_FINGERTIPS
from effect_coupled_v4.data.robot_descriptor import hand_checkpoint_delta
from effect_vla.data.allegro_fk import split_dexjoco_state


def step_delta_s(state_i: np.ndarray, state_j: np.ndarray) -> dict[str, np.ndarray]:
    a = split_dexjoco_state(state_i)
    e = split_dexjoco_state(state_j)
    tcp_a, j_a = a["right_tcp"], a["right_joints"]
    tcp_e, j_e = e["right_tcp"], e["right_joints"]
    if tcp_a is None or j_a is None or tcp_e is None or j_e is None:
        root = np.zeros((9,), np.float32)
        tips = np.zeros((NUM_FINGERTIPS, 3), np.float32)
        joint = np.zeros((16,), np.float32)
    else:
        d = hand_checkpoint_delta(tcp_a, j_a, tcp_e, j_e)
        root = np.concatenate([d.wrist_pos_rel, d.wrist_rot6d], axis=0).astype(np.float32)
        tips = d.tip_local_delta.astype(np.float32)
        joint = d.joint_delta.astype(np.float32)
    delta_s = np.concatenate([root, tips.reshape(-1), joint], axis=0).astype(np.float32)
    assert delta_s.shape[0] == DELTA_S_DIM
    return {"root_feat": root, "tip_feat": tips, "joint_feat": joint, "delta_s": delta_s}


def robot_horizon_delta_s(states: np.ndarray, t: int, horizon: int = ACTION_HORIZON) -> dict[str, np.ndarray]:
    """states must cover [t, t+H] inclusive. Returns (H, ...)."""
    roots, tips, joints, targets = [], [], [], []
    for i in range(horizon):
        pack = step_delta_s(states[t + i], states[t + i + 1])
        roots.append(pack["root_feat"])
        tips.append(pack["tip_feat"])
        joints.append(pack["joint_feat"])
        targets.append(pack["delta_s"])
    return {
        "root_feat": np.stack(roots, axis=0),
        "tip_feat": np.stack(tips, axis=0),
        "joint_feat": np.stack(joints, axis=0),
        "delta_s": np.stack(targets, axis=0),
    }
