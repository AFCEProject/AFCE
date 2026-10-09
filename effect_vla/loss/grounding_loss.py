"""Grounding geometry loss: wrist translation + 6D rotation + fingertip xyz.

L_G = L_wp + L_wr + L_tip. Weight λ_G is applied outside this module.
"""

from __future__ import annotations

import numpy as np

from effect_vla.constants import NUM_FINGERTIPS, PER_HAND_GEO_DIM
from effect_vla.data.geometry import rot6d_to_rotmat


def smooth_l1(pred: np.ndarray, target: np.ndarray, beta: float = 1.0) -> np.ndarray:
    diff = np.abs(pred - target)
    return np.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


def split_geo(geo: np.ndarray) -> dict[str, np.ndarray]:
    """geo: (..., GEO_DIM) with right then left hands of PER_HAND_GEO_DIM."""
    geo = np.asarray(geo, dtype=np.float32)
    hands = np.stack(np.split(geo, 2, axis=-1), axis=-2)  # (..., 2, 21)
    pos = hands[..., 0:3]
    rot6d = hands[..., 3:9]
    tips = hands[..., 9 : 9 + NUM_FINGERTIPS * 3].reshape(*hands.shape[:-1], NUM_FINGERTIPS, 3)
    return {"wrist_pos": pos, "wrist_rot6d": rot6d, "tips": tips}


def geodesic_rotation_error(pred_6d: np.ndarray, target_6d: np.ndarray) -> np.ndarray:
    r_pred = rot6d_to_rotmat(pred_6d)
    r_tgt = rot6d_to_rotmat(target_6d)
    r_rel = np.einsum("...ji,...jk->...ik", r_pred, r_tgt)
    # angle = arccos((tr(R)-1)/2)
    tr = r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2]
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cos)


def grounding_loss_numpy(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    p = split_geo(pred)
    t = split_geo(target)
    l_wp = smooth_l1(p["wrist_pos"], t["wrist_pos"]).mean()
    l_wr = geodesic_rotation_error(p["wrist_rot6d"], t["wrist_rot6d"]).mean()
    l_tip = smooth_l1(p["tips"], t["tips"]).mean()
    return {
        "loss": np.asarray(l_wp + l_wr + l_tip, dtype=np.float32),
        "wrist_pos_err": np.asarray(np.linalg.norm(p["wrist_pos"] - t["wrist_pos"], axis=-1).mean(), dtype=np.float32),
        "wrist_rot_err": np.asarray(l_wr, dtype=np.float32),
        "tip_err": np.asarray(np.linalg.norm(p["tips"] - t["tips"], axis=-1).mean(), dtype=np.float32),
    }


def grounding_loss_jax(pred, target):
    import jax.numpy as jnp

    target = jax_stop(target)

    def _split(geo):
        hands = geo.reshape(geo.shape[:-1] + (2, PER_HAND_GEO_DIM))
        pos = hands[..., 0:3]
        rot6d = hands[..., 3:9]
        tips = hands[..., 9:].reshape(hands.shape[:-1] + (NUM_FINGERTIPS, 3))
        return pos, rot6d, tips

    def _rot6d_to_R(d):
        a1 = d[..., :3]
        a2 = d[..., 3:6]
        b1 = a1 / jnp.maximum(jnp.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
        a2p = a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = a2p / jnp.maximum(jnp.linalg.norm(a2p, axis=-1, keepdims=True), 1e-8)
        b3 = jnp.cross(b1, b2)
        return jnp.stack([b1, b2, b3], axis=-1)

    p_pos, p_rot, p_tip = _split(pred)
    t_pos, t_rot, t_tip = _split(target)
    diff_p = jnp.abs(p_pos - t_pos)
    l_wp = jnp.where(diff_p < 1.0, 0.5 * diff_p * diff_p, diff_p - 0.5).mean()
    rp, rt = _rot6d_to_R(p_rot), _rot6d_to_R(t_rot)
    r_rel = jnp.einsum("...ji,...jk->...ik", rp, rt)
    tr = r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2]
    cos = jnp.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    l_wr = jnp.arccos(cos).mean()
    diff_t = jnp.abs(p_tip - t_tip)
    l_tip = jnp.where(diff_t < 1.0, 0.5 * diff_t * diff_t, diff_t - 0.5).mean()
    wrist_err = jnp.linalg.norm(p_pos - t_pos, axis=-1).mean()
    tip_err = jnp.linalg.norm(p_tip - t_tip, axis=-1).mean()
    return l_wp + l_wr + l_tip, {
        "wrist_pos_err": wrist_err,
        "wrist_rot_err": l_wr,
        "tip_err": tip_err,
    }


def jax_stop(x):
    import jax

    return jax.lax.stop_gradient(x)
