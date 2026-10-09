"""Effect prediction loss. Target is stop-grad.

L_E = mean_k [ 1 - cos(Ê_k, E_k*) + 0.1 SmoothL1(Ê_k, E_k*) ]
"""

from __future__ import annotations

import numpy as np

from effect_vla.constants import EFFECT_SMOOTH_L1_WEIGHT
from effect_vla.effect.correspondence import l2_normalize


def smooth_l1(pred: np.ndarray, target: np.ndarray, beta: float = 1.0) -> np.ndarray:
    diff = np.abs(pred - target)
    return np.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


def effect_loss_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    smooth_weight: float = EFFECT_SMOOTH_L1_WEIGHT,
) -> dict[str, np.ndarray]:
    """pred/target: (..., K, D). Returns scalar mean loss plus cosine."""
    pred = np.asarray(pred, dtype=np.float32)
    target = l2_normalize(np.asarray(target, dtype=np.float32))
    pred_n = l2_normalize(pred)
    cos = np.sum(pred_n * target, axis=-1)
    sl1 = smooth_l1(pred, target).mean(axis=-1)
    loss = (1.0 - cos) + smooth_weight * sl1
    return {
        "loss": np.mean(loss).astype(np.float32),
        "cosine": np.mean(cos).astype(np.float32),
        "smooth_l1": np.mean(sl1).astype(np.float32),
    }


def effect_loss_jax(pred, target, *, smooth_weight: float = EFFECT_SMOOTH_L1_WEIGHT):
    import jax.numpy as jnp

    def _norm(x, eps=1e-6):
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)

    pred_n = _norm(pred)
    target_n = _norm(jax_stop(target))
    cos = jnp.sum(pred_n * target_n, axis=-1)
    diff = jnp.abs(pred - target_n)
    sl1 = jnp.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5).mean(axis=-1)
    loss = (1.0 - cos) + smooth_weight * sl1
    return loss.mean(), {"effect_cosine": cos.mean(), "effect_smooth_l1": sl1.mean()}


def jax_stop(x):
    import jax

    return jax.lax.stop_gradient(x)
