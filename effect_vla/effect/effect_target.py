"""Deterministic Task-Effect target E*.

E = task-relevant controllable world change:
  correspondence → robot-motion suppression → top-r% dynamic aggregation
  → frozen linear projection → L2 normalize.

A*, ΔS_robot are never used. The projection P_E is a seeded frozen matrix,
not a trained encoder.
"""

from __future__ import annotations

import numpy as np

from effect_vla.constants import (
    AGGREGATION_BETA,
    DYNAMIC_MIN_PATCHES,
    DYNAMIC_TOP_RATIO,
    EFFECT_DIM,
    MATCH_TEMPERATURE,
    NUM_ANCHORS,
    PROJECTION_SEED,
)
from effect_vla.effect.correspondence import cosine_change, l2_normalize, match_future, residual
from effect_vla.effect.robot_mask import apply_robot_suppression


def frozen_projection(in_dim: int, out_dim: int = EFFECT_DIM, seed: int = PROJECTION_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=(out_dim, in_dim)).astype(np.float64)
    q, _ = np.linalg.qr(raw.T)
    # q is (in_dim, out_dim) with orthonormal columns when in_dim >= out_dim.
    if q.shape[1] < out_dim:
        # Rare: in_dim < out_dim. Pad extra random directions.
        extra = rng.normal(0.0, 1.0, size=(in_dim, out_dim - q.shape[1]))
        q = np.concatenate([q, extra], axis=1)
    return q.T.astype(np.float32)  # (out_dim, in_dim)


def select_dynamic_patches(
    change: np.ndarray,
    robot_mask: np.ndarray | None,
    *,
    top_ratio: float = DYNAMIC_TOP_RATIO,
    min_patches: int = DYNAMIC_MIN_PATCHES,
) -> np.ndarray:
    """Return a boolean mask over patches. change/robot_mask: (N,)."""
    n = change.shape[-1]
    valid = np.ones((n,), dtype=np.bool_)
    if robot_mask is not None:
        valid &= robot_mask < 0.5
    # Ignore near-zero change (static background / static objects).
    scores = np.where(valid, change, -1.0)
    k = max(min_patches, int(np.ceil(top_ratio * int(valid.sum() or n))))
    k = min(k, n)
    keep = np.zeros((n,), dtype=np.bool_)
    idx = np.argpartition(scores, -k)[-k:]
    keep[idx] = scores[idx] > 0
    if keep.sum() < min_patches:
        idx = np.argpartition(change, -min_patches)[-min_patches:]
        keep[idx] = True
    return keep


def aggregate_descriptor(
    delta: np.ndarray,
    change: np.ndarray,
    keep: np.ndarray,
    *,
    beta: float = AGGREGATION_BETA,
) -> np.ndarray:
    """Weighted mean of L2-normalized residuals over selected patches. (D,)."""
    delta_n = l2_normalize(delta)
    logits = beta * change
    logits = np.where(keep, logits, -1e9)
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / np.maximum(w.sum(), 1e-8)
    return (w[:, None] * delta_n).sum(axis=0)


def effect_from_pair(
    current: np.ndarray,
    future: np.ndarray,
    robot_mask: np.ndarray | None,
    projection: np.ndarray,
    *,
    temperature: float = MATCH_TEMPERATURE,
    generic_future: bool = False,
) -> dict[str, np.ndarray]:
    """Build one E* from a current/future patch-token pair.

    current/future: (N, D)
    robot_mask: (N,) or None
    projection: (d_E, D)

    generic_future=True skips correspondence, suppression, and dynamic
    selection (Ablation 2).
    """
    current = np.asarray(current, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    if generic_future:
        e = l2_normalize(future.mean(axis=0) - current.mean(axis=0))
        e_star = l2_normalize(projection @ e)
        n = current.shape[0]
        return {
            "effect": e_star.astype(np.float32),
            "change": np.zeros((n,), dtype=np.float32),
            "keep": np.ones((n,), dtype=np.bool_),
            "aligned": future,
        }

    aligned, _match = match_future(current, future, temperature=temperature)
    delta = residual(current, aligned)
    if robot_mask is not None:
        delta = apply_robot_suppression(delta, robot_mask)
    change = cosine_change(current, aligned)
    if robot_mask is not None:
        change = change * (1.0 - np.clip(robot_mask, 0.0, 1.0))
    keep = select_dynamic_patches(change, robot_mask)
    e = aggregate_descriptor(delta, change, keep)
    e_star = l2_normalize(projection @ e)
    return {
        "effect": e_star.astype(np.float32),
        "change": change.astype(np.float32),
        "keep": keep,
        "aligned": aligned.astype(np.float32),
    }


def effect_targets_episode(
    features: np.ndarray,
    robot_mask: np.ndarray | None,
    projection: np.ndarray,
    *,
    action_horizon: int,
    anchor_fractions: tuple[float, ...] = (0.5, 1.0),
    generic_future: bool = False,
) -> dict[str, np.ndarray]:
    """features: (T, N, D). Returns E* of shape (T, K, d_E) plus diagnostics."""
    t_len, n_patch, _dim = features.shape
    k = len(anchor_fractions)
    offsets = [max(1, int(round(frac * action_horizon))) for frac in anchor_fractions]
    effects = np.zeros((t_len, k, projection.shape[0]), dtype=np.float32)
    changes = np.zeros((t_len, k, n_patch), dtype=np.float32)
    keeps = np.zeros((t_len, k, n_patch), dtype=np.bool_)
    for t in range(t_len):
        mask_t = None if robot_mask is None else robot_mask[t]
        for ki, dt in enumerate(offsets):
            tk = min(t + dt, t_len - 1)
            packed = effect_from_pair(
                features[t],
                features[tk],
                mask_t,
                projection,
                generic_future=generic_future,
            )
            effects[t, ki] = packed["effect"]
            changes[t, ki] = packed["change"]
            keeps[t, ki] = packed["keep"]
    return {"effect": effects, "change": changes, "keep": keeps}
