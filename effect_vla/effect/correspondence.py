"""Cross-frame patch correspondence.

Same-location subtraction is invalid once objects move. Match current patch
tokens to future tokens, then take the aligned residual.
"""

from __future__ import annotations

import numpy as np

from effect_vla.constants import MATCH_TEMPERATURE


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def match_future(
    current: np.ndarray,
    future: np.ndarray,
    *,
    temperature: float = MATCH_TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-assign future patches onto the current grid.

    Args:
        current: (..., N, D)
        future:  (..., N, D)

    Returns:
        aligned_future: (..., N, D)
        match_matrix:   (..., N, N)
    """
    ft = l2_normalize(current)
    fk = l2_normalize(future)
    logits = np.einsum("...nd,...md->...nm", ft, fk) / float(temperature)
    logits = logits - logits.max(axis=-1, keepdims=True)
    match = np.exp(logits)
    match = match / np.maximum(match.sum(axis=-1, keepdims=True), 1e-8)
    aligned = np.einsum("...nm,...md->...nd", match, future)
    return aligned, match


def residual(current: np.ndarray, aligned_future: np.ndarray) -> np.ndarray:
    return aligned_future - current


def cosine_change(current: np.ndarray, aligned_future: np.ndarray) -> np.ndarray:
    """c_p = 1 - cos(F_t,p, aligned F_k,p). Shape (..., N)."""
    a = l2_normalize(current)
    b = l2_normalize(aligned_future)
    return 1.0 - np.sum(a * b, axis=-1)
