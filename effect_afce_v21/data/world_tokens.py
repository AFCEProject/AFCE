"""Sparse world evidence: K_W intervals × (6 dynamic + 2 reference) regions."""

from __future__ import annotations

import numpy as np

from effect_afce_v21.constants import (
    DINO_DIM,
    MATCH_TEMPERATURE,
    N_DYNAMIC_REGIONS,
    N_REFERENCE_REGIONS,
    N_WORLD_REGIONS,
    NUM_WORLD_INTERVALS,
    PCA_DIM,
    ROBOT_SUPPRESSION,
    W_RAW_DIM,
    Y_W_DIM,
)
from effect_coupled_v4.constants import MATCH_RADIUS_IMAGE_FRACTION, MATCH_TOP_CANDIDATES
from effect_coupled_v4.data.env_descriptor import local_candidate_matrix, patch_centers
from effect_coupled_v4.data.match_torch import soft_local_match_batch_numpy_via_torch
from effect_coupled_v4.data.window_index import segment_boundaries


def world_interval_bounds(t: int, horizon: int, k_w: int = NUM_WORLD_INTERVALS) -> list[tuple[int, int]]:
    b = segment_boundaries(t, horizon=horizon, k=k_w)
    return [(b[i], b[i + 1]) for i in range(k_w)]


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def make_pca_proj(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(PCA_DIM, DINO_DIM)).astype(np.float32)
    w = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-6)
    return w, np.zeros((PCA_DIM,), dtype=np.float32)


def _project(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    out = flat @ weight.T + bias
    return out.reshape(*x.shape[:-1], weight.shape[0]).astype(np.float32)


def build_interval_regions(
    f_start: np.ndarray,
    f_end: np.ndarray,
    robot_mask: np.ndarray | None,
    *,
    centers: np.ndarray,
    cand_mat: np.ndarray,
    cand_valid: np.ndarray,
    pca_w: np.ndarray,
    pca_b: np.ndarray,
    t_start: float,
    t_end: float,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    f_start = np.asarray(f_start, dtype=np.float32)
    f_end = np.asarray(f_end, dtype=np.float32)
    n = f_start.shape[0]
    if robot_mask is None:
        rho = np.zeros((n,), np.float32)
    else:
        rho = np.clip(np.asarray(robot_mask, dtype=np.float32).reshape(-1), 0.0, 1.0)
    r_w = 1.0 - ROBOT_SUPPRESSION * rho

    _u, d, delta_f, conf = soft_local_match_batch_numpy_via_torch(
        f_start[None],
        f_end[None],
        centers,
        cand_mat,
        cand_valid,
        temperature=MATCH_TEMPERATURE,
        top_k=MATCH_TOP_CANDIDATES,
        device=device,
    )
    d, delta_f, conf = d[0], delta_f[0], conf[0]

    a = _l2_normalize(f_start)
    b = _l2_normalize(f_start + delta_f)
    c_p = (1.0 - (a * b).sum(axis=-1)).astype(np.float32)
    activity = (conf * r_w * c_p).astype(np.float32)

    dyn_idx = np.argsort(-activity)[:N_DYNAMIC_REGIONS]
    conf_ok = np.where(conf * r_w > 0.05)[0]
    if conf_ok.size == 0:
        conf_ok = np.arange(n)
    ref_pool = conf_ok[np.argsort(activity[conf_ok])]
    dyn_set = set(int(i) for i in dyn_idx.tolist())
    ref_clean: list[int] = []
    for i in ref_pool.tolist():
        if len(ref_clean) >= N_REFERENCE_REGIONS:
            break
        if int(i) not in dyn_set:
            ref_clean.append(int(i))
    while len(ref_clean) < N_REFERENCE_REGIONS:
        ref_clean.append(0)
    ref_idx = np.asarray(ref_clean[:N_REFERENCE_REGIONS], dtype=np.int64)
    sel = np.concatenate([dyn_idx, ref_idx], axis=0)
    is_dyn = np.array([1.0] * N_DYNAMIC_REGIONS + [0.0] * N_REFERENCE_REGIONS, np.float32)

    df64 = _project(delta_f[sel], pca_w, pca_b)
    disp = d[sel].astype(np.float32)
    xy = centers[sel].astype(np.float32)
    c = (conf[sel] * r_w[sel]).astype(np.float32)
    valid = (c > 1e-6).astype(np.float32)
    t0 = np.full((N_WORLD_REGIONS, 1), float(t_start), np.float32)
    t1 = np.full((N_WORLD_REGIONS, 1), float(t_end), np.float32)
    raw = np.concatenate([df64, disp, xy, c[:, None], is_dyn[:, None], t0, t1], axis=-1)
    assert raw.shape[-1] == W_RAW_DIM
    y_w = np.concatenate([df64, disp], axis=-1)
    assert y_w.shape[-1] == Y_W_DIM
    return {
        "W_raw": raw.astype(np.float32),
        "Y_W": y_w.astype(np.float32),
        "W_valid": valid,
        "W_conf": c,
        "patch_idx": sel.astype(np.int32),
    }
