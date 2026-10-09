"""SE(3) helpers for wrist/fingertip grounding targets.

Internal quaternion convention is Hamilton (w, x, y, z).
DexJoCo TCP pose is stored as (x, y, z, qx, qy, qz, qw).
"""

from __future__ import annotations

import numpy as np

IDENTITY_ROT6D = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-8)
    return (q / n).astype(np.float32)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(np.asarray(q1, dtype=np.float64), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(q2, dtype=np.float64), -1, 0)
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return quat_normalize(out)


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return quat_normalize(np.stack([w, -x, -y, -z], axis=-1))


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    r00 = ww + xx - yy - zz
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = ww - xx - yy + zz
    return np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def rotmat_to_rot6d(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float32)
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def rot6d_to_rotmat(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    a1 = d[..., :3]
    a2 = d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    a2_proj = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_proj / np.maximum(np.linalg.norm(a2_proj, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def rotmat_to_quat(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    t = np.trace(r, axis1=-2, axis2=-1)
    q = np.zeros(r.shape[:-2] + (4,), dtype=np.float64)
    # Shepperd's method, (w, x, y, z)
    r00, r11, r22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    r01, r02, r10 = r[..., 0, 1], r[..., 0, 2], r[..., 1, 0]
    r12, r20, r21 = r[..., 1, 2], r[..., 2, 0], r[..., 2, 1]
    cond_w = t > 0
    s = np.sqrt(np.maximum(t + 1.0, 0.0)) * 2.0
    qw = 0.25 * s
    qx = (r21 - r12) / np.maximum(s, 1e-8)
    qy = (r02 - r20) / np.maximum(s, 1e-8)
    qz = (r10 - r01) / np.maximum(s, 1e-8)
    q_w = np.stack([qw, qx, qy, qz], axis=-1)

    cond_x = (~cond_w) & (r00 > r11) & (r00 > r22)
    s = np.sqrt(np.maximum(1.0 + r00 - r11 - r22, 0.0)) * 2.0
    q_x = np.stack(
        [
            (r21 - r12) / np.maximum(s, 1e-8),
            0.25 * s,
            (r01 + r10) / np.maximum(s, 1e-8),
            (r02 + r20) / np.maximum(s, 1e-8),
        ],
        axis=-1,
    )
    cond_y = (~cond_w) & (~cond_x) & (r11 > r22)
    s = np.sqrt(np.maximum(1.0 + r11 - r00 - r22, 0.0)) * 2.0
    q_y = np.stack(
        [
            (r02 - r20) / np.maximum(s, 1e-8),
            (r01 + r10) / np.maximum(s, 1e-8),
            0.25 * s,
            (r12 + r21) / np.maximum(s, 1e-8),
        ],
        axis=-1,
    )
    s = np.sqrt(np.maximum(1.0 + r22 - r00 - r11, 0.0)) * 2.0
    q_z = np.stack(
        [
            (r10 - r01) / np.maximum(s, 1e-8),
            (r02 + r20) / np.maximum(s, 1e-8),
            (r12 + r21) / np.maximum(s, 1e-8),
            0.25 * s,
        ],
        axis=-1,
    )
    q = np.where(cond_w[..., None], q_w, q)
    q = np.where(cond_x[..., None], q_x, q)
    q = np.where(cond_y[..., None], q_y, q)
    q = np.where((~cond_w & ~cond_x & ~cond_y)[..., None], q_z, q)
    return quat_normalize(q)


def pose7_xyzw_to_rt(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DexJoCo TCP: (x,y,z, qx,qy,qz,qw) -> R (3,3), t (3,)."""
    pose7 = np.asarray(pose7, dtype=np.float32)
    t = pose7[..., :3]
    q_wxyz = quat_xyzw_to_wxyz(pose7[..., 3:7])
    return quat_to_rotmat(q_wxyz), t


def invert_rt(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_inv = np.swapaxes(r, -1, -2)
    t_inv = -np.einsum("...ij,...j->...i", r_inv, t)
    return r_inv, t_inv


def transform_points(r: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...nj->...ni", r, pts) + t[..., None, :]


def relative_pose_in_frame(
    r_from: np.ndarray,
    t_from: np.ndarray,
    r_to: np.ndarray,
    t_to: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose of `to` expressed in the `from` frame."""
    r_inv, t_inv = invert_rt(r_from, t_from)
    r_rel = np.einsum("...ij,...jk->...ik", r_inv, r_to)
    t_rel = t_inv + np.einsum("...ij,...j->...i", r_inv, t_to)
    return r_rel, t_rel
