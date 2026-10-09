"""Unit tests for Effect target construction (no GPU, no OpenPI)."""

from __future__ import annotations

import unittest

import numpy as np

from effect_vla.data.geometry import pose7_xyzw_to_rt, relative_pose_in_frame, rotmat_to_rot6d
from effect_vla.data.grounding_targets import grounding_vector_at
from effect_vla.effect.correspondence import l2_normalize, match_future
from effect_vla.effect.effect_target import effect_from_pair, frozen_projection
from effect_vla.effect.robot_mask import apply_robot_suppression, downsample_mask_to_patches
from effect_vla.loss.effect_loss import effect_loss_numpy
from effect_vla.loss.grounding_loss import grounding_loss_numpy


class CorrespondenceTests(unittest.TestCase):
    def test_identity_matching(self):
        rng = np.random.default_rng(0)
        f = l2_normalize(rng.normal(size=(8, 16)).astype(np.float32))
        aligned, match = match_future(f, f, temperature=0.07)
        self.assertEqual(aligned.shape, f.shape)
        self.assertGreater(np.diag(match).mean(), 0.2)

    def test_permutation_recovers_tokens(self):
        rng = np.random.default_rng(1)
        current = l2_normalize(rng.normal(size=(6, 8)).astype(np.float32))
        perm = np.array([2, 0, 1, 5, 3, 4])
        future = current[perm]
        aligned, _ = match_future(current, future, temperature=0.02)
        # Soft matching should land close to the original tokens.
        cos = np.sum(l2_normalize(aligned) * current, axis=-1).mean()
        self.assertGreater(float(cos), 0.85)


class EffectTargetTests(unittest.TestCase):
    def test_unit_norm_and_no_action_leak(self):
        rng = np.random.default_rng(2)
        current = rng.normal(size=(16, 32)).astype(np.float32)
        future = current + rng.normal(size=current.shape).astype(np.float32) * 0.1
        proj = frozen_projection(32, 16, seed=0)
        packed = effect_from_pair(current, future, None, proj)
        e = packed["effect"]
        self.assertEqual(e.shape, (16,))
        np.testing.assert_allclose(np.linalg.norm(e), 1.0, atol=1e-5)
        # Deterministic: same inputs → same E*.
        packed2 = effect_from_pair(current, future, None, proj)
        np.testing.assert_allclose(packed["effect"], packed2["effect"], atol=1e-5)

    def test_robot_mask_zeros_those_patches(self):
        rng = np.random.default_rng(3)
        delta = rng.normal(size=(10, 4)).astype(np.float32)
        mask = np.zeros((10,), dtype=np.float32)
        mask[:3] = 1.0
        out = apply_robot_suppression(delta, mask)
        np.testing.assert_allclose(out[:3], 0.0)
        np.testing.assert_allclose(out[3:], delta[3:])

    def test_downsample_mask_shape(self):
        mask = np.zeros((224, 224), dtype=np.float32)
        mask[0:32, 0:32] = 1.0
        patches = downsample_mask_to_patches(mask, 14)
        self.assertEqual(patches.shape, (196,))
        self.assertGreater(patches[0], 0.0)


class GroundingTests(unittest.TestCase):
    def test_identity_delta_is_near_zero(self):
        pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        joints = np.zeros((16,), dtype=np.float32)
        state = np.concatenate([pose, joints], axis=0)
        g = grounding_vector_at(state, state)
        self.assertEqual(g.shape[-1], 42)
        np.testing.assert_allclose(g[:3], 0.0, atol=1e-5)
        np.testing.assert_allclose(g[9:21], 0.0, atol=1e-4)

    def test_pure_translation_recovers_delta(self):
        pose_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        pose_k = np.array([0.05, -0.02, 0.01, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        joints = np.zeros((16,), dtype=np.float32)
        g = grounding_vector_at(np.concatenate([pose_t, joints]), np.concatenate([pose_k, joints]))
        np.testing.assert_allclose(g[:3], [0.05, -0.02, 0.01], atol=1e-5)


class LossTests(unittest.TestCase):
    def test_effect_loss_zero_on_match(self):
        e = np.zeros((2, 8), dtype=np.float32)
        e[:, 0] = 1.0
        out = effect_loss_numpy(e, e)
        self.assertLess(float(out["loss"]), 1e-5)
        self.assertGreater(float(out["cosine"]), 0.999)

    def test_grounding_loss_finite(self):
        rng = np.random.default_rng(4)
        pred = rng.normal(size=(2, 42)).astype(np.float32)
        # Valid 6D rotations: identity-like.
        target = np.zeros_like(pred)
        target[..., 3] = 1.0
        target[..., 7] = 1.0
        target[..., 3 + 21] = 1.0
        target[..., 7 + 21] = 1.0
        out = grounding_loss_numpy(pred, target)
        self.assertTrue(np.isfinite(out["loss"]))


class AdapterZeroInitContract(unittest.TestCase):
    def test_zero_matrix_is_noop(self):
        tokens = np.ones((2, 4, 8), dtype=np.float32)
        extra = np.ones((2, 3, 8), dtype=np.float32)
        zero_proj = np.zeros((8, 8), dtype=np.float32)
        injected = extra @ zero_proj.T
        combined = np.concatenate([injected, tokens], axis=1)
        np.testing.assert_allclose(combined[:, 3:], tokens)
        np.testing.assert_allclose(combined[:, :3], 0.0)


if __name__ == "__main__":
    unittest.main()
