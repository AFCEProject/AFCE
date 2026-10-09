from __future__ import annotations

import unittest
from unittest import mock

import mujoco
import numpy as np

from dexjoco.sim.envs.panda_pick_bucket_env import PandaPickBucketGymEnv


class PickBucketTactileIntegrationTest(unittest.TestCase):
    def test_reset_and_step_publish_valid_tactile_observation(self):
        env = PandaPickBucketGymEnv(
            image_obs=False,
            render_mode="rgb_array",
            seed=7,
        )
        try:
            observation, _ = env.reset()
            tactile = observation["tactile_gt"]
            self.assertTrue(env.observation_space["tactile_gt"].contains(tactile))
            self.assertEqual(tactile["fingertip_wrench_local"].shape, (1, 4, 6))
            self.assertEqual(tactile["taxel_force_local"].shape, (1, 4, 16, 3))
            self.assertEqual(tactile["fingertip_wrench_local"].dtype, np.float32)
            self.assertEqual(tactile["taxel_force_local"].dtype, np.float32)
            np.testing.assert_array_equal(tactile["fingertip_wrench_local"], 0.0)
            np.testing.assert_array_equal(tactile["taxel_force_local"], 0.0)
            np.testing.assert_array_equal(tactile["sim_time"], [0.0])
            self.assertNotIn("tactile", observation)
            self.assertNotIn("tactile", env.observation_space.spaces)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertNotIn("images", observation)

            action = np.zeros(23, dtype=np.float64)
            observation, _, _, _, _ = env.step(action)
            tactile = observation["tactile_gt"]
            self.assertTrue(env.observation_space["tactile_gt"].contains(tactile))
            self.assertTrue(
                all(np.all(np.isfinite(value)) for value in tactile.values())
            )
            self.assertTrue(np.all(tactile["fingertip_normal"] >= 0.0))
            self.assertTrue(np.all(tactile["fingertip_force_peak"] >= 0.0))
            self.assertTrue(np.all(tactile["contact_fraction"] >= 0.0))
            self.assertTrue(np.all(tactile["contact_fraction"] <= 1.0))
            np.testing.assert_allclose(tactile["sim_time"], [0.02], atol=1e-12)
            self.assertNotIn("tactile", observation)
            self.assertTrue(env.observation_space.contains(observation))

            self.assertIsNone(env._viewer)
            fake_frame = np.zeros((8, 8, 3), dtype=np.uint8)
            with mock.patch(
                "dexjoco.sim.envs.panda_pick_bucket_env.MujocoRenderer"
            ) as renderer_type:
                renderer_type.return_value.render.return_value = fake_frame
                frames = env.render()
                renderer_type.assert_called_once_with(env.model, env.data)
                self.assertEqual(len(frames), len(env.camera_id))
                for frame in frames:
                    self.assertIs(frame, fake_frame)
        finally:
            env.close()

    def test_real_task_fingertip_contact_produces_nonzero_signal(self):
        env = PandaPickBucketGymEnv(
            image_obs=False,
            render_mode="rgb_array",
            seed=7,
        )
        try:
            env.reset()
            model, data = env.model, env.data

            fingertip_body_id = model.body("ff_tip").id
            fingertip_geom_ids = np.flatnonzero(
                (model.geom_bodyid == fingertip_body_id)
                & (
                    (model.geom_contype != 0)
                    | (model.geom_conaffinity != 0)
                )
            )
            self.assertGreater(fingertip_geom_ids.size, 0)
            fingertip_body_to_world = data.xmat[fingertip_body_id].reshape(3, 3)
            fingertip_origin = data.xpos[fingertip_body_id].copy()

            object_joint_id = model.joint("boxed_food_0_freejoint").id
            object_qpos_address = int(model.jnt_qposadr[object_joint_id])
            # Align the box face with the fingertip collision surface.
            data.qpos[object_qpos_address : object_qpos_address + 3] = (
                fingertip_origin
                + fingertip_body_to_world @ np.asarray([0.030, 0.0, 0.019])
            )
            data.qpos[object_qpos_address + 3 : object_qpos_address + 7] = (
                data.xquat[fingertip_body_id]
            )
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            action = np.zeros(23, dtype=np.float64)
            observation, _, _, _, _ = env.step(action)
            tactile = observation["tactile_gt"]

            self.assertGreater(float(tactile["fingertip_normal"].max()), 0.0)
            self.assertGreater(
                float(tactile["fingertip_force_peak"].max()),
                0.0,
            )
            self.assertGreater(
                float(
                    np.linalg.norm(
                        tactile["fingertip_impulse_local"], axis=-1
                    ).max()
                ),
                0.0,
            )
            self.assertGreater(float(tactile["contact_fraction"].max()), 0.0)
            self.assertGreater(
                float(np.linalg.norm(tactile["taxel_force_local"])),
                0.0,
            )
            reconstructed_force = np.einsum(
                "ftij,ftj->fi",
                env._tactile_runtime.taxel_layout.axes_body,
                tactile["taxel_force_local"][0],
            )
            np.testing.assert_allclose(
                reconstructed_force,
                tactile["fingertip_wrench_local"][0, :, :3],
                rtol=2e-5,
                atol=2e-5,
            )
            self.assertNotIn("tactile", observation)
            self.assertTrue(env.observation_space.contains(observation))
        finally:
            env.close()

    def test_controlled_sustained_object_contact_remains_observable(self):
        """A 20-step fixture distinguishes sustained load from a one-frame hit."""

        env = PandaPickBucketGymEnv(
            image_obs=False,
            render_mode="rgb_array",
            seed=7,
        )
        try:
            env.reset()
            model, data = env.model, env.data
            fingertip_body_id = model.body("ff_tip").id
            object_joint_id = model.joint("boxed_food_0_freejoint").id
            object_qpos_address = int(model.jnt_qposadr[object_joint_id])
            contact_fractions = []
            force_peaks = []

            for _ in range(20):
                fingertip_to_world = data.xmat[fingertip_body_id].reshape(3, 3)
                data.qpos[object_qpos_address : object_qpos_address + 3] = (
                    data.xpos[fingertip_body_id]
                    + fingertip_to_world @ np.asarray([0.030, 0.0, 0.019])
                )
                data.qpos[
                    object_qpos_address + 3 : object_qpos_address + 7
                ] = data.xquat[fingertip_body_id]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

                observation, _, _, _, _ = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                contact_fractions.append(
                    float(
                        observation["tactile_gt"]["contact_fraction"].max()
                    )
                )
                force_peaks.append(
                    float(
                        observation["tactile_gt"]["fingertip_force_peak"].max()
                    )
                )

            self.assertTrue(all(value > 0.8 for value in contact_fractions))
            self.assertTrue(all(value > 0.01 for value in force_peaks))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
