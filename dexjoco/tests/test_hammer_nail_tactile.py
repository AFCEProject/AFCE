from __future__ import annotations

import unittest
from unittest import mock

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexjoco.sim.envs.panda_hammer_nail_env import PandaHammerNailGymEnv
from dexjoco.sim.tactile import SINGLE_FINGERTIP_BODY_NAMES
from dexjoco.tasks.hammer_nail.config import TaskConfig


class HammerNailTactileIntegrationTest(unittest.TestCase):
    def test_reset_and_step_publish_valid_500hz_tactile_observation(self):
        env = PandaHammerNailGymEnv(
            image_obs=False,
            render_mode="rgb_array",
            seed=7,
        )
        try:
            observation, _ = env.reset()
            tactile = observation["tactile_gt"]

            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(
                tactile["fingertip_wrench_local"].shape,
                (1, len(SINGLE_FINGERTIP_BODY_NAMES), 6),
            )
            self.assertEqual(tactile["fingertip_wrench_local"].dtype, np.float32)
            for key, value in tactile.items():
                np.testing.assert_array_equal(
                    value,
                    np.zeros_like(value),
                    err_msg=f"reset did not zero tactile_gt.{key}",
            )
            self.assertNotIn("images", observation)
            self.assertIsNone(env._viewer)
            self.assertNotIn("tactile", observation)
            self.assertNotIn("tactile", env.observation_space.spaces)

            with mock.patch.object(
                env._tactile_extractor,
                "accumulate_substep",
                wraps=env._tactile_extractor.accumulate_substep,
            ) as accumulate_substep:
                observation, _, _, _, _ = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )

            tactile = observation["tactile_gt"]
            self.assertEqual(accumulate_substep.call_count, env._n_substeps)
            self.assertEqual(env._n_substeps, 10)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertTrue(
                all(np.all(np.isfinite(value)) for value in tactile.values())
            )
            self.assertTrue(np.all(tactile["fingertip_normal"] >= 0.0))
            self.assertTrue(np.all(tactile["fingertip_force_peak"] >= 0.0))
            self.assertTrue(np.all(tactile["contact_fraction"] >= 0.0))
            self.assertTrue(np.all(tactile["contact_fraction"] <= 1.0))
            np.testing.assert_allclose(tactile["sim_time"], [0.02], atol=1e-12)
            self.assertNotIn("tactile", observation)
        finally:
            env.close()

    def test_task_adapter_preserves_tactile_observation_and_space(self):
        env = TaskConfig().get_environment(
            policy_mode=True,
            image_obs=False,
            render_mode="rgb_array",
            seed=11,
        )
        try:
            observation, _ = env.reset()
            self.assertIn("tactile_gt", observation)
            self.assertNotIn("tactile", observation)
            self.assertNotIn("images", observation)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertTrue(
                env.observation_space["tactile_gt"].contains(
                    observation["tactile_gt"]
                )
            )
        finally:
            env.close()

    def test_real_nail_fingertip_contact_produces_nonzero_signal(self):
        env = PandaHammerNailGymEnv(
            image_obs=False,
            render_mode="rgb_array",
            seed=7,
        )
        try:
            env.reset()
            model, data = env.model, env.data

            fingertip_body_id = model.body(SINGLE_FINGERTIP_BODY_NAMES[0]).id
            fingertip_geom_ids = np.flatnonzero(
                (model.geom_bodyid == fingertip_body_id)
                & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            )
            self.assertGreater(fingertip_geom_ids.size, 0)
            fingertip_geom_id = int(fingertip_geom_ids[0])
            fingertip_position = data.geom_xpos[fingertip_geom_id].copy()

            nail_head_id = model.geom("nail_head").id
            nail_head_offset = model.geom_pos[nail_head_id].copy()
            data.mocap_quat[env._nail_mocap_id] = (1.0, 0.0, 0.0, 0.0)
            data.mocap_pos[env._nail_mocap_id] = (
                fingertip_position - nail_head_offset
            )
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            fingertip_geom_set = set(int(gid) for gid in fingertip_geom_ids)
            nail_geom_set = set(int(gid) for gid in env._nail_geom_ids)
            has_task_contact = any(
                (
                    int(data.contact[index].geom1) in fingertip_geom_set
                    and int(data.contact[index].geom2) in nail_geom_set
                )
                or (
                    int(data.contact[index].geom2) in fingertip_geom_set
                    and int(data.contact[index].geom1) in nail_geom_set
                )
                for index in range(int(data.ncon))
            )
            self.assertTrue(has_task_contact)

            observation, _, _, _, _ = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            tactile = observation["tactile_gt"]

            self.assertGreater(float(tactile["fingertip_normal"].max()), 0.0)
            self.assertGreater(float(tactile["fingertip_force_peak"].max()), 0.0)
            self.assertGreater(
                float(
                    np.linalg.norm(
                        tactile["fingertip_impulse_local"], axis=-1
                    ).max()
                ),
                0.0,
            )
            self.assertGreater(float(tactile["contact_fraction"].max()), 0.0)
            self.assertNotIn("tactile", observation)
            self.assertTrue(env.observation_space.contains(observation))
        finally:
            env.close()

    def test_hammer_nail_contact_transmits_load_through_handle_to_fingertip(self):
        def run_condition(with_nail_contact: bool):
            env = PandaHammerNailGymEnv(
                image_obs=False,
                render_mode="rgb_array",
                seed=7,
            )
            try:
                env.reset()
                model, data = env.model, env.data
                fingertip_body_id = model.body("ff_tip").id
                fingertip_to_world = data.xmat[fingertip_body_id].reshape(3, 3)
                fingertip_origin = data.xpos[fingertip_body_id]

                hammer_joint_id = model.joint("hammer_joint").id
                hammer_qpos_address = int(model.jnt_qposadr[hammer_joint_id])
                # Align the hammer handle with the finite palmar patch and add
                # a small penetration so the grasp-side contact is active.
                data.qpos[hammer_qpos_address : hammer_qpos_address + 3] = (
                    fingertip_origin
                    + fingertip_to_world @ np.asarray([0.0354, 0.0, 0.019])
                )
                data.qpos[hammer_qpos_address + 3 : hammer_qpos_address + 7] = (
                    data.xquat[fingertip_body_id]
                )
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

                if with_nail_contact:
                    face_id = model.geom("face").id
                    face_to_world = data.geom_xmat[face_id].reshape(3, 3)
                    face_center = data.geom_xpos[face_id]
                    face_axis = face_to_world[:, 2]
                    # The face and nail cylinders meet end-to-end with 2 mm
                    # penetration.  This produces an actual hammer--nail
                    # constraint while the same rigid hammer touches the hand.
                    nail_head_center = face_center + face_axis * (
                        0.0122 + 0.0048 - 0.002
                    )
                    data.mocap_quat[env._nail_mocap_id] = Rotation.from_matrix(
                        face_to_world
                    ).as_quat(scalar_first=True)
                    data.mocap_pos[env._nail_mocap_id] = (
                        nail_head_center
                        - face_to_world @ np.asarray([0.0, 0.0, 0.0036])
                    )
                else:
                    data.mocap_pos[env._nail_mocap_id] = (2.0, 2.0, 2.0)
                mujoco.mj_forward(model, data)

                fingertip_geom_ids = set(
                    int(geom_id)
                    for geom_id in np.flatnonzero(
                        (model.geom_bodyid == fingertip_body_id)
                        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
                    )
                )
                hammer_geom_ids = set(int(value) for value in env._hammer_geom_ids)
                grasp_hammer_geom_ids = hammer_geom_ids | {
                    int(model.geom("handle").id)
                }
                nail_geom_ids = set(int(value) for value in env._nail_geom_ids)

                def has_contact(first_set, second_set):
                    return any(
                        (
                            int(data.contact[index].geom1) in first_set
                            and int(data.contact[index].geom2) in second_set
                        )
                        or (
                            int(data.contact[index].geom2) in first_set
                            and int(data.contact[index].geom1) in second_set
                        )
                        for index in range(int(data.ncon))
                    )

                self.assertTrue(
                    has_contact(fingertip_geom_ids, grasp_hammer_geom_ids)
                )
                if with_nail_contact:
                    self.assertTrue(has_contact(hammer_geom_ids, nail_geom_ids))

                observation, _, _, _, _ = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                return float(
                    observation["tactile_gt"]["fingertip_force_peak"].max()
                )
            finally:
                env.close()

        baseline_peak = run_condition(False)
        impact_peak = run_condition(True)
        self.assertGreater(impact_peak, max(3.0 * baseline_peak, 0.05))


if __name__ == "__main__":
    unittest.main()
