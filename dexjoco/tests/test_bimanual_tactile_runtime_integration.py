from __future__ import annotations

import time
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np

from dexjoco.sim.envs import (
    panda_bimanual_assembly_env,
    panda_bimanual_hanoi_env,
    panda_bimanual_microwave_cook_env,
    panda_bimanual_photograph_env,
    panda_bimanual_unlock_ipad_env,
)
from dexjoco.sim.tactile import (
    LEFT_FINGERTIP_BODY_NAMES,
    RIGHT_FINGERTIP_BODY_NAMES,
)


class _DummyRenderer:
    def __init__(self, model, data):
        del model, data

    def render(self, *args, **kwargs):
        del args, kwargs
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        pass


_ENVIRONMENT_CASES = (
    (
        "assembly",
        panda_bimanual_assembly_env,
        panda_bimanual_assembly_env.PandaBimanualAssemblyGymEnv,
    ),
    (
        "hanoi",
        panda_bimanual_hanoi_env,
        panda_bimanual_hanoi_env.PandaBimanualHanoiGymEnv,
    ),
    (
        "microwave_cook",
        panda_bimanual_microwave_cook_env,
        panda_bimanual_microwave_cook_env.PandaBimanualMicrowaveCookGymEnv,
    ),
    (
        "photograph",
        panda_bimanual_photograph_env,
        panda_bimanual_photograph_env.PandaBimanualPhotographGymEnv,
    ),
    (
        "unlock_ipad",
        panda_bimanual_unlock_ipad_env,
        panda_bimanual_unlock_ipad_env.PandaBimanualUnlockIpadGymEnv,
    ),
)


class BimanualTactileRuntimeIntegrationTest(unittest.TestCase):
    def _patched_environment(self, module, env_type):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(module, "MujocoRenderer", _DummyRenderer))
        stack.enter_context(mock.patch.object(module.time, "sleep", return_value=None))
        try:
            env = env_type(
                image_obs=False,
                render_mode="rgb_array",
                randomize=False,
                seed=7,
            )
        except Exception:
            stack.close()
            raise
        return stack, env

    def _assert_ground_truth_schema(self, env, observation, expected_time):
        self.assertIn("tactile_gt", observation)
        self.assertNotIn("tactile", observation)
        self.assertIn("tactile_gt", env.observation_space.spaces)
        self.assertNotIn("tactile", env.observation_space.spaces)

        tactile_gt = observation["tactile_gt"]
        self.assertTrue(env.observation_space["tactile_gt"].contains(tactile_gt))
        self.assertEqual(tactile_gt["fingertip_wrench_local"].shape, (2, 4, 6))
        self.assertEqual(tactile_gt["taxel_force_local"].shape, (2, 4, 16, 3))
        self.assertEqual(tactile_gt["fingertip_normal"].shape, (2, 4))
        self.assertEqual(tactile_gt["fingertip_force_peak"].shape, (2, 4))
        self.assertEqual(tactile_gt["fingertip_impulse_local"].shape, (2, 4, 3))
        self.assertEqual(tactile_gt["contact_fraction"].shape, (2, 4))
        self.assertEqual(tactile_gt["fingertip_wrench_local"].dtype, np.float32)
        self.assertEqual(tactile_gt["taxel_force_local"].dtype, np.float32)
        self.assertEqual(tactile_gt["sim_time"].dtype, np.float64)
        np.testing.assert_allclose(
            tactile_gt["sim_time"], [expected_time], atol=1e-12
        )

    def test_all_five_bimanual_environments_publish_ground_truth_only(self):
        expected_body_order = (
            *RIGHT_FINGERTIP_BODY_NAMES,
            *LEFT_FINGERTIP_BODY_NAMES,
        )
        for name, module, env_type in _ENVIRONMENT_CASES:
            with self.subTest(environment=name):
                stack, env = self._patched_environment(module, env_type)
                try:
                    observation, _ = env.reset()
                    self.assertIs(
                        env._tactile_extractor, env._tactile_runtime.extractor
                    )
                    self.assertEqual(
                        env._tactile_runtime.fingertip_body_names,
                        expected_body_order,
                    )
                    self._assert_ground_truth_schema(env, observation, 0.0)
                    for value in observation["tactile_gt"].values():
                        np.testing.assert_array_equal(value, np.zeros_like(value))

                    with mock.patch.object(
                        env._tactile_runtime,
                        "accumulate_substep",
                        wraps=env._tactile_runtime.accumulate_substep,
                    ) as accumulate_substep:
                        observation, _, _, _, _ = env.step(0)

                    self.assertEqual(env._n_substeps, 10)
                    self.assertEqual(
                        accumulate_substep.call_count, env._n_substeps
                    )
                    self._assert_ground_truth_schema(
                        env, observation, env.control_dt
                    )
                finally:
                    env.close()
                    stack.close()

    def test_hanoi_pending_illegal_return_preserves_last_ground_truth(self):
        stack, env = self._patched_environment(
            panda_bimanual_hanoi_env,
            panda_bimanual_hanoi_env.PandaBimanualHanoiGymEnv,
        )
        try:
            env.reset()
            previous, _, _, _, _ = env.step(0)
            previous_tactile = {
                key: value.copy() for key, value in previous["tactile_gt"].items()
            }

            env._pending_illegal_reset = True
            env._pending_illegal_deadline = time.time() + 60.0
            env._pending_illegal_info = {"illegal_state": True}
            with mock.patch.object(
                env._tactile_runtime,
                "accumulate_substep",
                wraps=env._tactile_runtime.accumulate_substep,
            ) as accumulate_substep:
                observation, _, terminated, truncated, info = env.step(0)

            self.assertEqual(accumulate_substep.call_count, 0)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertFalse(info["auto_reset"])
            for key, expected in previous_tactile.items():
                np.testing.assert_array_equal(
                    observation["tactile_gt"][key], expected, err_msg=key
                )
        finally:
            env.close()
            stack.close()


if __name__ == "__main__":
    unittest.main()
