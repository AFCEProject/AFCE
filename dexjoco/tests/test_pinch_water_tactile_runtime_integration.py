from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from dexjoco.sim.envs import panda_pinch_tongs_env, panda_water_plant_env


class _DummyRenderer:
    def __init__(self, model, data):
        del model, data

    def render(self, *args, **kwargs):
        del args, kwargs
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        pass


class PinchWaterTactileRuntimeIntegrationTest(unittest.TestCase):
    def _assert_tactile_lifecycle(self, env):
        observation, _ = env.reset()

        tactile_gt = observation["tactile_gt"]
        self.assertTrue(env.observation_space["tactile_gt"].contains(tactile_gt))
        self.assertEqual(tactile_gt["fingertip_wrench_local"].shape, (1, 4, 6))
        np.testing.assert_allclose(tactile_gt["sim_time"], [0.0], atol=1e-12)
        self.assertNotIn("tactile", observation)
        self.assertNotIn("tactile", env.observation_space)

        with mock.patch.object(
            env._tactile_runtime,
            "accumulate_substep",
            wraps=env._tactile_runtime.accumulate_substep,
        ) as accumulate_substep:
            observation, _, _, _, _ = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )

        tactile_gt = observation["tactile_gt"]
        self.assertEqual(accumulate_substep.call_count, env._n_substeps)
        self.assertTrue(env.observation_space["tactile_gt"].contains(tactile_gt))
        self.assertEqual(tactile_gt["fingertip_wrench_local"].shape, (1, 4, 6))
        np.testing.assert_allclose(
            tactile_gt["sim_time"],
            [env.control_dt],
            atol=1e-12,
        )
        self.assertNotIn("tactile", observation)
        self.assertNotIn("tactile", env.observation_space)

    def test_pinch_tongs_tactile_lifecycle(self):
        with mock.patch.object(
            panda_pinch_tongs_env,
            "MujocoRenderer",
            _DummyRenderer,
        ):
            env = panda_pinch_tongs_env.PandaPinchTongsGymEnv(
                render_mode="rgb_array",
                randomize=False,
                seed=7,
            )
            try:
                self._assert_tactile_lifecycle(env)
            finally:
                env.close()

    def test_water_plant_tactile_lifecycle(self):
        env = panda_water_plant_env.PandaWaterPlantGymEnv(
            render_mode="none",
            randomize=False,
            seed=7,
        )
        try:
            self._assert_tactile_lifecycle(env)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
