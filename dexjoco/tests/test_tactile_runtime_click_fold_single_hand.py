from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from dexjoco.sim.envs.panda_click_mouse_env import PandaClickMouseGymEnv
from dexjoco.sim.envs.panda_fold_glasses_env import PandaFoldGlassesGymEnv


class ClickFoldSingleHandTactileRuntimeTest(unittest.TestCase):
    def _check_environment(
        self,
        env_type,
        *,
        renderer_target: str,
        sleep_target: str,
        env_kwargs: dict | None = None,
    ) -> None:
        fake_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            mock.patch(renderer_target) as renderer_type,
            mock.patch(sleep_target, return_value=None),
        ):
            renderer_type.return_value.render.return_value = fake_frame
            env = env_type(
                render_mode="rgb_array",
                seed=7,
                **({} if env_kwargs is None else env_kwargs),
            )
            try:
                observation, _ = env.reset(seed=23)
                self.assertIs(env._tactile_extractor, env._tactile_runtime.extractor)

                self.assertTrue(
                    env.observation_space["tactile_gt"].contains(
                        observation["tactile_gt"]
                    )
                )
                self.assertNotIn("tactile", observation)
                self.assertNotIn("tactile", env.observation_space.spaces)
                self.assertEqual(
                    observation["tactile_gt"]["fingertip_wrench_local"].shape,
                    (1, 4, 6),
                )
                np.testing.assert_allclose(
                    observation["tactile_gt"]["sim_time"],
                    [0.0],
                    atol=1e-12,
                )

                with mock.patch.object(
                    env._tactile_runtime,
                    "accumulate_substep",
                    wraps=env._tactile_runtime.accumulate_substep,
                ) as accumulate_substep:
                    observation, _, _, _, _ = env.step(
                        np.zeros(env.action_space.shape, dtype=np.float32)
                    )

                self.assertEqual(env._n_substeps, 10)
                self.assertEqual(accumulate_substep.call_count, env._n_substeps)
                self.assertTrue(
                    env.observation_space["tactile_gt"].contains(
                        observation["tactile_gt"]
                    )
                )
                self.assertNotIn("tactile", observation)
                self.assertNotIn("tactile", env.observation_space.spaces)
                np.testing.assert_allclose(
                    observation["tactile_gt"]["sim_time"],
                    [env.control_dt],
                    atol=1e-12,
                )
            finally:
                env.close()

    def test_click_mouse_tactile_runtime_lifecycle(self):
        self._check_environment(
            PandaClickMouseGymEnv,
            renderer_target="dexjoco.sim.envs.panda_click_mouse_env.MujocoRenderer",
            sleep_target="dexjoco.sim.envs.panda_click_mouse_env.time.sleep",
        )

    def test_fold_glasses_tactile_runtime_lifecycle(self):
        self._check_environment(
            PandaFoldGlassesGymEnv,
            renderer_target="dexjoco.sim.envs.panda_fold_glasses_env.MujocoRenderer",
            sleep_target="dexjoco.sim.envs.panda_fold_glasses_env.time.sleep",
            env_kwargs={"image_obs": False},
        )


if __name__ == "__main__":
    unittest.main()
