from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from dexjoco.tasks.obs_adapters import DexjocoObsAdapter


class _DummySensorEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(2,), dtype=np.float64
                        )
                    }
                ),
                "tactile_gt": gym.spaces.Dict(
                    {
                        "normal": gym.spaces.Box(
                            0.0, np.inf, shape=(1, 4), dtype=np.float32
                        )
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {
            "state": {"pose": np.asarray([1.0, 2.0], dtype=np.float64)},
            "tactile_gt": {
                "normal": np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
            },
        }, {}


class DexjocoObsAdapterTest(unittest.TestCase):
    def test_non_state_sensor_groups_are_preserved(self):
        env = DexjocoObsAdapter(_DummySensorEnv(), proprio_keys=["pose"])
        observation, _ = env.reset()

        np.testing.assert_array_equal(observation["state"], [1.0, 2.0])
        np.testing.assert_array_equal(
            observation["tactile_gt"]["normal"],
            [[0.0, 1.0, 2.0, 3.0]],
        )
        self.assertTrue(
            env.observation_space["tactile_gt"].contains(
                observation["tactile_gt"]
            )
        )


if __name__ == "__main__":
    unittest.main()
