from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from dexjoco.sim.mujoco_gym_env import MujocoGymEnv


_MINIMAL_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="free_body" pos="0 0 0.2">
      <freejoint/>
      <geom type="sphere" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


class MujocoGymEnvTimeDiscretizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self._xml_path = Path(self._temporary_directory.name) / "minimal.xml"
        self._xml_path.write_text(_MINIMAL_XML, encoding="utf-8")

    def test_integer_ratio_uses_nearest_integer_without_float_floor_loss(self):
        env = MujocoGymEnv(
            xml_path=self._xml_path,
            control_dt=0.03,
            physics_dt=0.002,
        )
        self.addCleanup(env.close)

        self.assertEqual(env._n_substeps, 15)
        self.assertEqual(env.control_dt, 0.03)
        self.assertEqual(env.physics_dt, 0.002)

        for control_interval in range(1, 4):
            for _ in range(env._n_substeps):
                mujoco.mj_step(env.model, env.data)
            np.testing.assert_allclose(
                env.data.time,
                control_interval * env.control_dt,
                rtol=0.0,
                atol=1e-12,
            )

    def test_noninteger_control_to_physics_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            MujocoGymEnv(
                xml_path=self._xml_path,
                control_dt=0.02,
                physics_dt=0.003,
            )

    def test_time_steps_must_be_finite_and_positive(self):
        invalid_pairs = (
            (0.0, 0.002),
            (-0.02, 0.002),
            (np.nan, 0.002),
            (0.02, 0.0),
            (0.02, -0.002),
            (0.02, np.inf),
        )
        for control_dt, physics_dt in invalid_pairs:
            with self.subTest(control_dt=control_dt, physics_dt=physics_dt):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    MujocoGymEnv(
                        xml_path=self._xml_path,
                        control_dt=control_dt,
                        physics_dt=physics_dt,
                    )


if __name__ == "__main__":
    unittest.main()
