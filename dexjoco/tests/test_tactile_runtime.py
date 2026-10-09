from __future__ import annotations

import unittest
from unittest import mock

import mujoco
import numpy as np

from dexjoco.sim.tactile import (
    ContactEvent,
    SINGLE_FINGERTIP_BODY_NAMES,
    TactileRuntime,
)


_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="hand" pos="0 0 0">
      <freejoint/>
      <body name="ff_tip" pos="-0.06 0 0">
        <geom type="capsule" size="0.012 0.010"/>
      </body>
      <body name="mf_tip" pos="-0.02 0 0">
        <geom type="capsule" size="0.012 0.010"/>
      </body>
      <body name="rf_tip" pos="0.02 0 0">
        <geom type="capsule" size="0.012 0.010"/>
      </body>
      <body name="th_tip" pos="0.06 0 0">
        <geom type="capsule" size="0.012 0.008"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_RIGHT_FINGERTIPS = ("r_ff_tip", "r_mf_tip", "r_rf_tip", "r_th_tip")
_LEFT_FINGERTIPS = ("l_ff_tip", "l_mf_tip", "l_rf_tip", "l_th_tip")
_BIMANUAL_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="right_hand" pos="-0.2 0 0">
      <body name="r_ff_tip" pos="-0.06 0 0">
        <geom type="capsule" size="0.012 0.010"/>
      </body>
      <body name="r_mf_tip" pos="-0.02 0 0">
        <geom type="capsule" size="0.012 0.010"/>
      </body>
      <body name="r_rf_tip" pos="0.02 0 0">
        <geom type="capsule" size="0.012 0.010"/>
      </body>
      <body name="r_th_tip" pos="0.06 0 0">
        <geom type="capsule" size="0.012 0.008"/>
      </body>
    </body>
    <body name="left_hand" pos="0.2 0 0">
      <body name="l_ff_tip" pos="-0.06 0 0">
        <geom type="capsule" size="0.012 0.010"/>
      </body>
      <body name="l_mf_tip" pos="-0.02 0 0">
        <geom type="capsule" size="0.012 0.010"/>
      </body>
      <body name="l_rf_tip" pos="0.02 0 0">
        <geom type="capsule" size="0.012 0.010"/>
      </body>
      <body name="l_th_tip" pos="0.06 0 0">
        <geom type="capsule" size="0.012 0.008"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class TactileRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)

    def _make_runtime(self) -> TactileRuntime:
        return TactileRuntime(
            self.model,
            self.data,
            (SINGLE_FINGERTIP_BODY_NAMES,),
            physics_dt=0.002,
        )

    @staticmethod
    def _event(
        runtime: TactileRuntime,
        fingertip_index: int,
        taxel_index: int,
        force_local: tuple[float, float, float],
    ) -> ContactEvent:
        return ContactEvent(
            fingertip_index=fingertip_index,
            position_local=runtime.taxel_layout.positions_local[
                fingertip_index, taxel_index
            ].copy(),
            force_local=np.asarray(force_local, dtype=np.float64),
            torque_local_at_contact=np.zeros(3, dtype=np.float64),
            normal_force=max(float(force_local[2]), 0.0),
        )

    @staticmethod
    def _run_interval(
        runtime: TactileRuntime,
        contact_batches: list[tuple[ContactEvent, ...]],
    ) -> np.ndarray:
        runtime.begin()
        with mock.patch.object(
            runtime.extractor,
            "current_contacts",
            side_effect=contact_batches,
        ) as current_contacts:
            for _ in contact_batches:
                mujoco.mj_step(runtime.model, runtime.data)
                runtime.accumulate_substep()
        runtime.finish()
        current_contacts.assert_has_calls(
            [mock.call(copy=False)] * len(contact_batches)
        )
        return runtime.observation()["tactile_gt"]["taxel_force_local"]

    def test_lifecycle_matches_public_single_hand_schema(self):
        runtime = self._make_runtime()
        runtime.reset()
        reset_observation = runtime.observation()
        self.assertTrue(runtime.observation_space.contains(reset_observation))
        self.assertEqual(
            reset_observation["tactile_gt"]["fingertip_wrench_local"].shape,
            (1, 4, 6),
        )
        self.assertEqual(
            reset_observation["tactile_gt"]["taxel_force_local"].shape,
            (1, 4, 16, 3),
        )
        self.assertEqual(
            set(reset_observation["tactile_gt"]),
            {
                "fingertip_wrench_local",
                "fingertip_normal",
                "fingertip_force_peak",
                "fingertip_impulse_local",
                "contact_fraction",
                "taxel_force_local",
                "sim_time",
            },
        )
        np.testing.assert_array_equal(
            reset_observation["tactile_gt"]["taxel_force_local"],
            0.0,
        )
        self.assertNotIn("tactile", reset_observation)

        runtime.begin()
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            runtime.accumulate_substep()
        runtime.finish()

        observation = runtime.observation()
        self.assertTrue(runtime.observation_space.contains(observation))
        np.testing.assert_allclose(
            observation["tactile_gt"]["sim_time"],
            [0.02],
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            observation["tactile_gt"]["taxel_force_local"],
            0.0,
        )
        self.assertNotIn("tactile", observation)

    def test_taxel_projection_conserves_force_in_fingertip_frames(self):
        runtime = self._make_runtime()
        events = (
            self._event(runtime, 0, 0, (1.25, -0.75, 4.0)),
            self._event(runtime, 0, 15, (-0.25, 0.5, 2.0)),
            self._event(runtime, 3, 7, (0.5, 1.0, -1.5)),
        )

        taxel_force = self._run_interval(runtime, [events]).reshape(4, 16, 3)
        # axes_local stores taxel-frame axes as rows in the fingertip body
        # frame.  Multiplication by its transpose maps each projected force
        # back to the fingertip frame before summing the 16 taxels.
        force_in_fingertip_frame = np.einsum(
            "ftjk,ftj->ftk",
            runtime.taxel_layout.axes_local,
            taxel_force,
        ).sum(axis=1)
        expected = np.zeros((4, 3), dtype=np.float64)
        for event in events:
            expected[event.fingertip_index] += event.force_local
        np.testing.assert_allclose(
            force_in_fingertip_frame,
            expected,
            rtol=2e-6,
            atol=2e-6,
        )

    def test_taxel_projection_is_mean_over_every_physics_substep(self):
        runtime = self._make_runtime()
        first = (self._event(runtime, 1, 2, (1.0, 2.0, 3.0)),)
        last = (self._event(runtime, 1, 12, (-0.5, 1.5, 6.0)),)
        expected = (
            runtime.taxel_projector.project(first)
            + runtime.taxel_projector.project(last)
        ) / 10.0

        # DexJoCo uses ten 2 ms physics samples per 20 ms control interval;
        # the eight no-contact samples must participate in the mean as zeros.
        actual = self._run_interval(
            runtime,
            [first, *([()] * 8), last],
        ).reshape(4, 16, 3)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_reset_clears_a_published_taxel_reading(self):
        runtime = self._make_runtime()
        event = self._event(runtime, 2, 8, (1.0, -2.0, 5.0))
        projected = self._run_interval(runtime, [(event,)])
        self.assertGreater(float(np.linalg.norm(projected)), 0.0)

        runtime.reset()

        observation = runtime.observation()
        np.testing.assert_array_equal(
            observation["tactile_gt"]["taxel_force_local"],
            0.0,
        )
        self.assertTrue(runtime.observation_space.contains(observation))

    def test_taxel_projection_is_deterministic_and_seed_independent(self):
        original_random_state = np.random.get_state()
        try:
            results = []
            following_random_values = []
            for seed in (7, 9381):
                np.random.seed(seed)
                model = mujoco.MjModel.from_xml_string(_XML)
                data = mujoco.MjData(model)
                runtime = TactileRuntime(
                    model,
                    data,
                    (SINGLE_FINGERTIP_BODY_NAMES,),
                    physics_dt=0.002,
                )
                event = self._event(runtime, 2, 5, (0.75, -1.25, 4.5))
                results.append(self._run_interval(runtime, [(event,)]))
                following_random_values.append(np.random.random(4))

                # Constructing and running the projection must not consume
                # process-global NumPy randomness.
                np.random.seed(seed)
                np.testing.assert_array_equal(
                    following_random_values[-1],
                    np.random.random(4),
                )
        finally:
            np.random.set_state(original_random_state)

        np.testing.assert_array_equal(results[0], results[1])

    def test_bimanual_taxel_shape_and_hand_order(self):
        model = mujoco.MjModel.from_xml_string(_BIMANUAL_XML)
        data = mujoco.MjData(model)
        runtime = TactileRuntime(
            model,
            data,
            (_RIGHT_FINGERTIPS, _LEFT_FINGERTIPS),
            physics_dt=0.002,
        )
        event = self._event(runtime, 7, 11, (1.0, 0.5, 3.0))

        taxel_force = self._run_interval(runtime, [(event,)])

        self.assertEqual(taxel_force.shape, (2, 4, 16, 3))
        np.testing.assert_array_equal(taxel_force[0], 0.0)
        self.assertGreater(float(np.linalg.norm(taxel_force[1, 3])), 0.0)
        self.assertTrue(runtime.observation_space.contains(runtime.observation()))

    def test_rejects_invalid_shape_and_physics_step(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            TactileRuntime(
                self.model,
                self.data,
                (
                    SINGLE_FINGERTIP_BODY_NAMES[:3],
                    SINGLE_FINGERTIP_BODY_NAMES[:2],
                ),
                physics_dt=0.002,
            )
        with self.assertRaisesRegex(ValueError, "physics_dt"):
            TactileRuntime(
                self.model,
                self.data,
                (SINGLE_FINGERTIP_BODY_NAMES,),
                physics_dt=0.0,
            )


if __name__ == "__main__":
    unittest.main()
