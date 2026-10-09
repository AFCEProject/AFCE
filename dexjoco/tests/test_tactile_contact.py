from __future__ import annotations

import unittest

import mujoco
import numpy as np

from dexjoco.sim.tactile import FingertipContactExtractor


def _contact_model(*, offset: float = 0.0):
    tip = f"""
      <body name="tip">
        <freejoint/>
        <geom name="tip_geom" type="sphere" pos="{offset} 0 0" size="0.05"/>
      </body>
    """
    plane = '<geom name="plane" type="plane" size="1 1 0.1"/>'
    xml = f"""
    <mujoco>
      <option gravity="0 0 0" timestep="0.002"/>
      <worldbody>{plane}{tip}</worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def _set_free_body_pose(data, position, quaternion=(1.0, 0.0, 0.0, 0.0)):
    data.qpos[:3] = position
    data.qpos[3:7] = quaternion


class FingertipContactExtractorTest(unittest.TestCase):
    def test_no_contact_is_zero_and_observation_is_a_copy(self):
        model, data = _contact_model()
        _set_free_body_pose(data, (0.0, 0.0, 0.2))
        mujoco.mj_forward(model, data)

        extractor = FingertipContactExtractor(model, data, ("tip",))
        extractor.begin()
        extractor.accumulate_substep()
        observation = extractor.finish()

        np.testing.assert_array_equal(observation["fingertip_wrench_local"], 0.0)
        np.testing.assert_array_equal(observation["fingertip_normal"], 0.0)
        np.testing.assert_array_equal(observation["fingertip_force_peak"], 0.0)
        np.testing.assert_array_equal(observation["fingertip_impulse_local"], 0.0)
        np.testing.assert_array_equal(observation["contact_fraction"], 0.0)

        observation["fingertip_normal"][0] = 123.0
        self.assertEqual(extractor.observation()["fingertip_normal"][0], 0.0)

    def test_margin_only_contact_without_constraint_is_zero(self):
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option gravity="0 0 0" timestep="0.002"/>
              <worldbody>
                <geom type="plane" size="1 1 0.1"/>
                <body name="tip">
                  <freejoint/>
                  <geom type="sphere" size="0.05" margin="0.1" gap="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        _set_free_body_pose(data, (0.0, 0.0, 0.125))
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 1)
        self.assertLess(data.contact[0].efc_address, 0)

        extractor = FingertipContactExtractor(model, data, ("tip",))
        extractor.begin()
        extractor.accumulate_substep()
        observation = extractor.finish()

        np.testing.assert_array_equal(observation["fingertip_wrench_local"], 0.0)
        np.testing.assert_array_equal(observation["fingertip_normal"], 0.0)
        np.testing.assert_array_equal(observation["contact_fraction"], 0.0)

    def test_current_contact_event_reconstructs_sample_wrench_and_is_a_copy(self):
        model, data = _contact_model(offset=0.02)
        _set_free_body_pose(data, (0.0, 0.0, 0.045))
        mujoco.mj_forward(model, data)

        extractor = FingertipContactExtractor(model, data, ("tip",))
        extractor.begin()
        extractor.accumulate_substep()
        events = extractor.current_contacts()
        wrench = extractor.finish()["fingertip_wrench_local"][0]

        self.assertGreater(len(events), 0)
        event_force = np.sum([event.force_local for event in events], axis=0)
        event_moment = np.sum(
            [
                event.torque_local_at_contact
                + np.cross(event.position_local, event.force_local)
                for event in events
            ],
            axis=0,
        )
        np.testing.assert_allclose(event_force, wrench[:3], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(event_moment, wrench[3:], rtol=1e-5, atol=1e-6)

        events[0].position_local[:] = 123.0
        self.assertFalse(
            np.array_equal(
                events[0].position_local,
                extractor.current_contacts()[0].position_local,
            )
        )

    def test_geom1_and_geom2_sign_matches_constraint_force(self):
        plane_model, plane_data = _contact_model()
        fixtures = [(plane_model, plane_data, (0.0, 0.0, 0.045), 2)]

        wall_model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option gravity="0 0 0" timestep="0.002"/>
              <worldbody>
                <geom name="wall" type="box" size="0.1 0.1 0.1"/>
                <body name="tip">
                  <freejoint/>
                  <geom name="tip_geom" type="sphere" size="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        wall_data = mujoco.MjData(wall_model)
        fixtures.append((wall_model, wall_data, (0.12, 0.0, 0.0), 1))

        for model, data, position, expected_contact_side in fixtures:
            with self.subTest(expected_contact_side=expected_contact_side):
                _set_free_body_pose(data, position)
                mujoco.mj_forward(model, data)
                self.assertGreater(data.ncon, 0)

                tip_geom_id = model.geom("tip_geom").id
                contact = data.contact[0]
                actual_side = 1 if contact.geom1 == tip_geom_id else 2
                self.assertEqual(actual_side, expected_contact_side)

                extractor = FingertipContactExtractor(model, data, ("tip",))
                extractor.begin()
                extractor.accumulate_substep()
                observation = extractor.finish()

                body_id = model.body("tip").id
                body_to_world = data.xmat[body_id].reshape(3, 3)
                force_world = (
                    body_to_world @ observation["fingertip_wrench_local"][0, :3]
                )
                self.assertGreater(np.linalg.norm(force_world), 0.0)
                np.testing.assert_allclose(
                    force_world,
                    data.qfrc_constraint[:3],
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_rotated_local_frame(self):
        half_angle = np.pi / 4.0
        quaternion_y_90 = (
            np.cos(half_angle),
            0.0,
            np.sin(half_angle),
            0.0,
        )
        model, data = _contact_model()
        _set_free_body_pose(data, (0.0, 0.0, 0.045), quaternion_y_90)
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)

        extractor = FingertipContactExtractor(model, data, ("tip",))
        extractor.begin()
        extractor.accumulate_substep()
        observation = extractor.finish()

        body_id = model.body("tip").id
        body_to_world = data.xmat[body_id].reshape(3, 3)
        wrench_local = observation["fingertip_wrench_local"][0]
        force_world = body_to_world @ wrench_local[:3]
        np.testing.assert_allclose(force_world[:2], 0.0, atol=1e-6)
        self.assertGreater(force_world[2], 0.0)
        # A 90-degree body rotation means world +z is not local +z.
        self.assertGreater(abs(wrench_local[0]), 1e-3)
        self.assertLess(abs(wrench_local[2]), 1e-4)

    def test_offset_contact_has_expected_moment_arm(self):
        offset = 0.02
        model, data = _contact_model(offset=offset)
        _set_free_body_pose(data, (0.0, 0.0, 0.045))
        mujoco.mj_forward(model, data)

        extractor = FingertipContactExtractor(model, data, ("tip",))
        extractor.begin()
        extractor.accumulate_substep()
        wrench = extractor.finish()["fingertip_wrench_local"][0]

        self.assertGreater(wrench[2], 0.0)
        self.assertLess(wrench[4], 0.0)
        self.assertAlmostEqual(wrench[4], -offset * wrench[2], places=5)

    def test_contact_fraction_threshold_uses_total_fingertip_load(self):
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option gravity="0 0 0" timestep="0.002"/>
              <worldbody>
                <geom type="plane" size="1 1 0.1"/>
                <body name="tip">
                  <freejoint/>
                  <geom type="box" size="0.1 0.1 0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        _set_free_body_pose(data, (0.0, 0.0, 0.045))
        mujoco.mj_forward(model, data)

        point_normals = []
        for contact_index in range(data.ncon):
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, wrench)
            point_normals.append(float(wrench[0]))
        self.assertGreater(len(point_normals), 1)
        total_normal = sum(point_normals)
        threshold = (max(point_normals) + total_normal) / 2.0
        self.assertGreater(threshold, max(point_normals))
        self.assertLess(threshold, total_normal)

        extractor = FingertipContactExtractor(
            model,
            data,
            ("tip",),
            contact_threshold=threshold,
        )
        extractor.begin()
        extractor.accumulate_substep()
        observation = extractor.finish()

        self.assertAlmostEqual(
            float(observation["fingertip_normal"][0]),
            total_normal,
            places=5,
        )
        np.testing.assert_array_equal(observation["contact_fraction"], 1.0)

    def test_control_window_reduction_includes_zero_substeps(self):
        model, data = _contact_model(offset=0.02)
        extractor = FingertipContactExtractor(model, data, ("tip",))

        _set_free_body_pose(data, (0.0, 0.0, 0.045))
        mujoco.mj_forward(model, data)
        extractor.begin()
        extractor.accumulate_substep()
        one_contact = extractor.finish()

        extractor.begin()
        _set_free_body_pose(data, (0.0, 0.0, 0.045))
        mujoco.mj_forward(model, data)
        extractor.accumulate_substep()
        _set_free_body_pose(data, (0.0, 0.0, 0.2))
        mujoco.mj_forward(model, data)
        extractor.accumulate_substep()
        reduced = extractor.finish()

        np.testing.assert_allclose(
            reduced["fingertip_wrench_local"],
            one_contact["fingertip_wrench_local"] / 2.0,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            reduced["fingertip_force_peak"],
            one_contact["fingertip_force_peak"],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            reduced["fingertip_normal"],
            one_contact["fingertip_normal"] / 2.0,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(reduced["contact_fraction"], 0.5)
        np.testing.assert_allclose(
            reduced["fingertip_impulse_local"],
            one_contact["fingertip_wrench_local"][:, :3] * model.opt.timestep,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_mj_step_impulse_matches_translational_momentum_change(self):
        model, data = _contact_model()
        _set_free_body_pose(data, (0.0, 0.0, 0.045))
        mujoco.mj_forward(model, data)

        extractor = FingertipContactExtractor(model, data, ("tip",))
        velocity_before = data.qvel[:3].copy()
        extractor.begin()
        mujoco.mj_step(model, data)
        extractor.accumulate_substep()
        observation = extractor.finish()

        body_id = model.body("tip").id
        body_to_world = data.xmat[body_id].reshape(3, 3)
        impulse_world = (
            body_to_world @ observation["fingertip_impulse_local"][0]
        )
        momentum_change = model.body_mass[body_id] * (
            data.qvel[:3] - velocity_before
        )
        np.testing.assert_allclose(
            impulse_world,
            momentum_change,
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            observation["sim_time"],
            [model.opt.timestep],
            atol=1e-12,
        )

    def test_non_rigid_contact_ids_do_not_use_numpy_negative_indexing(self):
        model, data = _contact_model()
        extractor = FingertipContactExtractor(model, data, ("tip",))

        self.assertEqual(
            extractor._fingertip_index_for_geom(model.geom("tip_geom").id),
            0,
        )
        self.assertEqual(extractor._fingertip_index_for_geom(-1), -1)
        self.assertEqual(extractor._fingertip_index_for_geom(model.ngeom), -1)

    def test_fingertip_requires_a_collision_enabled_geom(self):
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="tip">
                  <freejoint/>
                  <geom type="sphere" size="0.05"
                        contype="0" conaffinity="0"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        with self.assertRaisesRegex(ValueError, "no collision-enabled"):
            FingertipContactExtractor(model, data, ("tip",))

    def test_reset_and_call_order_validation(self):
        model, data = _contact_model()
        extractor = FingertipContactExtractor(model, data, ("tip",))

        with self.assertRaises(RuntimeError):
            extractor.accumulate_substep()
        with self.assertRaises(RuntimeError):
            extractor.finish()

        extractor.begin()
        with self.assertRaises(RuntimeError):
            extractor.finish()

        extractor.reset()
        np.testing.assert_array_equal(
            extractor.observation()["fingertip_wrench_local"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
