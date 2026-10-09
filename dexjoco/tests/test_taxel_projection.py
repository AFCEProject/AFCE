from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from dexjoco.sim.tactile import (
    BIMANUAL_FINGERTIP_BODY_NAMES,
    SINGLE_FINGERTIP_BODY_NAMES,
    CapsuleTaxelLayout,
    ContactEvent,
    GaussianTaxelProjector,
)


_XML_DIR = (
    Path(__file__).resolve().parents[1]
    / "dexjoco"
    / "sim"
    / "envs"
    / "xmls"
)


def _capsule_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="tip_a">
              <geom name="capsule_a" type="capsule"
                    pos="0.01 -0.02 0.03"
                    quat="0.9238795325 0 0.3826834324 0"
                    size="0.012 0.010"/>
            </body>
            <body name="tip_b">
              <geom name="capsule_b" type="capsule"
                    pos="-0.02 0.01 0.04"
                    quat="0.9659258263 0.2588190451 0 0"
                    size="0.009 0.007"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _event(
    fingertip_index: int,
    position_local: np.ndarray,
    force_local: np.ndarray,
) -> ContactEvent:
    return ContactEvent(
        fingertip_index=fingertip_index,
        position_local=np.asarray(position_local, dtype=np.float64),
        force_local=np.asarray(force_local, dtype=np.float64),
        torque_local_at_contact=np.zeros(3, dtype=np.float64),
        normal_force=float(np.linalg.norm(force_local)),
    )


def _geom_rotation(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, model.geom_quat[geom_id])
    return matrix.reshape(3, 3)


class CapsuleTaxelLayoutTest(unittest.TestCase):
    def test_layout_uses_capsule_pose_and_has_inward_orthonormal_axes(self):
        model = _capsule_model()
        layout = CapsuleTaxelLayout(model, ("tip_a", "tip_b"))

        self.assertEqual(layout.num_fingertips, 2)
        self.assertEqual(layout.num_taxels, 16)
        self.assertEqual(layout.positions_local.shape, (2, 16, 3))
        self.assertEqual(layout.axes_body.shape, (2, 16, 3, 3))
        self.assertFalse(layout.centers_body.flags.writeable)
        self.assertFalse(layout.axes_body.flags.writeable)
        np.testing.assert_array_equal(
            layout.axes_local,
            layout.axes_body.transpose(0, 1, 3, 2),
        )

        for fingertip_index, geom_id in enumerate(layout.geom_ids):
            geom_rotation = _geom_rotation(model, int(geom_id))
            capsule_axis = geom_rotation[:, 2]
            geom_center = model.geom_pos[geom_id]
            radius = model.geom_size[geom_id, 0]
            half_length = model.geom_size[geom_id, 1]

            axial_coordinates = []
            for position, axes_body in zip(
                layout.positions_local[fingertip_index],
                layout.axes_body[fingertip_index],
                strict=True,
            ):
                np.testing.assert_allclose(
                    axes_body.T @ axes_body,
                    np.eye(3),
                    atol=1e-12,
                )
                self.assertAlmostEqual(float(np.linalg.det(axes_body)), 1.0)

                relative_position = position - geom_center
                axial_coordinate = float(relative_position @ capsule_axis)
                radial_vector = (
                    relative_position - axial_coordinate * capsule_axis
                )
                np.testing.assert_allclose(
                    np.linalg.norm(radial_vector), radius, atol=1e-12
                )
                # Taxel local z points from the capsule surface towards its
                # centreline at the same axial coordinate.
                inward = -radial_vector / np.linalg.norm(radial_vector)
                np.testing.assert_allclose(axes_body[:, 2], inward, atol=1e-12)
                axial_coordinates.append(axial_coordinate)

            self.assertAlmostEqual(min(axial_coordinates), -half_length)
            self.assertAlmostEqual(max(axial_coordinates), half_length)
            self.assertGreater(layout.spacing[fingertip_index], 0.0)

    def test_all_eleven_task_models_construct_from_collision_capsules(self):
        xml_paths = sorted(_XML_DIR.glob("arena_arm_hand_*.xml"))
        self.assertEqual(len(xml_paths), 11)

        for xml_path in xml_paths:
            with self.subTest(model=xml_path.name):
                model = mujoco.MjModel.from_xml_path(str(xml_path))
                names = (
                    BIMANUAL_FINGERTIP_BODY_NAMES
                    if mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        BIMANUAL_FINGERTIP_BODY_NAMES[0],
                    )
                    >= 0
                    else SINGLE_FINGERTIP_BODY_NAMES
                )
                layout = CapsuleTaxelLayout.from_model(model, names)

                self.assertEqual(layout.num_fingertips, len(names))
                self.assertEqual(layout.num_taxels, 16)
                self.assertEqual(
                    layout.positions_local.shape,
                    (len(names), 16, 3),
                )
                self.assertTrue(np.all(np.isfinite(layout.positions_local)))
                self.assertTrue(np.all(np.isfinite(layout.axes_body)))


class GaussianTaxelProjectorTest(unittest.TestCase):
    def setUp(self):
        self.model = _capsule_model()
        self.layout = CapsuleTaxelLayout(self.model, ("tip_a", "tip_b"))

    def test_weights_are_normalized_nearest_maximal_and_underflow_safe(self):
        target_taxel = 6
        projector = GaussianTaxelProjector(self.layout, sigma=0.001)
        weights = projector.weights(
            0, self.layout.positions_local[0, target_taxel]
        )

        self.assertEqual(weights.shape, (16,))
        self.assertEqual(int(np.argmax(weights)), target_taxel)
        self.assertEqual(float(np.sum(weights)), 1.0)
        self.assertGreater(weights[target_taxel], weights.max(initial=0.0) * 0.999)

        # Subtracting the maximum log-weight keeps even a very distant contact
        # finite instead of producing sixteen zero weights and NaNs.
        far_weights = projector.weights(0, np.asarray([1e4, -2e4, 3e4]))
        self.assertTrue(np.all(np.isfinite(far_weights)))
        self.assertEqual(float(np.sum(far_weights)), 1.0)

    def test_default_sigma_comes_from_layout_spacing_and_can_be_explicit(self):
        default_projector = GaussianTaxelProjector(self.layout)
        explicit_projector = GaussianTaxelProjector(
            self.layout,
            sigma=[0.003, 0.004],
        )

        np.testing.assert_array_equal(default_projector.sigma, self.layout.spacing)
        np.testing.assert_array_equal(
            explicit_projector.sigma,
            np.asarray([0.003, 0.004]),
        )

    def test_zero_contacts_are_zero_and_projection_is_deterministic(self):
        projector = GaussianTaxelProjector(self.layout)
        zero = projector.project(())

        self.assertEqual(zero.shape, (2, 16, 3))
        self.assertEqual(zero.dtype, np.float32)
        np.testing.assert_array_equal(zero, 0.0)

        event = _event(
            1,
            self.layout.positions_local[1, 9] + np.asarray([0.001, 0.0, 0.0]),
            np.asarray([1.25, -2.5, 4.0]),
        )
        first = projector.project((event,))
        second = projector.project((event,))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], 0.0)

    def test_projection_conserves_every_contact_force(self):
        projector = GaussianTaxelProjector(self.layout, sigma=0.004)
        events = (
            _event(
                0,
                self.layout.positions_local[0, 3] + np.asarray([0.001, 0.0, 0.0]),
                np.asarray([2.0, -1.0, 5.0]),
            ),
            _event(
                1,
                self.layout.positions_local[1, 12] + np.asarray([0.0, -0.002, 0.0]),
                np.asarray([-3.0, 0.25, 1.5]),
            ),
        )

        for event in events:
            with self.subTest(fingertip=event.fingertip_index):
                projected = projector.project((event,))
                reconstructed_body_force = np.einsum(
                    "tij,tj->i",
                    self.layout.axes_body[event.fingertip_index],
                    projected[event.fingertip_index],
                )
                np.testing.assert_allclose(
                    reconstructed_body_force,
                    event.force_local,
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_multiple_contacts_superpose_and_conserve_total_force(self):
        projector = GaussianTaxelProjector(self.layout, sigma=0.005)
        events = (
            _event(
                0,
                self.layout.positions_local[0, 1],
                np.asarray([1.0, 2.0, 3.0]),
            ),
            _event(
                0,
                self.layout.positions_local[0, 14],
                np.asarray([-0.5, 0.25, 4.0]),
            ),
            _event(
                1,
                self.layout.positions_local[1, 7],
                np.asarray([2.5, -1.5, 0.75]),
            ),
        )

        combined = projector.project(events)
        separately_summed = sum(
            (projector.project((event,)) for event in events),
            start=np.zeros_like(combined),
        )
        np.testing.assert_allclose(combined, separately_summed, atol=1e-7)

        for fingertip_index in range(self.layout.num_fingertips):
            expected_force = sum(
                (
                    event.force_local
                    for event in events
                    if event.fingertip_index == fingertip_index
                ),
                start=np.zeros(3),
            )
            reconstructed_body_force = np.einsum(
                "tij,tj->i",
                self.layout.axes_body[fingertip_index],
                combined[fingertip_index],
            )
            np.testing.assert_allclose(
                reconstructed_body_force,
                expected_force,
                rtol=1e-6,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
