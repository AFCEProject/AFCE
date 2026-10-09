from __future__ import annotations

import unittest

import numpy as np
import zarr

from dexjoco.data.episode_store import ZarrEpisodeStore
from dexjoco.data.observation_fields import (
    build_sensor_metadata,
    derive_episode_timestamps,
    flatten_sensor_observation_groups,
    is_rgb_observation,
    stack_sensor_observation_groups,
    stack_sensor_transition_groups,
)


def _sensor_frame(value: float):
    return {
        "tactile_gt": {
            "fingertip_wrench_local": np.full(
                (1, 4, 6), value, dtype=np.float32
            ),
            "taxel_force_local": np.full(
                (1, 4, 16, 3), value, dtype=np.float32
            ),
            "sim_time": np.asarray([value * 0.02], dtype=np.float64),
        }
    }


def _complete_tactile_gt_episode(
    *, steps: int = 2, hands: int = 1
) -> dict[str, np.ndarray]:
    return {
        "tactile_gt.fingertip_wrench_local": np.zeros(
            (steps, hands, 4, 6), dtype=np.float32
        ),
        "tactile_gt.taxel_force_local": np.zeros(
            (steps, hands, 4, 16, 3), dtype=np.float32
        ),
        "tactile_gt.fingertip_normal": np.zeros(
            (steps, hands, 4), dtype=np.float32
        ),
        "tactile_gt.fingertip_force_peak": np.zeros(
            (steps, hands, 4), dtype=np.float32
        ),
        "tactile_gt.fingertip_impulse_local": np.zeros(
            (steps, hands, 4, 3), dtype=np.float32
        ),
        "tactile_gt.contact_fraction": np.zeros(
            (steps, hands, 4), dtype=np.float32
        ),
        "tactile_gt.sim_time": np.arange(steps, dtype=np.float64)[:, None]
        * 0.02,
    }


class ObservationFieldsTest(unittest.TestCase):
    def test_timestamps_use_authoritative_sim_clock_and_reject_wrong_fps(self):
        observations = [
            {"tactile_gt": {"sim_time": np.asarray([time], dtype=np.float64)}}
            for time in (1.0, 1.02, 1.04)
        ]
        np.testing.assert_allclose(
            derive_episode_timestamps(observations),
            [0.0, 0.02, 0.04],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            derive_episode_timestamps(observations, data_fps=50.0),
            [0.0, 0.02, 0.04],
            atol=1e-12,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            derive_episode_timestamps(observations, data_fps=30.0)
        np.testing.assert_allclose(
            derive_episode_timestamps([{}, {}, {}], control_dt=0.02),
            [0.0, 0.02, 0.04],
            atol=1e-12,
        )

    def test_stack_multidimensional_sensor_fields(self):
        observations = [_sensor_frame(0.0), _sensor_frame(1.0), _sensor_frame(2.0)]
        stacked = stack_sensor_observation_groups(observations)

        self.assertEqual(
            stacked["tactile_gt.fingertip_wrench_local"].shape,
            (3, 1, 4, 6),
        )
        self.assertEqual(
            stacked["tactile_gt.fingertip_wrench_local"].dtype,
            np.float32,
        )
        self.assertEqual(
            stacked["tactile_gt.taxel_force_local"].shape,
            (3, 1, 4, 16, 3),
        )
        self.assertEqual(
            stacked["tactile_gt.taxel_force_local"].dtype,
            np.float32,
        )
        self.assertEqual(stacked["tactile_gt.sim_time"].shape, (3, 1))
        self.assertEqual(stacked["tactile_gt.sim_time"].dtype, np.float64)

        next_stacked = stack_sensor_observation_groups(
            observations, dataset_prefix="next_"
        )
        self.assertIn(
            "next_tactile_gt.fingertip_wrench_local",
            next_stacked,
        )
        self.assertIn("next_tactile_gt.taxel_force_local", next_stacked)

    def test_partial_or_changing_schema_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "schema changed"):
            stack_sensor_observation_groups([_sensor_frame(0.0), {}])

        changed = _sensor_frame(1.0)
        changed["tactile_gt"]["fingertip_wrench_local"] = np.zeros(
            (1, 3, 6), dtype=np.float32
        )
        with self.assertRaisesRegex(ValueError, "changed shape"):
            stack_sensor_observation_groups([_sensor_frame(0.0), changed])

        changed_dtype = _sensor_frame(1.0)
        changed_dtype["tactile_gt"]["fingertip_wrench_local"] = np.zeros(
            (1, 4, 6), dtype=np.float64
        )
        with self.assertRaisesRegex(ValueError, "changed dtype"):
            stack_sensor_observation_groups(
                [_sensor_frame(0.0), changed_dtype]
            )

    def test_pre_and_next_sensor_schemas_must_match(self):
        self.assertEqual(
            stack_sensor_transition_groups([{}, {}], [{}, {}]),
            {},
        )

        current = [_sensor_frame(0.0), _sensor_frame(1.0)]
        following = [_sensor_frame(1.0), _sensor_frame(2.0)]
        stacked = stack_sensor_transition_groups(current, following)

        np.testing.assert_array_equal(
            stacked["tactile_gt.fingertip_wrench_local"][:, 0, 0, 0],
            [0.0, 1.0],
        )
        np.testing.assert_array_equal(
            stacked["next_tactile_gt.fingertip_wrench_local"][:, 0, 0, 0],
            [1.0, 2.0],
        )
        np.testing.assert_array_equal(
            stacked["next_tactile_gt.taxel_force_local"][:, 0, 0, 0, 0],
            [1.0, 2.0],
        )

        with self.assertRaisesRegex(ValueError, "same length"):
            stack_sensor_transition_groups(current, following[:1])
        with self.assertRaisesRegex(ValueError, "schemas differ"):
            stack_sensor_transition_groups(current, [{}, {}])

        next_float64 = [_sensor_frame(1.0), _sensor_frame(2.0)]
        for frame in next_float64:
            frame["tactile_gt"]["fingertip_wrench_local"] = frame[
                "tactile_gt"
            ]["fingertip_wrench_local"].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "dtype differs"):
            stack_sensor_transition_groups(current, next_float64)

    def test_field_names_values_and_boolean_masks_are_validated(self):
        valid = {
            "tactile_gt": {
                "hand_valid": np.asarray([True, False], dtype=np.bool_),
            }
        }
        flattened = flatten_sensor_observation_groups(valid)
        self.assertEqual(flattened["tactile_gt.hand_valid"].dtype, np.bool_)

        for invalid_name in ("bad.name", "bad/name"):
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaisesRegex(ValueError, "cannot contain"):
                    flatten_sensor_observation_groups(
                        {
                            "tactile_gt": {
                                invalid_name: np.zeros((1,), dtype=np.float32)
                            }
                        }
                    )

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            flatten_sensor_observation_groups(
                valid,
                groups=("tactile_gt", "tactile_gt"),
            )

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            flatten_sensor_observation_groups({"tactile_gt": {}})

        nonfinite = _sensor_frame(0.0)
        nonfinite["tactile_gt"]["fingertip_wrench_local"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            flatten_sensor_observation_groups(nonfinite)

    def test_structured_ground_truth_force_is_not_misclassified_as_rgb(self):
        self.assertFalse(
            is_rgb_observation(
                "tactile_gt",
                np.zeros((1, 4, 3), dtype=np.uint8),
            )
        )
        self.assertTrue(
            is_rgb_observation(
                "wrist",
                np.zeros((64, 64, 3), dtype=np.uint8),
            )
        )
        self.assertTrue(
            is_rgb_observation(
                "wrist",
                np.zeros((1, 64, 64, 3), dtype=np.uint8),
            )
        )
        self.assertFalse(
            is_rgb_observation(
                "wrist",
                np.zeros((2, 4, 5, 3), dtype=np.uint8),
            )
        )
        self.assertFalse(
            is_rgb_observation(
                "wrist",
                np.zeros((64, 64, 3), dtype=np.float32),
            )
        )

    def test_retired_tactile_input_fails_closed(self):
        retired_observation = {
            "tactile": {
                "retired_field": np.zeros((1, 4, 3), dtype=np.float32),
            }
        }
        with self.assertRaisesRegex(ValueError, "retired 'tactile'"):
            flatten_sensor_observation_groups(retired_observation)

        retired_episode = {
            "tactile.retired_field": np.zeros((2, 1, 4, 3), dtype=np.float32),
            "next_tactile.retired_field": np.zeros(
                (2, 1, 4, 3), dtype=np.float32
            ),
        }
        with self.assertRaisesRegex(ValueError, "Retired tactile"):
            build_sensor_metadata(retired_episode)

    def test_tactile_gt_v2_metadata_describes_taxel_projection(self):
        episode = _complete_tactile_gt_episode(hands=2)
        metadata = build_sensor_metadata(episode)["tactile_gt"]

        self.assertEqual(metadata["schema_version"], "dexjoco.tactile_gt.v2")
        self.assertEqual(metadata["hand_order"], ["right", "left"])
        self.assertEqual(metadata["taxels_per_fingertip"], 16)
        self.assertEqual(metadata["taxel_grid_shape"], [4, 4])
        self.assertEqual(
            metadata["taxel_channels"],
            ["tangent_u", "tangent_v", "normal_inward"],
        )
        self.assertIn("Gaussian", metadata["taxel_projection"]["method"])
        self.assertEqual(
            metadata["field_units"]["taxel_force_local"], ["N", "N", "N"]
        )
        self.assertEqual(
            metadata["field_specs"]["taxel_force_local"]["axes"],
            ["time", "hand", "fingertip", "taxel", "xyz"],
        )

    def test_tactile_gt_v2_does_not_accept_missing_or_malformed_taxels(self):
        missing = _complete_tactile_gt_episode()
        missing.pop("tactile_gt.taxel_force_local")
        with self.assertRaisesRegex(ValueError, "taxel_force_local"):
            build_sensor_metadata(missing)

        wrong_shape = _complete_tactile_gt_episode()
        wrong_shape["tactile_gt.taxel_force_local"] = np.zeros(
            (2, 1, 4, 15, 3), dtype=np.float32
        )
        with self.assertRaisesRegex(ValueError, "must have shape"):
            build_sensor_metadata(wrong_shape)

    def test_episode_store_round_trip_preserves_trailing_shape(self):
        store = ZarrEpisodeStore.create_empty(storage=zarr.MemoryStore())
        episode1 = {
            "action": np.zeros((2, 23), dtype=np.float32),
            "tactile_gt.fingertip_wrench_local": np.ones(
                (2, 1, 4, 6), dtype=np.float32
            ),
        }
        episode2 = {
            "action": np.zeros((1, 23), dtype=np.float32),
            "tactile_gt.fingertip_wrench_local": np.full(
                (1, 1, 4, 6), 2.0, dtype=np.float32
            ),
        }

        store.append_episode(episode1, compressors=None)
        store.append_episode(episode2, compressors=None)

        self.assertEqual(
            store.data["tactile_gt.fingertip_wrench_local"].shape,
            (3, 1, 4, 6),
        )
        np.testing.assert_array_equal(store.episode_ends[:], [2, 3])
        np.testing.assert_array_equal(
            store.data["tactile_gt.fingertip_wrench_local"][2],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
