from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import zarr
from absl import flags as absl_flags
from absl.testing import flagsaver


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "record_demos_zarr.py"
_SPEC = importlib.util.spec_from_file_location(
    "_dexjoco_record_demos_zarr_for_test",
    _SCRIPT_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load recorder script from {_SCRIPT_PATH}.")
_RECORDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RECORDER)


def _define_flag_if_missing(definer):
    def define(name, *args, **kwargs):
        if name in absl_flags.FLAGS:
            return None
        return definer(name, *args, **kwargs)

    return define


_REPLAY_SCRIPT_PATH = _REPO_ROOT / "scripts" / "replay_demos_zarr.py"
_REPLAY_SPEC = importlib.util.spec_from_file_location(
    "_dexjoco_replay_demos_zarr_for_test",
    _REPLAY_SCRIPT_PATH,
)
if _REPLAY_SPEC is None or _REPLAY_SPEC.loader is None:
    raise RuntimeError(
        f"Could not load replay script from {_REPLAY_SCRIPT_PATH}."
    )
_REPLAYER = importlib.util.module_from_spec(_REPLAY_SPEC)
with (
    mock.patch.object(
        absl_flags,
        "DEFINE_string",
        _define_flag_if_missing(absl_flags.DEFINE_string),
    ),
    mock.patch.object(
        absl_flags,
        "DEFINE_integer",
        _define_flag_if_missing(absl_flags.DEFINE_integer),
    ),
    mock.patch.object(
        absl_flags,
        "DEFINE_float",
        _define_flag_if_missing(absl_flags.DEFINE_float),
    ),
    mock.patch.object(
        absl_flags,
        "DEFINE_bool",
        _define_flag_if_missing(absl_flags.DEFINE_bool),
    ),
):
    _REPLAY_SPEC.loader.exec_module(_REPLAYER)


def _sensor_observation(value: float, sim_time: float) -> dict:
    return {
        "tactile_gt": {
            "fingertip_wrench_local": np.full(
                (1, 4, 6), value, dtype=np.float32
            ),
            "fingertip_normal": np.full(
                (1, 4), abs(value), dtype=np.float32
            ),
            "fingertip_force_peak": np.full(
                (1, 4), abs(value), dtype=np.float32
            ),
            "fingertip_impulse_local": np.full(
                (1, 4, 3), value * 0.02, dtype=np.float32
            ),
            "contact_fraction": np.full(
                (1, 4), float(value != 0.0), dtype=np.float32
            ),
            "taxel_force_local": np.full(
                (1, 4, 16, 3), value, dtype=np.float32
            ),
            "sim_time": np.asarray([sim_time], dtype=np.float64),
        }
    }


def _retired_sensor_observation() -> dict:
    return {
        "tactile": {
            "retired_field": np.zeros((1, 4, 3), dtype=np.float32),
        }
    }


class _FakeReplayEnvironment:
    control_dt = 0.02

    @property
    def unwrapped(self):
        return self

    def reset(self):
        return {
            "state": np.zeros((1,), dtype=np.float64),
            **_sensor_observation(0.0, 0.0),
        }, {}

    def step(self, action):
        del action
        return (
            {
                "state": np.ones((1,), dtype=np.float64),
                **_sensor_observation(1.0, 0.02),
            },
            1.0,
            True,
            False,
            {"succeed": True},
        )

    def observation(self, observation):
        return observation

    def close(self):
        pass


class _FakeReplayConfig:
    def __init__(self):
        self.environment = _FakeReplayEnvironment()
        self.environment_kwargs = None

    def get_environment(self, **kwargs):
        self.environment_kwargs = kwargs
        return self.environment


class RecordDemosTactileIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _RECORDER.FLAGS.is_parsed():
            _RECORDER.FLAGS(["test_record_demos_tactile"])

    @flagsaver.flagsaver(data_fps=50.0, save_depth=False)
    def test_writer_preserves_pre_next_alignment_and_terminal_response(self):
        action = np.zeros(23, dtype=np.float64)
        action[3] = 1.0
        trajectory = []
        for index in range(2):
            observation = {
                "state": np.asarray([index], dtype=np.float64),
                **_sensor_observation(float(index), index * 0.02),
            }
            next_sensor = _sensor_observation(
                float(index + 10),
                (index + 1) * 0.02,
            )
            trajectory.append(
                {
                    "observations": observation,
                    "next_sensor_observations": next_sensor,
                    "actions": action.copy(),
                    "dones": index == 1,
                    "infos": {"succeed": index == 1},
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            demo_dir = Path(
                _RECORDER._write_demo_zarr_and_videos(
                    trajectory,
                    exp_name="pick_bucket",
                    success_index=1,
                    base_out=Path(temp_dir),
                    video_fps=30,
                )
            )
            root = zarr.open(str(demo_dir / "replay.zarr"), mode="r")
            data = root["data"]

            np.testing.assert_array_equal(
                data["tactile_gt.fingertip_wrench_local"][:, 0, 0, 0],
                [0.0, 1.0],
            )
            np.testing.assert_array_equal(
                data["next_tactile_gt.fingertip_wrench_local"][:, 0, 0, 0],
                [10.0, 11.0],
            )
            np.testing.assert_array_equal(
                data["tactile_gt.taxel_force_local"][:, 0, 0, 0, 0],
                [0.0, 1.0],
            )
            np.testing.assert_array_equal(
                data["next_tactile_gt.taxel_force_local"][:, 0, 0, 0, 0],
                [10.0, 11.0],
            )
            np.testing.assert_allclose(
                data["tactile_gt.sim_time"][:, 0],
                [0.0, 0.02],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                data["next_tactile_gt.sim_time"][:, 0],
                [0.02, 0.04],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                data["timestamp"][:],
                [0.0, 0.02],
                atol=1e-12,
            )
            np.testing.assert_array_equal(root["meta"]["episode_ends"][:], [2])
            tactile_metadata = root.attrs["tactile_gt"]
            self.assertEqual(
                tactile_metadata["schema_version"],
                "dexjoco.tactile_gt.v2",
            )
            self.assertEqual(tactile_metadata["hand_order"], ["right"])
            self.assertEqual(
                tactile_metadata["fingertip_order"],
                ["index", "middle", "ring", "thumb"],
            )
            self.assertEqual(tactile_metadata["taxel_grid_shape"], [4, 4])
            self.assertEqual(
                tactile_metadata["taxel_channels"],
                ["tangent_u", "tangent_v", "normal_inward"],
            )
            data_keys = set(data.array_keys())
            self.assertFalse(
                any(
                    key.startswith(("tactile.", "next_tactile."))
                    for key in data_keys
                )
            )
            self.assertNotIn("tactile", root.attrs)
            self.assertTrue((demo_dir / "videos").is_dir())

    @flagsaver.flagsaver(data_fps=50.0, save_depth=False)
    def test_invalid_sensor_schema_does_not_create_partial_demo(self):
        action = np.zeros(23, dtype=np.float64)
        trajectory = [
            {
                "observations": {
                    "state": np.zeros((1,), dtype=np.float64),
                    **_sensor_observation(0.0, 0.0),
                },
                "next_sensor_observations": {},
                "actions": action,
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "schemas differ"):
                _RECORDER._write_demo_zarr_and_videos(
                    trajectory,
                    exp_name="pick_bucket",
                    success_index=1,
                    base_out=output,
                    video_fps=30,
                )
            self.assertEqual(list(output.iterdir()), [])

            malformed = _sensor_observation(0.0, 0.0)
            malformed["tactile_gt"]["fingertip_wrench_local"] = np.zeros(
                (1, 4, 5), dtype=np.float32
            )
            malformed_trajectory = [
                {
                    "observations": {
                        "state": np.zeros((1,), dtype=np.float64),
                        **malformed,
                    },
                    "next_sensor_observations": malformed,
                    "actions": action,
                }
            ]
            with self.assertRaisesRegex(ValueError, "hands, fingertips, 6"):
                _RECORDER._write_demo_zarr_and_videos(
                    malformed_trajectory,
                    exp_name="pick_bucket",
                    success_index=2,
                    base_out=output,
                    video_fps=30,
                )
            self.assertEqual(list(output.iterdir()), [])

    @flagsaver.flagsaver(data_fps=50.0, save_depth=False)
    def test_record_and_replay_writers_reject_retired_tactile(self):
        action = np.zeros(23, dtype=np.float64)
        observation = {
            "state": np.zeros((1,), dtype=np.float64),
            **_sensor_observation(0.0, 0.0),
            **_retired_sensor_observation(),
        }
        next_observation = {
            **_sensor_observation(1.0, 0.02),
            **_retired_sensor_observation(),
        }
        trajectory = [
            {
                "observations": observation,
                "next_sensor_observations": next_observation,
                "actions": action,
            }
        ]

        for writer in (
            _RECORDER._write_demo_zarr_and_videos,
            _REPLAYER._write_demo_zarr_and_videos,
        ):
            with (
                self.subTest(writer=writer.__module__),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                output = Path(temp_dir)
                with self.assertRaisesRegex(
                    ValueError,
                    "retired 'tactile'",
                ):
                    writer(
                        trajectory,
                        exp_name="pick_bucket",
                        success_index=1,
                        base_out=output,
                        video_fps=30,
                        control_dt=0.02,
                    )
                self.assertEqual(list(output.iterdir()), [])


class ReplayDemosTactileIntegrationTest(unittest.TestCase):
    @flagsaver.flagsaver(
        randomize=False,
        restore_state=False,
        extend_steps=0,
        save_depth=False,
        data_only=False,
    )
    def test_replay_single_demo_returns_gt_only_three_value_contract(self):
        config = _FakeReplayConfig()
        actions = np.zeros((1, 23), dtype=np.float64)

        with mock.patch.object(
            _REPLAYER,
            "tqdm",
            lambda iterable, desc: iterable,
        ):
            result = _REPLAYER._replay_single_demo(
                actions,
                initial_state=None,
                task_id="pick_bucket",
                config=config,
                env_seed=7,
                desc="unit-test",
            )

        self.assertEqual(len(result), 3)
        succeed, trajectory, control_dt = result
        self.assertTrue(succeed)
        self.assertEqual(control_dt, 0.02)
        self.assertEqual(len(trajectory), 1)
        self.assertNotIn("tactile", trajectory[0]["observations"])
        self.assertEqual(
            set(trajectory[0]["next_sensor_observations"]),
            {"tactile_gt"},
        )
        self.assertEqual(config.environment_kwargs["render_mode"], "rgb_array")

    @flagsaver.flagsaver(
        randomize=False,
        restore_state=False,
        extend_steps=0,
        save_depth=False,
        data_only=True,
    )
    def test_data_only_replay_requests_no_render_environment(self):
        config = _FakeReplayConfig()
        actions = np.zeros((1, 23), dtype=np.float64)

        with mock.patch.object(
            _REPLAYER,
            "tqdm",
            lambda iterable, desc: iterable,
        ):
            succeed, trajectory, control_dt = _REPLAYER._replay_single_demo(
                actions,
                initial_state=None,
                task_id="pinch_tongs",
                config=config,
                env_seed=7,
                desc="unit-test-data-only",
            )

        self.assertTrue(succeed)
        self.assertEqual(control_dt, 0.02)
        self.assertEqual(len(trajectory), 1)
        self.assertEqual(config.environment_kwargs["render_mode"], "none")
        self.assertEqual(
            _REPLAYER._collect_camera_keys(trajectory[0]["observations"]), []
        )

    @flagsaver.flagsaver(data_only=True)
    def test_data_only_fails_closed_on_rgb_observation(self):
        observation = {
            "state": np.zeros((1,), dtype=np.float64),
            "front": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        with self.assertRaisesRegex(RuntimeError, "unexpectedly received RGB"):
            _REPLAYER._validate_data_only_observation(observation, stage="test")

    @flagsaver.flagsaver(data_fps=50.0, save_depth=False, data_only=False)
    def test_replay_writer_persists_gt_only(self):
        action = np.zeros(23, dtype=np.float64)
        trajectory = []
        for index in range(2):
            trajectory.append(
                {
                    "observations": {
                        "state": np.asarray([index], dtype=np.float64),
                        **_sensor_observation(float(index), index * 0.02),
                    },
                    "next_sensor_observations": _sensor_observation(
                        float(index + 1),
                        (index + 1) * 0.02,
                    ),
                    "actions": action.copy(),
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            demo_dir = Path(
                _REPLAYER._write_demo_zarr_and_videos(
                    trajectory,
                    exp_name="pick_bucket",
                    success_index=1,
                    base_out=Path(temp_dir),
                    video_fps=30,
                    control_dt=0.02,
                )
            )
            root = zarr.open(str(demo_dir / "replay.zarr"), mode="r")
            data = root["data"]
            np.testing.assert_array_equal(
                data["next_tactile_gt.fingertip_wrench_local"][:, 0, 0, 0],
                [1.0, 2.0],
            )
            np.testing.assert_array_equal(
                data["next_tactile_gt.taxel_force_local"][:, 0, 0, 0, 0],
                [1.0, 2.0],
            )
            data_keys = set(data.array_keys())
            self.assertFalse(
                any(
                    key.startswith(("tactile.", "next_tactile."))
                    for key in data_keys
                )
            )
            self.assertIn("tactile_gt", root.attrs)
            self.assertNotIn("tactile", root.attrs)
            np.testing.assert_allclose(data["timestamp"][:], [0.0, 0.02])
            self.assertTrue((demo_dir / "videos").is_dir())
            self.assertNotIn("replay_output", root.attrs)

    @flagsaver.flagsaver(
        data_fps=50.0,
        save_depth=False,
        data_only=True,
        randomize=False,
        restore_state=True,
        extend_steps=0,
    )
    def test_data_only_writer_marks_metadata_and_omits_video_directory(self):
        action = np.zeros(23, dtype=np.float64)
        action[3] = 1.0
        trajectory = []
        for index in range(2):
            trajectory.append(
                {
                    "observations": {
                        "state": np.asarray([index], dtype=np.float64),
                        **_sensor_observation(float(index), index * 0.02),
                    },
                    "next_sensor_observations": _sensor_observation(
                        float(index + 1),
                        (index + 1) * 0.02,
                    ),
                    "actions": action.copy(),
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            demo_dir = Path(
                _REPLAYER._write_demo_zarr_and_videos(
                    trajectory,
                    exp_name="pinch_tongs",
                    success_index=1,
                    base_out=Path(temp_dir),
                    video_fps=30,
                    control_dt=0.02,
                    source_action_steps=2,
                    initial_state_actually_restored=True,
                )
            )
            root = zarr.open(str(demo_dir / "replay.zarr"), mode="r")
            metadata = root.attrs["replay_output"]
            self.assertEqual(
                metadata["schema_version"], "dexjoco.replay_output.v1"
            )
            self.assertEqual(metadata["mode"], "data_only")
            self.assertTrue(metadata["data_only"])
            self.assertEqual(metadata["task"], "pinch_tongs")
            self.assertFalse(metadata["randomize"])
            self.assertFalse(metadata["randomize_dynamics"])
            self.assertTrue(metadata["restore_state_requested"])
            self.assertTrue(metadata["initial_state_actually_restored"])
            self.assertEqual(metadata["extend_steps"], 0)
            self.assertEqual(metadata["source_action_steps"], 2)
            self.assertFalse(metadata["rgb_observations_generated"])
            self.assertFalse(metadata["rgb_videos_written"])
            self.assertFalse(metadata["depth_outputs_written"])
            self.assertFalse((demo_dir / "videos").exists())

            data = root["data"]
            np.testing.assert_array_equal(
                data["action"][:], np.stack([action, action])
            )
            self.assertIn("state", data)
            self.assertIn("action_rotvec", data)
            self.assertIn("tactile_gt.taxel_force_local", data)
            self.assertIn("next_tactile_gt.taxel_force_local", data)

    @flagsaver.flagsaver(data_only=True, save_depth=True)
    def test_data_only_rejects_depth_capture(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _REPLAYER.main([])


if __name__ == "__main__":
    unittest.main()
