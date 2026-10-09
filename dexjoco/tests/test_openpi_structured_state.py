import numpy as np

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv


def _make_env(*, dual_arm: bool, structured: bool) -> DexJoCoOpenPIEnv:
    return DexJoCoOpenPIEnv(
        env_name="unused",
        camera_mapping={},
        seed=0,
        rand_full=False,
        randomize_dynamics=False,
        dual_arm=dual_arm,
        prompt="test",
        render_mode="rgb_array",
        pad_state_dim46=True,
        structured_hand_state=structured,
    )


def test_structured_single_arm_observation_matches_training_layout():
    env = _make_env(dual_arm=False, structured=True)
    native_state = np.arange(23, dtype=np.float32)

    observation = env._process_obs({"state": native_state})

    expected = np.concatenate(
        [native_state[:7], np.zeros(7), native_state[7:23], np.zeros(16)]
    ).astype(np.float32)
    np.testing.assert_array_equal(observation["state"], expected)
    np.testing.assert_array_equal(observation["hand_presence"], [True, False])
    np.testing.assert_array_equal(env._control_state, native_state)


def test_structured_dual_arm_observation_preserves_canonical_layout():
    env = _make_env(dual_arm=True, structured=True)
    native_state = np.arange(46, dtype=np.float32)

    observation = env._process_obs({"state": native_state})

    np.testing.assert_array_equal(observation["state"], native_state)
    np.testing.assert_array_equal(observation["hand_presence"], [True, True])
    np.testing.assert_array_equal(env._control_state, native_state)


def test_flat_single_arm_padding_is_unchanged():
    env = _make_env(dual_arm=False, structured=False)
    native_state = np.arange(23, dtype=np.float32)

    observation = env._process_obs({"state": native_state})

    np.testing.assert_array_equal(observation["state"][:23], native_state)
    np.testing.assert_array_equal(observation["state"][23:], np.zeros(23))
    assert "hand_presence" not in observation


def test_multitask_single_arm_observation_duplicates_wrist_into_canonical_slots():
    env = DexJoCoOpenPIEnv(
        env_name="unused",
        camera_mapping={"base": "front", "wrist": "wrist_camera"},
        seed=0,
        rand_full=False,
        randomize_dynamics=False,
        dual_arm=False,
        prompt="test",
        render_mode="rgb_array",
        pad_state_dim46=True,
        structured_hand_state=True,
    )
    wrist = np.full((224, 224, 3), 127, dtype=np.uint8)
    observation = env._process_obs(
        {
            "front": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_camera": wrist,
            "state": np.arange(23, dtype=np.float32),
        }
    )

    np.testing.assert_array_equal(observation["wrist_left"], wrist)
    np.testing.assert_array_equal(observation["wrist_right"], wrist)
