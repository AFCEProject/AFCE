import pathlib

import numpy as np

from openpi.training import data_loader as _data_loader


class _ListDataset:
    def __init__(self, source: str, length: int):
        self._samples = [{"source": source, "local_index": index} for index in range(length)]

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)


def test_task_balanced_multi_source_indices():
    dataset = _data_loader.MultiSourceDataset(
        [
            _ListDataset("click_mouse", 2),
            _ListDataset("fold_glasses", 5),
            _ListDataset("hammer_nail", 3),
            _ListDataset("pick_bucket", 4),
            _ListDataset("pinch_tongs", 1),
            _ListDataset("water_plant", 5),
        ],
        balance="task",
    )

    assert len(dataset) == 30
    samples = [dataset[index] for index in range(len(dataset))]
    source_counts = {
        source: [sample["source"] for sample in samples].count(source)
        for source in ("click_mouse", "fold_glasses", "hammer_nail", "pick_bucket", "pinch_tongs", "water_plant")
    }
    assert set(source_counts.values()) == {5}
    short_indices = [sample["local_index"] for sample in samples if sample["source"] == "click_mouse"]
    assert short_indices == [0, 0, 0, 1, 1]
    assert (
        max(short_indices.count(index) for index in range(2)) - min(short_indices.count(index) for index in range(2))
        == 1
    )
    assert dataset.resolve_index(-1) == (5, 4)


def test_source_base_image_key_is_renamed_without_copying_value():
    image_value = object()
    original_sample = {
        "observation.images.ego_right": image_value,
        "observation.images.wrist": object(),
    }
    dataset = _data_loader.RenamedKeyDataset(
        [original_sample],
        source_key="observation.images.ego_right",
        target_key="observation.images.base",
    )

    renamed = dataset[0]

    assert "observation.images.ego_right" not in renamed
    assert renamed["observation.images.base"] is image_value
    assert original_sample["observation.images.ego_right"] is image_value
    assert "observation.images.base" not in original_sample


def test_dexjoco_source_is_canonicalized_and_right_padded_without_copying_images():
    base_image = object()
    wrist_image = object()
    original_sample = {
        "observation.images.front": base_image,
        "observation.images.wrist": wrist_image,
        "observation.state": np.arange(23, dtype=np.float32),
        "action": np.arange(60 * 22, dtype=np.float32).reshape(60, 22),
    }
    source = _data_loader._config.LeRobotDatasetSource(  # noqa: SLF001
        root=pathlib.Path("/unused"),
        base_image_key="observation.images.front",
        wrist_left_image_key="observation.images.wrist",
        wrist_right_image_key="observation.images.wrist",
    )
    dataset = _data_loader.CanonicalizedDexJoCoDataset(
        [original_sample],
        source,
        target_state_dim=46,
        target_action_dim=44,
    )

    canonical = dataset[0]

    assert canonical["observation.images.base"] is base_image
    assert canonical["observation.images.wrist1"] is wrist_image
    assert canonical["observation.images.wrist2"] is wrist_image
    assert canonical["observation.state"].shape == (46,)
    assert canonical["action"].shape == (60, 44)
    np.testing.assert_array_equal(canonical["observation.state"][:23], original_sample["observation.state"])
    np.testing.assert_array_equal(canonical["observation.state"][23:], 0)
    np.testing.assert_array_equal(canonical["action"][:, :22], original_sample["action"])
    np.testing.assert_array_equal(canonical["action"][:, 22:], 0)
    assert "observation.images.base" not in original_sample


def test_dexjoco_single_arm_structured_state_uses_bimanual_semantic_slots():
    state = np.arange(23, dtype=np.float32)
    sample = {
        "observation.images.front": object(),
        "observation.images.wrist": object(),
        "observation.state": state,
        "action": np.arange(2 * 22, dtype=np.float32).reshape(2, 22),
    }
    source = _data_loader._config.LeRobotDatasetSource(  # noqa: SLF001
        root=pathlib.Path("/unused"),
        base_image_key="observation.images.front",
        wrist_left_image_key="observation.images.wrist",
        wrist_right_image_key="observation.images.wrist",
    )
    dataset = _data_loader.CanonicalizedDexJoCoDataset(
        [sample],
        source,
        target_state_dim=46,
        target_action_dim=44,
        structured_hand_state=True,
    )

    canonical = dataset[0]

    np.testing.assert_array_equal(canonical["observation.state"][:7], state[:7])
    np.testing.assert_array_equal(canonical["observation.state"][7:14], 0)
    np.testing.assert_array_equal(canonical["observation.state"][14:30], state[7:23])
    np.testing.assert_array_equal(canonical["observation.state"][30:46], 0)
    np.testing.assert_array_equal(canonical["observation.hand_presence"], [True, False])


def test_dexjoco_bimanual_structured_state_is_not_reordered():
    state = np.arange(46, dtype=np.float32)
    sample = {
        "observation.images.ego": object(),
        "observation.images.wrist_left": object(),
        "observation.images.wrist_right": object(),
        "observation.state": state,
        "action": np.arange(2 * 44, dtype=np.float32).reshape(2, 44),
    }
    source = _data_loader._config.LeRobotDatasetSource(  # noqa: SLF001
        root=pathlib.Path("/unused"),
        base_image_key="observation.images.ego",
        wrist_left_image_key="observation.images.wrist_left",
        wrist_right_image_key="observation.images.wrist_right",
    )
    dataset = _data_loader.CanonicalizedDexJoCoDataset(
        [sample],
        source,
        target_state_dim=46,
        target_action_dim=44,
        structured_hand_state=True,
    )

    canonical = dataset[0]

    np.testing.assert_array_equal(canonical["observation.state"], state)
    np.testing.assert_array_equal(canonical["observation.hand_presence"], [True, True])


def test_video_backend_is_only_forwarded_when_explicit(monkeypatch):
    calls = []

    class _Metadata:
        fps = 30

        def __init__(self, repo_id, *, root):
            del repo_id, root

    class _LeRobotDataset:
        def __init__(self, repo_id, **kwargs):
            del repo_id
            calls.append(kwargs)

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _Metadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _LeRobotDataset)

    common = {
        "repo_id": "local_repo",
        "root": "/dataset",
        "action_sequence_keys": ("action",),
        "action_horizon": 2,
        "prompt_from_task": False,
    }
    _data_loader._create_lerobot_dataset(**common, video_backend="pyav")  # noqa: SLF001
    _data_loader._create_lerobot_dataset(**common, video_backend=None)  # noqa: SLF001

    assert calls[0]["video_backend"] == "pyav"
    assert "video_backend" not in calls[1]


def test_single_source_preserves_lerobot_default_root(monkeypatch):
    roots = []

    class _Metadata:
        fps = 30

        def __init__(self, repo_id, *, root):
            del repo_id
            roots.append(("metadata", root))

    class _LeRobotDataset:
        def __init__(self, repo_id, **kwargs):
            del repo_id
            roots.append(("dataset", kwargs["root"]))

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _Metadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _LeRobotDataset)

    data_config = _data_loader._config.DataConfig(repo_id="remote_repo", root=None)  # noqa: SLF001
    _data_loader.create_torch_dataset(data_config, action_horizon=2, model_config=object())

    assert roots == [("metadata", None), ("dataset", None)]
