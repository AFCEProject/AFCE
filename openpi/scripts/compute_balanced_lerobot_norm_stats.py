"""Compute task-balanced OpenPI normalization statistics from six LeRobot v3 datasets.

This utility intentionally reads only ``meta/*.json`` and parquet columns. It never
loads or decodes video. Each of the six fixed DexJoCo task roots contributes exactly
one sixth of the state distribution and one sixth of the action distribution,
regardless of its number of frames.

OpenPI requests a 30-step action chunk at every dataset frame. LeRobot repeat-pads
queries past an episode boundary with that episode's last action. Instead of
materializing every chunk, this script assigns each stored action row the exact
number of times it occurs in those chunks. The resulting action mean, standard
deviation, and quantiles are therefore equivalent to flattening all repeat-padded
``(num_frames, 30, action_dim)`` chunks within each task before task balancing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.shared import normalize

TASKS = (
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
STATE_DIM = 23
ACTION_DIM = 22
ACTION_HORIZON = 30

EPISODE_COLUMNS = (
    "episode_index",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
)
DATA_COLUMNS = (
    STATE_COLUMN,
    ACTION_COLUMN,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


class DatasetValidationError(ValueError):
    """Raised when a LeRobot root violates the expected v3 single-task contract."""


@dataclasses.dataclass(frozen=True)
class _Episode:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    data_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class _TaskSamples:
    task: str
    states: np.ndarray
    actions: np.ndarray
    action_multiplicities: np.ndarray
    episode_count: int

    @property
    def frame_count(self) -> int:
        return int(self.states.shape[0])


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise DatasetValidationError(message)


def _read_json_object(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetValidationError(f"Missing required metadata file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"Cannot read valid JSON object from {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetValidationError(f"Metadata must be a JSON object: {path}")
    return value


def _integer(value: Any, *, context: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise DatasetValidationError(f"{context} must be an integer, got boolean {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DatasetValidationError(f"{context} must be an integer, got {value!r}") from error
    if isinstance(value, (float, np.floating)) and (not np.isfinite(value) or float(value) != result):
        raise DatasetValidationError(f"{context} must be an exact integer, got {value!r}")
    if isinstance(value, str) and value.strip() != str(result):
        raise DatasetValidationError(f"{context} must be an integer, got {value!r}")
    if minimum is not None and result < minimum:
        raise DatasetValidationError(f"{context} must be >= {minimum}, got {result}")
    return result


def _resolve_inside(root: pathlib.Path, relative: str, *, context: str) -> pathlib.Path:
    relative_path = pathlib.Path(relative)
    if relative_path.is_absolute():
        raise DatasetValidationError(f"{context} must be relative to {root}, got {relative!r}")
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise DatasetValidationError(f"{context} escapes dataset root {root}: {relative!r}")
    return resolved


def _validate_info(
    root: pathlib.Path,
    *,
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
) -> tuple[dict[str, Any], int, int, int]:
    info = _read_json_object(root / "meta/info.json")
    _require(
        info.get("codebase_version") == "v3.0",
        f"{root}: codebase_version must be exactly 'v3.0', got {info.get('codebase_version')!r}",
    )
    total_episodes = _integer(info.get("total_episodes"), context=f"{root}: total_episodes", minimum=1)
    total_frames = _integer(info.get("total_frames"), context=f"{root}: total_frames", minimum=1)
    total_tasks = _integer(info.get("total_tasks"), context=f"{root}: total_tasks", minimum=1)
    _require(total_tasks == 1, f"{root}: expected a single-task root, but total_tasks is {total_tasks}")
    _integer(info.get("chunks_size"), context=f"{root}: chunks_size", minimum=1)
    fps = _integer(info.get("fps"), context=f"{root}: fps", minimum=1)

    data_path = info.get("data_path")
    _require(isinstance(data_path, str) and data_path, f"{root}: data_path must be a non-empty string")
    for placeholder in ("{chunk_index", "{file_index"):
        _require(placeholder in data_path, f"{root}: data_path is missing placeholder {placeholder!r}: {data_path!r}")

    features = info.get("features")
    _require(isinstance(features, dict), f"{root}: features must be an object")
    expected_features = {STATE_COLUMN: state_dim, ACTION_COLUMN: action_dim}
    for key, dimension in expected_features.items():
        feature = features.get(key)
        _require(isinstance(feature, dict), f"{root}: missing feature metadata for {key!r}")
        _require(
            feature.get("dtype") == "float32" and feature.get("shape") == [dimension],
            f"{root}: feature {key!r} must be float32 with shape [{dimension}], got {feature!r}",
        )
    return info, fps, total_episodes, total_frames


def _validate_single_task_table(root: pathlib.Path) -> None:
    path = root / "meta/tasks.parquet"
    if not path.is_file():
        raise DatasetValidationError(f"Missing single-task metadata table: {path}")
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as error:
        raise DatasetValidationError(f"Cannot read task metadata {path}: {error}") from error
    _require(table.num_rows == 1, f"{path}: expected exactly one task row, found {table.num_rows}")
    if "task_index" in table.column_names:
        task_indices = _integer_vector(table, "task_index", path)
        _require(np.array_equal(task_indices, np.array([0])), f"{path}: task_index must be [0]")
    task_column = next((key for key in ("task", "__index_level_0__") if key in table.column_names), None)
    _require(task_column is not None, f"{path}: expected a 'task' or '__index_level_0__' text column")
    task_value = table[task_column][0].as_py()
    _require(isinstance(task_value, str) and task_value.strip(), f"{path}: task text must be non-empty")


def _read_episode_metadata(
    root: pathlib.Path,
    info: dict[str, Any],
    *,
    total_episodes: int,
    total_frames: int,
) -> list[_Episode]:
    paths = sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not paths:
        raise DatasetValidationError(f"Missing LeRobot v3 episode metadata under {root / 'meta/episodes'}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            table = pq.read_table(path)
        except (OSError, pa.ArrowException) as error:
            raise DatasetValidationError(f"Cannot read episode metadata {path}: {error}") from error
        missing = sorted(set(EPISODE_COLUMNS) - set(table.column_names))
        _require(not missing, f"{path}: missing required episode columns: {missing}")
        rows.extend(table.select(EPISODE_COLUMNS).to_pylist())

    _require(
        len(rows) == total_episodes,
        f"{root}: episode metadata has {len(rows)} rows, but info.json declares {total_episodes}",
    )
    rows.sort(key=lambda row: _integer(row.get("episode_index"), context=f"{root}: episode_index", minimum=0))

    data_template = info["data_path"]
    episodes: list[_Episode] = []
    expected_from = 0
    for expected_episode_index, row in enumerate(rows):
        context = f"{root}: episode metadata row {expected_episode_index}"
        episode_index = _integer(row.get("episode_index"), context=f"{context} episode_index", minimum=0)
        _require(
            episode_index == expected_episode_index,
            f"{root}: episode indices must be unique and contiguous from zero; expected "
            f"{expected_episode_index}, got {episode_index}",
        )
        length = _integer(row.get("length"), context=f"{context} length", minimum=1)
        dataset_from = _integer(row.get("dataset_from_index"), context=f"{context} dataset_from_index", minimum=0)
        dataset_to = _integer(row.get("dataset_to_index"), context=f"{context} dataset_to_index", minimum=1)
        _require(
            dataset_from == expected_from,
            f"{context}: dataset intervals must be contiguous; expected start {expected_from}, got {dataset_from}",
        )
        _require(
            dataset_to - dataset_from == length,
            f"{context}: interval [{dataset_from}, {dataset_to}) does not match length {length}",
        )
        chunk_index = _integer(row.get("data/chunk_index"), context=f"{context} data/chunk_index", minimum=0)
        file_index = _integer(row.get("data/file_index"), context=f"{context} data/file_index", minimum=0)
        try:
            relative_data_path = data_template.format(chunk_index=chunk_index, file_index=file_index)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise DatasetValidationError(f"{context}: cannot format data_path {data_template!r}: {error}") from error
        _require(
            isinstance(relative_data_path, str) and relative_data_path.endswith(".parquet"),
            f"{context}: formatted data_path must name a parquet file, got {relative_data_path!r}",
        )
        data_path = _resolve_inside(root, relative_data_path, context=f"{context} data_path")
        episodes.append(
            _Episode(
                episode_index=episode_index,
                length=length,
                dataset_from_index=dataset_from,
                dataset_to_index=dataset_to,
                data_path=data_path,
            )
        )
        expected_from = dataset_to

    _require(
        expected_from == total_frames,
        f"{root}: episode intervals cover {expected_from} frames, but info.json declares {total_frames}",
    )
    return episodes


def _integer_vector(table: pa.Table, key: str, path: pathlib.Path) -> np.ndarray:
    column = table[key].combine_chunks()
    _require(column.null_count == 0, f"{path}: column {key!r} contains null values")
    _require(pa.types.is_integer(column.type), f"{path}: column {key!r} must be integer, got {column.type}")
    values = np.asarray(column.to_numpy(zero_copy_only=False))
    _require(values.ndim == 1 and values.shape[0] == table.num_rows, f"{path}: invalid shape for column {key!r}")
    return values.astype(np.int64, copy=False)


def _timestamp_vector(table: pa.Table, path: pathlib.Path) -> np.ndarray:
    column = table["timestamp"].combine_chunks()
    _require(column.null_count == 0, f"{path}: column 'timestamp' contains null values")
    _require(pa.types.is_floating(column.type), f"{path}: column 'timestamp' must be floating-point, got {column.type}")
    values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
    _require(values.ndim == 1 and values.shape[0] == table.num_rows, f"{path}: invalid timestamp shape")
    _require(np.isfinite(values).all(), f"{path}: timestamp contains NaN or infinity")
    return values


def _float32_matrix(table: pa.Table, key: str, dimension: int, path: pathlib.Path) -> np.ndarray:
    column = table[key].combine_chunks()
    _require(column.null_count == 0, f"{path}: column {key!r} contains null lists")
    if pa.types.is_fixed_size_list(column.type):
        _require(
            column.type.list_size == dimension,
            f"{path}: column {key!r} list size must be {dimension}, got {column.type.list_size}",
        )
        _require(
            pa.types.is_float32(column.type.value_type),
            f"{path}: column {key!r} values must be float32, got {column.type.value_type}",
        )
        _require(column.values.null_count == 0, f"{path}: column {key!r} contains null scalar values")
        values = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(
            table.num_rows, dimension
        )
    elif pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        _require(
            pa.types.is_float32(column.type.value_type),
            f"{path}: column {key!r} values must be float32, got {column.type.value_type}",
        )
        rows = column.to_pylist()
        bad_row = next((index for index, row in enumerate(rows) if row is None or len(row) != dimension), None)
        _require(
            bad_row is None,
            f"{path}: column {key!r} row {bad_row} does not have exactly {dimension} values",
        )
        values = np.asarray(rows, dtype=np.float32)
    else:
        raise DatasetValidationError(
            f"{path}: column {key!r} must be a float32 list of length {dimension}, got {column.type}"
        )
    _require(
        values.shape == (table.num_rows, dimension),
        f"{path}: column {key!r} has shape {values.shape}, expected {(table.num_rows, dimension)}",
    )
    _require(np.isfinite(values).all(), f"{path}: column {key!r} contains NaN or infinity")
    return np.ascontiguousarray(values)


def _repeat_padding_multiplicities(length: int) -> np.ndarray:
    """Return each action row's count in all length-by-ACTION_HORIZON chunks."""

    counts = np.minimum(np.arange(1, length + 1, dtype=np.int64), ACTION_HORIZON)
    counts[-1] = length * ACTION_HORIZON - int(counts[:-1].sum(dtype=np.int64))
    _require(np.all(counts > 0), f"Internal error: non-positive action multiplicity for episode length {length}")
    _require(
        int(counts.sum(dtype=np.int64)) == length * ACTION_HORIZON,
        f"Internal error: action multiplicities do not cover all chunks for episode length {length}",
    )
    return counts


def _load_task_samples(
    task: str,
    root: pathlib.Path,
    *,
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
) -> _TaskSamples:
    info, fps, total_episodes, total_frames = _validate_info(
        root,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    _validate_single_task_table(root)
    episodes = _read_episode_metadata(
        root,
        info,
        total_episodes=total_episodes,
        total_frames=total_frames,
    )

    episodes_by_path: dict[pathlib.Path, list[_Episode]] = defaultdict(list)
    for episode in episodes:
        episodes_by_path[episode.data_path].append(episode)
    referenced_paths = set(episodes_by_path)
    actual_paths = {path.resolve() for path in (root / "data").rglob("*.parquet")}
    missing_paths = sorted(str(path) for path in referenced_paths - actual_paths)
    extra_paths = sorted(str(path) for path in actual_paths - referenced_paths)
    _require(not missing_paths, f"{root}: metadata references missing data parquet files: {missing_paths}")
    _require(not extra_paths, f"{root}: found data parquet files not referenced by episode metadata: {extra_paths}")

    samples_by_episode: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for path, file_episodes in sorted(episodes_by_path.items()):
        try:
            parquet_file = pq.ParquetFile(path)
        except (OSError, pa.ArrowException) as error:
            raise DatasetValidationError(f"Cannot inspect data parquet {path}: {error}") from error
        missing_columns = sorted(set(DATA_COLUMNS) - set(parquet_file.schema_arrow.names))
        _require(not missing_columns, f"{path}: missing required data columns: {missing_columns}")
        try:
            table = parquet_file.read(columns=DATA_COLUMNS)
        except (OSError, pa.ArrowException) as error:
            raise DatasetValidationError(f"Cannot read required data columns from {path}: {error}") from error

        expected_rows = sum(episode.length for episode in file_episodes)
        _require(
            table.num_rows == expected_rows,
            f"{path}: contains {table.num_rows} rows, but mapped episode metadata requires {expected_rows}",
        )
        episode_indices = _integer_vector(table, "episode_index", path)
        frame_indices = _integer_vector(table, "frame_index", path)
        global_indices = _integer_vector(table, "index", path)
        task_indices = _integer_vector(table, "task_index", path)
        timestamps = _timestamp_vector(table, path)
        states = _float32_matrix(table, STATE_COLUMN, state_dim, path)
        actions = _float32_matrix(table, ACTION_COLUMN, action_dim, path)

        expected_episode_indices = {episode.episode_index for episode in file_episodes}
        actual_episode_indices = {int(value) for value in np.unique(episode_indices)}
        _require(
            actual_episode_indices == expected_episode_indices,
            f"{path}: episode_index values {sorted(actual_episode_indices)} do not match metadata "
            f"{sorted(expected_episode_indices)}",
        )
        _require(
            np.unique(global_indices).size == table.num_rows,
            f"{path}: global 'index' values must be unique",
        )

        for episode in file_episodes:
            positions = np.flatnonzero(episode_indices == episode.episode_index)
            _require(
                positions.size == episode.length,
                f"{path}: episode {episode.episode_index} has {positions.size} rows, metadata says {episode.length}",
            )
            expected_positions = np.arange(int(positions[0]), int(positions[0]) + episode.length)
            _require(
                np.array_equal(positions, expected_positions),
                f"{path}: rows for episode {episode.episode_index} are not contiguous",
            )
            episode_slice = slice(int(positions[0]), int(positions[0]) + episode.length)
            _require(
                np.array_equal(frame_indices[episode_slice], np.arange(episode.length)),
                f"{path}: episode {episode.episode_index} frame_index must be 0..{episode.length - 1}",
            )
            _require(
                np.array_equal(
                    global_indices[episode_slice],
                    np.arange(episode.dataset_from_index, episode.dataset_to_index),
                ),
                f"{path}: episode {episode.episode_index} global index interval does not match metadata",
            )
            _require(
                np.array_equal(task_indices[episode_slice], np.zeros(episode.length, dtype=np.int64)),
                f"{path}: episode {episode.episode_index} task_index must be zero in a single-task root",
            )
            expected_timestamps = np.arange(episode.length, dtype=np.float64) / fps
            _require(
                np.allclose(
                    timestamps[episode_slice],
                    expected_timestamps,
                    rtol=1e-4,
                    atol=1e-6,
                ),
                f"{path}: episode {episode.episode_index} timestamps are not row-aligned at {fps} Hz",
            )
            samples_by_episode[episode.episode_index] = (
                np.ascontiguousarray(states[episode_slice]),
                np.ascontiguousarray(actions[episode_slice]),
                _repeat_padding_multiplicities(episode.length),
            )

    _require(
        set(samples_by_episode) == set(range(total_episodes)),
        f"{root}: did not load every declared episode exactly once",
    )
    ordered_samples = [samples_by_episode[index] for index in range(total_episodes)]
    states = np.concatenate([sample[0] for sample in ordered_samples], axis=0)
    actions = np.concatenate([sample[1] for sample in ordered_samples], axis=0)
    action_multiplicities = np.concatenate([sample[2] for sample in ordered_samples], axis=0)
    _require(
        states.shape == (total_frames, state_dim) and actions.shape == (total_frames, action_dim),
        f"{root}: loaded sample dimensions do not match declared frame count",
    )
    _require(
        int(action_multiplicities.sum(dtype=np.int64)) == total_frames * ACTION_HORIZON,
        f"{root}: repeat-padding multiplicities do not represent every 30-step action chunk",
    )
    return _TaskSamples(
        task=task,
        states=states,
        actions=actions,
        action_multiplicities=action_multiplicities,
        episode_count=total_episodes,
    )


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> list[np.ndarray]:
    results = [np.empty(values.shape[1], dtype=np.float64) for _ in quantiles]
    target_weights = np.asarray(quantiles, dtype=np.float64) * weights.sum(dtype=np.float64)
    for dimension in range(values.shape[1]):
        order = np.argsort(values[:, dimension], kind="stable")
        sorted_values = values[order, dimension]
        cumulative_weights = np.cumsum(weights[order], dtype=np.float64)
        for output, target_weight in zip(results, target_weights, strict=True):
            index = int(np.searchsorted(cumulative_weights, target_weight, side="left"))
            output[dimension] = sorted_values[min(index, sorted_values.size - 1)]
    return results


def _task_balanced_stats(
    values_by_task: list[np.ndarray],
    multiplicities_by_task: list[np.ndarray],
) -> normalize.NormStats:
    _require(values_by_task, "Internal error: expected at least one task array")
    _require(
        len(values_by_task) == len(multiplicities_by_task),
        "Internal error: values and multiplicities must have the same task count",
    )
    task_count = len(values_by_task)
    dimension = values_by_task[0].shape[1]
    normalized_task_weights: list[np.ndarray] = []
    for task_index, (values, multiplicities) in enumerate(zip(values_by_task, multiplicities_by_task, strict=True)):
        _require(values.ndim == 2 and values.shape[1] == dimension, f"Task {task_index}: inconsistent value shape")
        _require(
            multiplicities.shape == (values.shape[0],),
            f"Task {task_index}: multiplicities do not align with values",
        )
        _require(np.isfinite(values).all(), f"Task {task_index}: values contain NaN or infinity")
        _require(np.all(multiplicities > 0), f"Task {task_index}: multiplicities must be positive")
        local_total = multiplicities.sum(dtype=np.float64)
        normalized_task_weights.append(multiplicities.astype(np.float64) / local_total / task_count)

    values = np.concatenate(values_by_task, axis=0)
    weights = np.concatenate(normalized_task_weights)
    weights /= weights.sum(dtype=np.float64)
    mean = np.einsum("n,nd->d", weights, values, dtype=np.float64)
    mean_of_squares = np.einsum("n,nd,nd->d", weights, values, values, dtype=np.float64)
    variance = np.maximum(0.0, mean_of_squares - np.square(mean))
    q01, q99 = _weighted_quantiles(values, weights, (0.01, 0.99))
    return normalize.NormStats(mean=mean, std=np.sqrt(variance), q01=q01, q99=q99)


def compute_balanced_stats(data_parent: pathlib.Path) -> tuple[dict[str, normalize.NormStats], list[_TaskSamples]]:
    """Validate all fixed task roots and compute their equal-mixture statistics."""

    data_parent = data_parent.expanduser().resolve()
    if not data_parent.is_dir():
        raise FileNotFoundError(f"--data-parent is not an existing directory: {data_parent}")
    task_samples = []
    for task in TASKS:
        root = _resolve_inside(data_parent, task, context=f"Task root {task!r}")
        if not root.is_dir():
            raise FileNotFoundError(f"Missing fixed DexJoCo task root: {root}")
        task_samples.append(_load_task_samples(task, root))

    state_values = [sample.states for sample in task_samples]
    state_multiplicities = [np.ones(sample.frame_count, dtype=np.int64) for sample in task_samples]
    action_values = [sample.actions for sample in task_samples]
    action_multiplicities = [sample.action_multiplicities for sample in task_samples]
    return {
        "state": _task_balanced_stats(state_values, state_multiplicities),
        "actions": _task_balanced_stats(action_values, action_multiplicities),
    }, task_samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute equal-task OpenPI state/action norm stats from the six fixed DexJoCo LeRobot v3 roots. "
            "Only parquet and metadata are read; video is never decoded."
        )
    )
    parser.add_argument(
        "--data-parent",
        required=True,
        type=pathlib.Path,
        help="Parent containing click_mouse, fold_glasses, hammer_nail, pick_bucket, pinch_tongs, and water_plant.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Directory in which OpenPI-compatible norm_stats.json will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    norm_stats, task_samples = compute_balanced_stats(args.data_parent)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"--output-dir exists but is not a directory: {output_dir}")
    normalize.save(output_dir, norm_stats)
    for sample in task_samples:
        print(f"validated {sample.task}: episodes={sample.episode_count}, frames={sample.frame_count}, task_weight=1/6")
    print(f"wrote OpenPI task-balanced normalization stats to {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
