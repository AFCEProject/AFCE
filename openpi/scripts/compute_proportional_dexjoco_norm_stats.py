"""Compute official-style frame-proportional statistics for all DexJoCo tasks.

DexJoCo's released multi-task converter appends every retained frame from the
eleven task datasets to one LeRobot dataset.  Sampling that merged dataset
uniformly therefore weights each task in proportion to its retained frame
count.  This module reproduces the same weighting without copying or decoding
the videos.
"""

from __future__ import annotations

import dataclasses
import pathlib

import compute_balanced_dexjoco_norm_stats as _dexjoco
import compute_balanced_lerobot_norm_stats as _base
import numpy as np

from openpi.shared import normalize


def _frame_proportional_stats(
    values_by_task: list[np.ndarray],
    multiplicities_by_task: list[np.ndarray],
) -> normalize.NormStats:
    """Return statistics for the concatenation of all task frames."""

    values = np.concatenate(values_by_task, axis=0)
    multiplicities = np.concatenate(multiplicities_by_task, axis=0)
    # A one-element task list makes the validated weighted-statistics helper
    # preserve every row's raw multiplicity instead of normalizing each task to
    # an equal 1/11 share.
    return _base._task_balanced_stats([values], [multiplicities])  # noqa: SLF001


def compute_proportional_stats(
    data_parent: pathlib.Path,
    *,
    structured_hand_state: bool = False,
) -> tuple[dict[str, normalize.NormStats], list[_base._TaskSamples]]:
    """Validate all eleven roots and return merged-frame statistics."""

    data_parent = data_parent.expanduser().resolve()
    if not data_parent.is_dir():
        raise FileNotFoundError(f"--data-parent is not an existing directory: {data_parent}")

    task_samples: list[_base._TaskSamples] = []
    for task, state_dim, action_dim in _dexjoco.TASK_SPECS:
        root = _base._resolve_inside(data_parent, task, context=f"Task root {task!r}")  # noqa: SLF001
        if not root.is_dir():
            raise FileNotFoundError(f"Missing fixed DexJoCo task root: {root}")
        sample = _base._load_task_samples(  # noqa: SLF001
            task,
            root,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        task_samples.append(
            dataclasses.replace(
                sample,
                states=np.ascontiguousarray(
                    _dexjoco._canonicalize_structured_hand_state(sample.states)  # noqa: SLF001
                    if structured_hand_state
                    else _dexjoco._right_pad(sample.states, _dexjoco.TARGET_STATE_DIM)  # noqa: SLF001
                ),
                actions=np.ascontiguousarray(
                    _dexjoco._right_pad(sample.actions, _dexjoco.TARGET_ACTION_DIM)  # noqa: SLF001
                ),
            )
        )

    state_values = [sample.states for sample in task_samples]
    state_multiplicities = [np.ones(sample.frame_count, dtype=np.int64) for sample in task_samples]
    action_values = [sample.actions for sample in task_samples]
    action_multiplicities = [sample.action_multiplicities for sample in task_samples]
    return {
        "state": _frame_proportional_stats(state_values, state_multiplicities),
        "actions": _frame_proportional_stats(action_values, action_multiplicities),
    }, task_samples

