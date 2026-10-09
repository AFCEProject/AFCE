"""Compute equal-task normalization statistics for selected DexJoCo tasks.

The released task roots have two schemas: six single-arm datasets use 23-D
state and 22-D action, while five bimanual datasets use 46-D state and 44-D
action. By default this script reproduces official-style right-padding. The
structured-hand option instead places single-arm palm and joint values in the
same semantic slots as bimanual state. Both modes avoid video decoding/copying
and give every selected task exactly 1/N of the normalization weight.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import compute_balanced_lerobot_norm_stats as _base
import numpy as np

from openpi.shared import normalize

TARGET_STATE_DIM = 46
TARGET_ACTION_DIM = 44

TASK_SPECS = (
    ("bimanual_assembly", 46, 44),
    ("bimanual_hanoi", 46, 44),
    ("bimanual_microwave_cook", 46, 44),
    ("bimanual_photograph", 46, 44),
    ("bimanual_unlock_ipad", 46, 44),
    ("click_mouse", 23, 22),
    ("fold_glasses", 23, 22),
    ("hammer_nail", 23, 22),
    ("pick_bucket", 23, 22),
    ("pinch_tongs", 23, 22),
    ("water_plant", 23, 22),
)


def _right_pad(values: np.ndarray, target_dim: int) -> np.ndarray:
    if values.shape[1] > target_dim:
        raise _base.DatasetValidationError(
            f"Cannot right-pad shape {values.shape}: source dimension exceeds target {target_dim}"
        )
    if values.shape[1] == target_dim:
        return values
    return np.pad(values, ((0, 0), (0, target_dim - values.shape[1])), mode="constant")


def _canonicalize_structured_hand_state(values: np.ndarray) -> np.ndarray:
    """Map [right_tcp7, right_joints16] to the shared 46-D bimanual layout."""
    if values.shape[1] == TARGET_STATE_DIM:
        return values
    if values.shape[1] != 23:
        raise _base.DatasetValidationError(
            f"Structured hand state expects 23-D or 46-D values, got {values.shape}"
        )
    return np.concatenate(
        [
            values[:, :7],
            np.zeros((values.shape[0], 7), dtype=values.dtype),
            values[:, 7:23],
            np.zeros((values.shape[0], 16), dtype=values.dtype),
        ],
        axis=1,
    )


TASK_SPEC_MAP = {task: (state_dim, action_dim) for task, state_dim, action_dim in TASK_SPECS}


def compute_balanced_stats(
    data_parent: pathlib.Path,
    *,
    structured_hand_state: bool = False,
    tasks: tuple[str, ...] | None = None,
) -> tuple[dict[str, normalize.NormStats], list[_base._TaskSamples]]:
    """Validate selected DexJoCo roots and return equal-mixture (1/N) statistics."""

    data_parent = data_parent.expanduser().resolve()
    if not data_parent.is_dir():
        raise FileNotFoundError(f"--data-parent is not an existing directory: {data_parent}")

    if tasks is None:
        selected = TASK_SPECS
    else:
        unknown = [t for t in tasks if t not in TASK_SPEC_MAP]
        if unknown:
            raise ValueError(f"Unknown DexJoCo tasks: {unknown}. Known: {sorted(TASK_SPEC_MAP)}")
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"--tasks must be unique, got {tasks}")
        selected = tuple((t, *TASK_SPEC_MAP[t]) for t in tasks)

    n_tasks = len(selected)
    task_samples: list[_base._TaskSamples] = []
    for task, state_dim, action_dim in selected:
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
                    _canonicalize_structured_hand_state(sample.states)
                    if structured_hand_state
                    else _right_pad(sample.states, TARGET_STATE_DIM)
                ),
                actions=np.ascontiguousarray(_right_pad(sample.actions, TARGET_ACTION_DIM)),
            )
        )

    state_values = [sample.states for sample in task_samples]
    state_multiplicities = [np.ones(sample.frame_count, dtype=np.int64) for sample in task_samples]
    action_values = [sample.actions for sample in task_samples]
    action_multiplicities = [sample.action_multiplicities for sample in task_samples]
    return {
        "state": _base._task_balanced_stats(state_values, state_multiplicities),  # noqa: SLF001
        "actions": _base._task_balanced_stats(action_values, action_multiplicities),  # noqa: SLF001
    }, task_samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 1/N task-balanced OpenPI norm stats for selected DexJoCo roots."
    )
    parser.add_argument("--data-parent", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        default=None,
        help="Task id. Repeatable. Default: all 11 DexJoCo tasks.",
    )
    parser.add_argument(
        "--structured-hand-state",
        action="store_true",
        help="Use [right_tcp, left_tcp, right_joints, left_joints] semantic state slots.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = tuple(args.tasks) if args.tasks else None
    norm_stats, task_samples = compute_balanced_stats(
        args.data_parent,
        structured_hand_state=args.structured_hand_state,
        tasks=tasks,
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"--output-dir exists but is not a directory: {output_dir}")
    normalize.save(output_dir, norm_stats)
    n = len(task_samples)
    for sample in task_samples:
        print(
            f"validated {sample.task}: episodes={sample.episode_count}, "
            f"frames={sample.frame_count}, task_weight=1/{n}"
        )
    print(f"wrote OpenPI task-balanced normalization stats to {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
