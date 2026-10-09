"""Helpers for recording structured sensor observations as Zarr tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

SENSOR_OBSERVATION_GROUPS = ("tactile_gt",)
TACTILE_GT_SCHEMA_VERSION = "dexjoco.tactile_gt.v2"
_TAXELS_PER_FINGERTIP = 16
_NON_RGB_SENSOR_ROOTS = frozenset(
    (
        *SENSOR_OBSERVATION_GROUPS,
        *(f"next_{name}" for name in SENSOR_OBSERVATION_GROUPS),
    )
)


def is_rgb_observation(key: str, value) -> bool:
    """Return whether a top-level observation is an RGB image tensor.

    The semantic key guard prevents structured force arrays ending in an xyz
    dimension from being mistaken for images.
    """

    if not isinstance(key, str):
        return False
    root_key = key.split(".", 1)[0]
    if root_key in _NON_RGB_SENSOR_ROOTS:
        return False
    if not isinstance(value, np.ndarray):
        return False
    if value.dtype != np.uint8 or value.shape[-1:] != (3,):
        return False
    return value.ndim == 3 or (value.ndim == 4 and value.shape[0] == 1)


def flatten_sensor_observation_groups(
    observation: Mapping,
    groups: Sequence[str] = SENSOR_OBSERVATION_GROUPS,
) -> dict[str, np.ndarray]:
    """Flatten configured nested sensor groups into dotted numeric fields.

    For example, ``{"tactile_gt": {"normal": array}}`` becomes
    ``{"tactile_gt.normal": array}``.  Dotted names remain single Zarr dataset
    names and avoid the accidental subgroup semantics of ``/``.
    """

    if "tactile" in observation:
        raise ValueError(
            "The retired 'tactile' observation is no longer "
            "supported; record MuJoCo 'tactile_gt' only."
        )

    flattened: dict[str, np.ndarray] = {}
    for group_name in groups:
        _validate_field_segment(group_name, "sensor group")
        if group_name not in observation:
            continue
        group = observation[group_name]
        if not isinstance(group, Mapping):
            raise TypeError(
                f"Observation group {group_name!r} must be a mapping, "
                f"got {type(group).__name__}."
            )
        if not group:
            raise ValueError(
                f"Observation group {group_name!r} must not be empty."
            )
        _flatten_numeric_mapping(group, group_name, flattened)
    return flattened


def stack_sensor_observation_groups(
    observations: Sequence[Mapping],
    *,
    dataset_prefix: str = "",
    groups: Sequence[str] = SENSOR_OBSERVATION_GROUPS,
) -> dict[str, np.ndarray]:
    """Stack sensor fields across time, rejecting partial or changing schemas."""

    if not observations:
        return {}

    frames = [flatten_sensor_observation_groups(obs, groups) for obs in observations]
    expected_keys = set(frames[0])
    for frame_index, frame in enumerate(frames[1:], start=1):
        frame_keys = set(frame)
        if frame_keys != expected_keys:
            missing = sorted(expected_keys - frame_keys)
            extra = sorted(frame_keys - expected_keys)
            raise ValueError(
                "Sensor observation schema changed at frame "
                f"{frame_index}: missing={missing}, extra={extra}."
            )

    stacked: dict[str, np.ndarray] = {}
    for field_name in sorted(expected_keys):
        values = [frame[field_name] for frame in frames]
        expected_shape = values[0].shape
        expected_dtype = values[0].dtype
        for frame_index, value in enumerate(values[1:], start=1):
            if value.shape != expected_shape:
                raise ValueError(
                    f"Sensor observation field {field_name!r} changed shape at "
                    f"frame {frame_index}: {expected_shape} versus {value.shape}."
                )
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"Sensor observation field {field_name!r} changed dtype at "
                    f"frame {frame_index}: {expected_dtype} versus {value.dtype}."
                )
        try:
            array = np.stack(values, axis=0)
        except ValueError as exc:
            shapes = [value.shape for value in values]
            raise ValueError(
                f"Sensor observation field {field_name!r} changed shape: {shapes}."
            ) from exc
        stacked[f"{dataset_prefix}{field_name}"] = np.ascontiguousarray(array)
    return stacked


def stack_sensor_transition_groups(
    observations: Sequence[Mapping],
    next_observations: Sequence[Mapping],
    *,
    groups: Sequence[str] = SENSOR_OBSERVATION_GROUPS,
) -> dict[str, np.ndarray]:
    """Stack aligned pre-action and post-action sensor observations.

    The two sides must have identical field sets, per-frame shapes, and dtypes.
    This prevents a recorder from silently writing an incomplete terminal
    response or a pre/next schema that downstream code cannot pair by row.
    """

    if len(observations) != len(next_observations):
        raise ValueError(
            "Pre-action and post-action sensor sequences must have the same "
            f"length, got {len(observations)} and {len(next_observations)}."
        )

    current = stack_sensor_observation_groups(observations, groups=groups)
    following = stack_sensor_observation_groups(
        next_observations,
        dataset_prefix="next_",
        groups=groups,
    )

    current_fields = set(current)
    following_fields = {
        name[len("next_") :]
        for name in following
        if name.startswith("next_")
    }
    if current_fields != following_fields:
        missing_next = sorted(current_fields - following_fields)
        extra_next = sorted(following_fields - current_fields)
        raise ValueError(
            "Pre-action and post-action sensor schemas differ: "
            f"missing_next={missing_next}, extra_next={extra_next}."
        )

    for field_name in sorted(current_fields):
        current_array = current[field_name]
        following_array = following[f"next_{field_name}"]
        if current_array.shape != following_array.shape:
            raise ValueError(
                f"Pre/next sensor field {field_name!r} shape differs: "
                f"{current_array.shape} versus {following_array.shape}."
            )
        if current_array.dtype != following_array.dtype:
            raise ValueError(
                f"Pre/next sensor field {field_name!r} dtype differs: "
                f"{current_array.dtype} versus {following_array.dtype}."
            )

    return {**current, **following}


def derive_episode_timestamps(
    observations: Sequence[Mapping],
    *,
    data_fps: float = 0.0,
    control_dt: float | None = None,
) -> np.ndarray:
    """Build a relative pre-action time axis without hiding clock mismatches.

    ``tactile_gt.sim_time`` is authoritative when present on every frame.  A
    positive ``data_fps`` is treated as an assertion and must agree with that
    clock (or with ``control_dt``).  With ``data_fps == 0``, the function
    infers timestamps from ground truth first and then from ``control_dt``.
    """

    frame_count = len(observations)
    if frame_count == 0:
        return np.zeros((0,), dtype=np.float64)
    data_fps = float(data_fps)
    if not np.isfinite(data_fps) or data_fps < 0.0:
        raise ValueError(f"data_fps must be finite and non-negative, got {data_fps}.")
    if control_dt is not None:
        control_dt = float(control_dt)
        if not np.isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError(
                f"control_dt must be finite and positive, got {control_dt}."
            )

    sim_times: list[float] = []
    has_sim_time: list[bool] = []
    for frame_index, observation in enumerate(observations):
        tactile_gt = observation.get("tactile_gt")
        present = isinstance(tactile_gt, Mapping) and "sim_time" in tactile_gt
        has_sim_time.append(present)
        if not present:
            continue
        value = np.asarray(tactile_gt["sim_time"])
        if value.shape != (1,) or value.dtype != np.float64:
            raise ValueError(
                "tactile_gt.sim_time must have per-frame shape (1,) and "
                f"dtype float64 at frame {frame_index}, got {value.shape} "
                f"and {value.dtype}."
            )
        sim_times.append(float(value[0]))

    if any(has_sim_time) and not all(has_sim_time):
        raise ValueError("tactile_gt.sim_time is missing from only part of the episode.")

    authoritative = None
    if all(has_sim_time):
        authoritative = np.asarray(sim_times, dtype=np.float64)
        if not np.all(np.isfinite(authoritative)):
            raise ValueError("tactile_gt.sim_time contains non-finite values.")
        if frame_count > 1 and np.any(np.diff(authoritative) <= 0.0):
            raise ValueError("tactile_gt.sim_time must be strictly increasing by frame.")
        authoritative = authoritative - authoritative[0]

    asserted_dt = 1.0 / data_fps if data_fps > 0.0 else None
    if asserted_dt is not None and control_dt is not None and not np.isclose(
        asserted_dt, control_dt, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            f"data_fps={data_fps:g} implies dt={asserted_dt:g}s, which does "
            f"not match environment control_dt={control_dt:g}s."
        )
    if authoritative is not None:
        if asserted_dt is not None and frame_count > 1 and not np.allclose(
            np.diff(authoritative), asserted_dt, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                f"data_fps={data_fps:g} does not match tactile_gt.sim_time."
            )
        if control_dt is not None and frame_count > 1 and not np.allclose(
            np.diff(authoritative), control_dt, rtol=0.0, atol=1e-9
        ):
            raise ValueError("control_dt does not match tactile_gt.sim_time.")
        return authoritative

    dt = asserted_dt if asserted_dt is not None else control_dt
    if dt is None:
        raise ValueError(
            "Cannot infer timestamps: provide control_dt, set data_fps > 0, "
            "or record tactile_gt.sim_time."
        )
    return np.arange(frame_count, dtype=np.float64) * dt


def build_sensor_metadata(
    episode_data: Mapping[str, np.ndarray],
) -> dict[str, dict]:
    """Build self-describing metadata for tactile fields in an episode."""

    retired_keys = sorted(
        name
        for name in episode_data
        if name.startswith("tactile.") or name.startswith("next_tactile.")
    )
    if retired_keys:
        raise ValueError(
            "Retired tactile fields are no longer supported: "
            f"{retired_keys}."
        )

    metadata: dict[str, dict] = {}
    wrench_key = "tactile_gt.fingertip_wrench_local"
    tactile_gt_keys = {
        name for name in episode_data if name.startswith("tactile_gt.")
    }
    if tactile_gt_keys and wrench_key not in tactile_gt_keys:
        raise ValueError(
            f"Tactile ground-truth data requires field {wrench_key!r}."
        )
    if tactile_gt_keys:
        wrench = np.asarray(episode_data[wrench_key])
        if wrench.ndim != 4 or wrench.shape[-1] != 6:
            raise ValueError(
                f"{wrench_key!r} must have shape (time, hands, fingertips, 6), "
                f"got {wrench.shape}."
            )

        hand_count = int(wrench.shape[1])
        fingertip_count = int(wrench.shape[2])
        time_count = int(wrench.shape[0])
        expected_fields = {
            wrench_key: ((time_count, hand_count, fingertip_count, 6), np.float32),
            "tactile_gt.taxel_force_local": (
                (
                    time_count,
                    hand_count,
                    fingertip_count,
                    _TAXELS_PER_FINGERTIP,
                    3,
                ),
                np.float32,
            ),
            "tactile_gt.fingertip_normal": (
                (time_count, hand_count, fingertip_count),
                np.float32,
            ),
            "tactile_gt.fingertip_force_peak": (
                (time_count, hand_count, fingertip_count),
                np.float32,
            ),
            "tactile_gt.fingertip_impulse_local": (
                (time_count, hand_count, fingertip_count, 3),
                np.float32,
            ),
            "tactile_gt.contact_fraction": (
                (time_count, hand_count, fingertip_count),
                np.float32,
            ),
            "tactile_gt.sim_time": ((time_count, 1), np.float64),
        }
        _validate_complete_field_set(
            episode_data,
            tactile_gt_keys,
            expected_fields,
            group_label="Tactile ground-truth v2",
        )

        hand_order = _hand_order(hand_count)
        fingertip_order = _fingertip_order(fingertip_count)
        metadata["tactile_gt"] = {
            "schema_version": TACTILE_GT_SCHEMA_VERSION,
            "frame": "per-sample fingertip body-local frame at the body origin",
            "hand_order": hand_order,
            "fingertip_order": fingertip_order,
            "wrench_channels": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "taxel_channels": ["tangent_u", "tangent_v", "normal_inward"],
            "taxels_per_fingertip": _TAXELS_PER_FINGERTIP,
            "taxel_grid_shape": [4, 4],
            "taxel_frame": "per-taxel local frame",
            "taxel_projection": {
                "method": "deterministic normalized Gaussian projection",
                "source": "MuJoCo contact positions and local contact forces",
                "normalization": "per-contact taxel weights sum to one",
                "sigma": (
                    "per-fingertip median nearest-neighbor taxel-center "
                    "spacing when unspecified"
                ),
                "support": "all 16 taxels on the fingertip",
                "layout": "4 axial rows x 4 circumferential columns",
                "layout_source": (
                    "taxel centers and local frames are deterministically "
                    "derived from each fingertip collision capsule, including "
                    "the MuJoCo geom pose"
                ),
                "aggregation": (
                    "sum projected contact events per physics substep, then "
                    "arithmetic mean over control-step substeps"
                ),
                "conservation": (
                    "projected force sums to the source force after each "
                    "taxel vector is rotated back to the fingertip body frame"
                ),
            },
            "field_units": {
                "fingertip_wrench_local": [
                    "N",
                    "N",
                    "N",
                    "N*m",
                    "N*m",
                    "N*m",
                ],
                "taxel_force_local": ["N", "N", "N"],
                "fingertip_normal": "N",
                "fingertip_force_peak": "N",
                "fingertip_impulse_local": "N*s",
                "contact_fraction": "1",
                "sim_time": "s",
            },
            "reduction": {
                "fingertip_wrench_local": "mean over all physics substeps",
                "taxel_force_local": (
                    "mean over all physics substeps after deterministic "
                    "Gaussian projection"
                ),
                "fingertip_normal": "mean summed normal load",
                "fingertip_force_peak": "max summed normal load in one substep",
                "fingertip_impulse_local": "sum of local force times physics_dt",
                "contact_fraction": "fraction of substeps above contact threshold",
            },
            "alignment": {
                "tactile_gt.*": "pre-action observation at row t",
                "next_tactile_gt.*": "post-action response to action at row t",
            },
            "field_specs": {
                "fingertip_wrench_local": {
                    "axes": ["time", "hand", "fingertip", "wrench_channel"],
                    "dtype": "float32",
                },
                "taxel_force_local": {
                    "axes": [
                        "time",
                        "hand",
                        "fingertip",
                        "taxel",
                        "xyz",
                    ],
                    "dtype": "float32",
                },
                "fingertip_normal": {
                    "axes": ["time", "hand", "fingertip"],
                    "dtype": "float32",
                },
                "fingertip_force_peak": {
                    "axes": ["time", "hand", "fingertip"],
                    "dtype": "float32",
                },
                "fingertip_impulse_local": {
                    "axes": ["time", "hand", "fingertip", "xyz"],
                    "dtype": "float32",
                },
                "contact_fraction": {
                    "axes": ["time", "hand", "fingertip"],
                    "dtype": "float32",
                },
                "sim_time": {"axes": ["time", "singleton"], "dtype": "float64"},
            },
        }

    return metadata


def _validate_complete_field_set(
    episode_data: Mapping[str, np.ndarray],
    actual_keys: set[str],
    expected_fields: Mapping[str, tuple[tuple[int, ...], type]],
    *,
    group_label: str,
) -> None:
    missing_fields = sorted(set(expected_fields) - actual_keys)
    extra_fields = sorted(actual_keys - set(expected_fields))
    if missing_fields or extra_fields:
        raise ValueError(
            f"{group_label} fields differ from the schema: "
            f"missing={missing_fields}, extra={extra_fields}."
        )
    for field_name, (expected_shape, expected_dtype) in expected_fields.items():
        array = np.asarray(episode_data[field_name])
        if array.shape != expected_shape:
            raise ValueError(
                f"{field_name!r} must have shape {expected_shape}, got {array.shape}."
            )
        if array.dtype != expected_dtype:
            raise ValueError(
                f"{field_name!r} must have dtype {np.dtype(expected_dtype)}, "
                f"got {array.dtype}."
            )


def _hand_order(hand_count: int) -> list[str]:
    if hand_count == 1:
        return ["right"]
    if hand_count == 2:
        return ["right", "left"]
    return [f"hand_{index}" for index in range(hand_count)]


def _fingertip_order(fingertip_count: int) -> list[str]:
    if fingertip_count == 4:
        return ["index", "middle", "ring", "thumb"]
    return [f"fingertip_{index}" for index in range(fingertip_count)]


def _flatten_numeric_mapping(
    mapping: Mapping,
    prefix: str,
    output: dict[str, np.ndarray],
) -> None:
    for key, value in mapping.items():
        _validate_field_segment(key, "sensor field")
        field_name = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            if not value:
                raise ValueError(
                    f"Nested sensor mapping {field_name!r} must not be empty."
                )
            _flatten_numeric_mapping(value, field_name, output)
            continue

        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError(
                f"Sensor field {field_name!r} must have a non-scalar per-frame shape."
            )
        is_supported_dtype = (
            np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        )
        if array.dtype == object or not is_supported_dtype:
            raise TypeError(
                f"Sensor field {field_name!r} must be a fixed-shape real numeric "
                "or boolean array, "
                f"got dtype {array.dtype}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"Sensor field {field_name!r} contains NaN or infinite values."
            )
        if field_name in output:
            raise ValueError(f"Duplicate flattened sensor field {field_name!r}.")
        output[field_name] = np.ascontiguousarray(array)


def _validate_field_segment(value, kind: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{kind.capitalize()} names must be non-empty strings, got {value!r}."
        )
    if "." in value or "/" in value:
        raise ValueError(
            f"{kind.capitalize()} name {value!r} cannot contain '.' or '/'."
        )
