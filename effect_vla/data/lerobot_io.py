"""LeRobot DexJoCo I/O helpers (videos + low-dim state, no policy training)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from effect_vla.constants import (
    BIMANUAL_EFFECT_CAMERA,
    BIMANUAL_TASKS,
    CLICK_MOUSE_EFFECT_CAMERA,
    SINGLE_ARM_EFFECT_CAMERA,
)


def effect_camera_key(task: str) -> str:
    if task == "click_mouse":
        return CLICK_MOUSE_EFFECT_CAMERA
    if task in BIMANUAL_TASKS:
        return BIMANUAL_EFFECT_CAMERA
    return SINGLE_ARM_EFFECT_CAMERA


def is_bimanual(task: str) -> bool:
    return task in BIMANUAL_TASKS


def load_info(task_root: Path) -> dict:
    return json.loads((task_root / "meta" / "info.json").read_text())


def iter_episode_rows(task_root: Path) -> list[dict]:
    path = task_root / "meta" / "episodes.jsonl"
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stack_state_column(values) -> np.ndarray:
    first = values.iloc[0] if hasattr(values, "iloc") else values[0]
    if isinstance(first, np.ndarray):
        return np.stack([np.asarray(v, dtype=np.float32) for v in values], axis=0)
    return np.asarray(list(values), dtype=np.float32)


def load_episode_parquet(task_root: Path, episode_index: int) -> dict[str, np.ndarray]:
    """Load low-dim columns for one episode from the LeRobot v3 parquet layout."""
    import pandas as pd

    parquet_files = sorted((task_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {task_root / 'data'}")
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["observation.state", "frame_index", "episode_index"])
        sub = df[df["episode_index"] == episode_index]
        if sub.empty:
            continue
        state = _stack_state_column(sub["observation.state"])
        return {
            "state": state,
            "frame_index": np.asarray(sub["frame_index"].to_numpy(), dtype=np.int64),
            "length": int(len(state)),
        }
    raise KeyError(f"Episode {episode_index} not found in {task_root}")


def read_video_frames(video_path: Path) -> np.ndarray:
    """Return (T, H, W, 3) uint8. Tries torchvision, then imageio, then cv2."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    try:
        import torchvision.io as tio

        frames, _, _ = tio.read_video(str(video_path), output_format="THWC", pts_unit="sec")
        return frames.numpy().astype(np.uint8)
    except Exception:
        pass
    try:
        import imageio.v3 as iio

        frames = iio.imread(video_path)
        if frames.ndim != 4:
            raise ValueError(frames.shape)
        return np.asarray(frames, dtype=np.uint8)
    except Exception:
        pass
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise RuntimeError(f"Failed to read video {video_path}")
    return np.stack(out, axis=0)


def episode_front_video_path(task_root: Path, task: str, episode_index: int) -> Path:
    """Locate the static/front video for one episode."""
    key = effect_camera_key(task)
    # LeRobot v3: videos/{video_key}/chunk-XXX/file-YYY.mp4
    video_root = task_root / "videos" / key
    if not video_root.exists():
        # Some dumps omit the "observation.images." prefix in the folder name.
        short = key.split(".")[-1]
        alt = task_root / "videos" / short
        video_root = alt if alt.exists() else video_root
    files = sorted(video_root.rglob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No videos under {video_root}")
    # DexJoCo currently stores one concatenated file per camera chunk. Decode
    # the whole file and slice by episode using parquet lengths if needed.
    if len(files) == 1:
        return files[0]
    # Prefer a file whose name contains the episode index.
    tagged = [p for p in files if f"{episode_index:06d}" in p.name or f"{episode_index:03d}" in p.name]
    if tagged:
        return tagged[0]
    # Fall back to file index == episode index when 1:1.
    if episode_index < len(files):
        return files[episode_index]
    return files[0]


def slice_episode_frames(all_frames: np.ndarray, episode_index: int, episodes: list[dict]) -> np.ndarray:
    row = next(r for r in episodes if int(r.get("episode_index", r.get("index", -1))) == episode_index)
    length = int(row.get("length", row.get("num_frames", -1)))
    start = int(row.get("dataset_from_index", 0))
    # If the video is already per-episode, just return it.
    if length > 0 and all_frames.shape[0] == length:
        return all_frames
    if length > 0 and start >= 0 and start + length <= all_frames.shape[0]:
        return all_frames[start : start + length]
    # Cumulative lengths as last resort.
    cursor = 0
    for r in sorted(episodes, key=lambda x: int(x.get("episode_index", x.get("index", 0)))):
        n = int(r.get("length", r.get("num_frames", 0)))
        idx = int(r.get("episode_index", r.get("index", -1)))
        if idx == episode_index:
            return all_frames[cursor : cursor + n]
        cursor += n
    return all_frames
