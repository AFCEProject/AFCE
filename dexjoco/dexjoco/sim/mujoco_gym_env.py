from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np


@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 640
    width: int = 640
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        render_spec: GymRenderingSpec = GymRenderingSpec(),
    ):
        control_dt = float(control_dt)
        physics_dt = float(physics_dt)
        if not np.isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError(
                f"control_dt must be finite and positive, got {control_dt!r}."
            )
        if not np.isfinite(physics_dt) or physics_dt <= 0.0:
            raise ValueError(
                f"physics_dt must be finite and positive, got {physics_dt!r}."
            )

        substep_ratio = control_dt / physics_dt
        n_substeps = int(round(substep_ratio))
        if n_substeps < 1 or not np.isclose(
            substep_ratio,
            n_substeps,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "control_dt must be an integer multiple of physics_dt; "
                f"got control_dt={control_dt!r}, physics_dt={physics_dt!r}, "
                f"ratio={substep_ratio!r}."
            )

        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = n_substeps
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def time_limit_exceeded(self) -> bool:
        return self._data.time >= self._time_limit

    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random
