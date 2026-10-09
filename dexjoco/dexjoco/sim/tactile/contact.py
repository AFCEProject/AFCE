"""Ground-truth fingertip contact wrench extraction from MuJoCo."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ContactEvent:
    """One force-bearing contact expressed in a fingertip body frame.

    ``force_local`` and ``torque_local_at_contact`` are the wrench applied to
    the fingertip at ``position_local``.  The latter is MuJoCo's contact-point
    torque (for example torsional friction); the moment about the fingertip
    body origin is therefore ``torque + cross(position, force)``.

    Arrays returned by :meth:`FingertipContactExtractor.current_contacts` are
    defensive copies by default, so downstream callers cannot mutate extractor
    state.  The explicit ``copy=False`` fast path is reserved for immediate,
    read-only consumption inside the environment loop.
    """

    fingertip_index: int
    position_local: np.ndarray
    force_local: np.ndarray
    torque_local_at_contact: np.ndarray
    normal_force: float


class FingertipContactExtractor:
    """Aggregate MuJoCo contacts into local fingertip wrench observations.

    A control interval is explicitly delimited with :meth:`begin`, one sample is
    accumulated immediately after every ``mj_step`` with
    :meth:`accumulate_substep`, and :meth:`finish` publishes the interval
    statistics.  Forces are expressed in each fingertip body's local frame at
    that body's origin.

    MuJoCo's ``mj_contactForce`` returns the wrench applied to ``geom2`` in the
    contact frame.  The equal-and-opposite wrench is applied to ``geom1``.  The
    extractor handles that sign convention before transforming the wrench into
    each fingertip frame.

    Args:
        model: Compiled MuJoCo model.
        data: MuJoCo data associated with ``model``.
        fingertip_body_names: Stable fingertip body names in output order.
        physics_dt: Duration represented by each accumulated sample.  Defaults
            to ``model.opt.timestep``.
        contact_threshold: Minimum summed normal load counted as an active
            contact substep when computing ``contact_fraction``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        fingertip_body_names: Sequence[str],
        physics_dt: float | None = None,
        contact_threshold: float = 1e-9,
    ) -> None:
        if not fingertip_body_names:
            raise ValueError("At least one fingertip body name is required.")
        if len(set(fingertip_body_names)) != len(fingertip_body_names):
            raise ValueError("Fingertip body names must be unique.")

        self.model = model
        self.data = data
        self.fingertip_body_names = tuple(fingertip_body_names)
        self.physics_dt = float(
            model.opt.timestep if physics_dt is None else physics_dt
        )
        if not np.isfinite(self.physics_dt) or self.physics_dt <= 0:
            raise ValueError(f"physics_dt must be positive, got {self.physics_dt!r}.")
        self.contact_threshold = float(contact_threshold)
        if not np.isfinite(self.contact_threshold) or self.contact_threshold < 0:
            raise ValueError(
                "contact_threshold must be finite and non-negative, got "
                f"{self.contact_threshold!r}."
            )

        body_ids = []
        for name in self.fingertip_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"Fingertip body {name!r} was not found in the model.")
            body_ids.append(body_id)
        self.body_ids = np.asarray(body_ids, dtype=np.int32)

        explicitly_paired_geoms = np.zeros(model.ngeom, dtype=bool)
        if model.npair:
            explicitly_paired_geoms[model.pair_geom1] = True
            explicitly_paired_geoms[model.pair_geom2] = True
        collision_enabled = (
            (model.geom_contype != 0)
            | (model.geom_conaffinity != 0)
            | explicitly_paired_geoms
        )

        self._geom_to_fingertip = np.full(model.ngeom, -1, dtype=np.int32)
        for fingertip_index, body_id in enumerate(self.body_ids):
            geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
            if geom_ids.size == 0:
                raise ValueError(
                    f"Fingertip body {self.fingertip_body_names[fingertip_index]!r} "
                    "has no attached geoms."
                )
            contact_geom_ids = geom_ids[collision_enabled[geom_ids]]
            if contact_geom_ids.size == 0:
                raise ValueError(
                    f"Fingertip body {self.fingertip_body_names[fingertip_index]!r} "
                    "has no collision-enabled or explicitly paired geoms."
                )
            self._geom_to_fingertip[contact_geom_ids] = fingertip_index

        self._contact_wrench = np.zeros(6, dtype=np.float64)
        self._sample_wrench = np.zeros((self.num_fingertips, 6), dtype=np.float64)
        self._sample_normal = np.zeros(self.num_fingertips, dtype=np.float64)
        self._sample_contact = np.zeros(self.num_fingertips, dtype=bool)
        self._body_origins_world = np.zeros(
            (self.num_fingertips, 3), dtype=np.float64
        )
        self._world_to_body = np.zeros(
            (self.num_fingertips, 3, 3), dtype=np.float64
        )
        self._current_contacts: list[ContactEvent] = []
        self._active = False
        self.reset()

    @property
    def num_fingertips(self) -> int:
        return len(self.fingertip_body_names)

    def reset(self) -> None:
        """Clear the active interval and publish a zero-valued reset reading."""

        self._clear_interval()
        self._active = False
        self._latest = {
            "fingertip_wrench_local": np.zeros(
                (self.num_fingertips, 6), dtype=np.float32
            ),
            "fingertip_normal": np.zeros(self.num_fingertips, dtype=np.float32),
            "fingertip_force_peak": np.zeros(
                self.num_fingertips, dtype=np.float32
            ),
            "fingertip_impulse_local": np.zeros(
                (self.num_fingertips, 3), dtype=np.float32
            ),
            "contact_fraction": np.zeros(self.num_fingertips, dtype=np.float32),
            "sim_time": np.asarray([self.data.time], dtype=np.float64),
        }

    def begin(self) -> None:
        """Begin a new control interval, discarding unfinished accumulations."""

        self._clear_interval()
        self._active = True

    def accumulate_substep(self) -> None:
        """Accumulate the contacts currently present in ``mjData``.

        Call this exactly once immediately after each ``mujoco.mj_step`` in the
        control interval.
        """

        if not self._active:
            raise RuntimeError("begin() must be called before accumulate_substep().")

        self._extract_current_contacts()
        self._wrench_sum += self._sample_wrench
        self._normal_sum += self._sample_normal
        self._normal_peak = np.maximum(self._normal_peak, self._sample_normal)
        self._force_impulse += self._sample_wrench[:, :3] * self.physics_dt
        self._contact_substeps += self._sample_contact
        self._substep_count += 1

    def finish(self) -> dict[str, np.ndarray]:
        """Finish the interval and return a copy of its observation fields."""

        if not self._active:
            raise RuntimeError("begin() must be called before finish().")
        if self._substep_count == 0:
            raise RuntimeError("At least one physics substep must be accumulated.")

        count = float(self._substep_count)
        self._latest = {
            "fingertip_wrench_local": np.asarray(
                self._wrench_sum / count, dtype=np.float32
            ),
            "fingertip_normal": np.asarray(
                self._normal_sum / count, dtype=np.float32
            ),
            # Peak of the summed contact-normal force within any physics substep.
            "fingertip_force_peak": np.asarray(
                self._normal_peak, dtype=np.float32
            ),
            "fingertip_impulse_local": np.asarray(
                self._force_impulse, dtype=np.float32
            ),
            "contact_fraction": np.asarray(
                self._contact_substeps / count, dtype=np.float32
            ),
            "sim_time": np.asarray([self.data.time], dtype=np.float64),
        }
        self._active = False
        return self.observation()

    def observation(self) -> dict[str, np.ndarray]:
        """Return defensive copies of the most recently completed interval."""

        return {name: value.copy() for name, value in self._latest.items()}

    def current_contacts(self, *, copy: bool = True) -> tuple[ContactEvent, ...]:
        """Return the force-bearing contacts from the latest substep.

        This is deliberately a substep-level diagnostic interface.  The next
        call replaces the event list with the new MuJoCo contacts.
        The default returns defensive array copies.  ``copy=False`` avoids
        those allocations but the returned arrays must be treated as read-only.
        """

        if not copy:
            return tuple(self._current_contacts)
        return tuple(
            ContactEvent(
                fingertip_index=event.fingertip_index,
                position_local=event.position_local.copy(),
                force_local=event.force_local.copy(),
                torque_local_at_contact=event.torque_local_at_contact.copy(),
                normal_force=event.normal_force,
            )
            for event in self._current_contacts
        )

    def _clear_interval(self) -> None:
        self._wrench_sum = np.zeros((self.num_fingertips, 6), dtype=np.float64)
        self._normal_sum = np.zeros(self.num_fingertips, dtype=np.float64)
        self._normal_peak = np.zeros(self.num_fingertips, dtype=np.float64)
        self._force_impulse = np.zeros((self.num_fingertips, 3), dtype=np.float64)
        self._contact_substeps = np.zeros(self.num_fingertips, dtype=np.int64)
        self._substep_count = 0

    def _extract_current_contacts(self) -> None:
        self._sample_wrench.fill(0.0)
        self._sample_normal.fill(0.0)
        self._sample_contact.fill(False)
        self._current_contacts.clear()
        self._body_origins_world[:] = self.data.xpos[self.body_ids]
        body_to_world = self.data.xmat[self.body_ids].reshape(-1, 3, 3)
        self._world_to_body[:] = body_to_world.transpose(0, 2, 1)

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            # MuJoCo represents flex contacts with a geom id of -1.  Indexing
            # the lookup array directly would interpret that as the last rigid
            # geom and could silently attribute a flex contact to a fingertip.
            fingertip1 = self._fingertip_index_for_geom(contact.geom[0])
            fingertip2 = self._fingertip_index_for_geom(contact.geom[1])
            if fingertip1 < 0 and fingertip2 < 0:
                continue
            # Contacts inside a margin can exist without an active constraint.
            # They are geometric proximity, not a force-bearing tactile event.
            if contact.efc_address < 0:
                continue

            self._contact_wrench.fill(0.0)
            mujoco.mj_contactForce(
                self.model, self.data, contact_index, self._contact_wrench
            )

            # contact.frame stores contact axes as rows, so its transpose maps
            # contact-frame vectors into the world frame.
            contact_to_world = np.asarray(contact.frame).reshape(3, 3).T
            force_world_on_geom2 = contact_to_world @ self._contact_wrench[:3]
            torque_world_on_geom2 = contact_to_world @ self._contact_wrench[3:]
            normal_force = max(float(self._contact_wrench[0]), 0.0)

            if fingertip1 >= 0:
                self._add_contact_to_fingertip(
                    fingertip1,
                    contact.pos,
                    -force_world_on_geom2,
                    -torque_world_on_geom2,
                    normal_force,
                )
            if fingertip2 >= 0:
                self._add_contact_to_fingertip(
                    fingertip2,
                    contact.pos,
                    force_world_on_geom2,
                    torque_world_on_geom2,
                    normal_force,
                )

        # Apply the activity threshold to the fingertip's total load in this
        # substep.  Comparing each individual contact would incorrectly mark a
        # multi-point contact inactive when every point is below the threshold
        # but their physically measured sum is above it.
        self._sample_contact[:] = self._sample_normal > self.contact_threshold

    def _fingertip_index_for_geom(self, geom_id: int) -> int:
        """Return the tracked fingertip for a rigid geom, or -1 otherwise."""

        geom_id = int(geom_id)
        if geom_id < 0 or geom_id >= self.model.ngeom:
            return -1
        return int(self._geom_to_fingertip[geom_id])

    def _add_contact_to_fingertip(
        self,
        fingertip_index: int,
        contact_position_world: np.ndarray,
        force_world: np.ndarray,
        torque_world_at_contact: np.ndarray,
        normal_force: float,
    ) -> None:
        body_origin_world = self._body_origins_world[fingertip_index]
        world_to_body = self._world_to_body[fingertip_index]
        position_local = world_to_body @ (
            contact_position_world - body_origin_world
        )
        force_local = world_to_body @ force_world
        torque_local_at_contact = world_to_body @ torque_world_at_contact
        self._current_contacts.append(
            ContactEvent(
                fingertip_index=int(fingertip_index),
                position_local=np.asarray(position_local, dtype=np.float64),
                force_local=np.asarray(force_local, dtype=np.float64),
                torque_local_at_contact=np.asarray(
                    torque_local_at_contact, dtype=np.float64
                ),
                normal_force=float(normal_force),
            )
        )

        torque_world_at_body = torque_world_at_contact + _cross3(
            contact_position_world - body_origin_world, force_world
        )
        self._sample_wrench[fingertip_index, :3] += force_local
        self._sample_wrench[fingertip_index, 3:] += (
            world_to_body @ torque_world_at_body
        )
        self._sample_normal[fingertip_index] += normal_force


def _cross3(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    ax, ay, az = first
    bx, by, bz = second
    return np.asarray(
        [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx],
        dtype=np.float64,
    )
