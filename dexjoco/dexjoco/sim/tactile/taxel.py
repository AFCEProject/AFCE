"""Deterministic taxel projection derived from MuJoCo fingertip capsules.

This module turns the point contacts already solved by MuJoCo into a spatial
4-by-4 taxel reading on each fingertip.  It is deliberately an ideal,
simulation-only mapping: no sensor noise, delay, drift, saturation, or
hardware-specific calibration is introduced.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import mujoco
import numpy as np

from .contact import ContactEvent


class CapsuleTaxelLayout:
    """A regular taxel grid on each fingertip collision capsule.

    The default 4-by-4 grid has four rows along the capsule axis and four
    columns around its cylindrical contact surface.  A contact on either
    hemispherical cap is handled by the same Gaussian and naturally projects
    most strongly onto the nearest end ring.  Taxel positions and axes are
    expressed in the corresponding fingertip body frame.  For every taxel,
    ``axes_body`` stores
    its local x, y, and z axes as *columns*, so
    ``axes_body[f, t] @ force_in_taxel_frame`` maps a force back to the
    fingertip body frame.  ``axes_local`` is its transpose and performs the
    inverse mapping.  The taxel z axis is the inward capsule-surface normal, so
    a compressive contact produces a positive local-z component.

    The layout is inferred from the single collision-enabled capsule attached
    to every named fingertip body.  This keeps the spatial projection tied to
    the geometry which actually participates in MuJoCo contact generation.

    Args:
        model: Compiled MuJoCo model.
        fingertip_body_names: Fingertip body names in output order.
        rows: Number of positions along the capsule axis.
        columns: Number of positions around the capsule circumference.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        fingertip_body_names: Sequence[str],
        *,
        rows: int = 4,
        columns: int = 4,
    ) -> None:
        if not fingertip_body_names:
            raise ValueError("At least one fingertip body name is required.")
        if len(set(fingertip_body_names)) != len(fingertip_body_names):
            raise ValueError("Fingertip body names must be unique.")
        if isinstance(rows, bool) or not isinstance(rows, (int, np.integer)):
            raise TypeError(f"rows must be an integer, got {rows!r}.")
        if isinstance(columns, bool) or not isinstance(
            columns, (int, np.integer)
        ):
            raise TypeError(f"columns must be an integer, got {columns!r}.")
        if rows < 2 or columns < 3:
            raise ValueError(
                "A capsule layout requires at least 2 rows and 3 columns; "
                f"got rows={rows}, columns={columns}."
            )

        self.model = model
        self.fingertip_body_names = tuple(fingertip_body_names)
        self.rows = int(rows)
        self.columns = int(columns)

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

        geom_ids = []
        for body_name, body_id in zip(
            self.fingertip_body_names, self.body_ids, strict=True
        ):
            attached = np.flatnonzero(model.geom_bodyid == body_id)
            capsules = attached[
                collision_enabled[attached]
                & (model.geom_type[attached] == mujoco.mjtGeom.mjGEOM_CAPSULE)
            ]
            if capsules.size != 1:
                raise ValueError(
                    f"Fingertip body {body_name!r} must have exactly one "
                    "collision-enabled capsule geom; found "
                    f"{capsules.size}."
                )
            geom_ids.append(int(capsules[0]))
        self.geom_ids = np.asarray(geom_ids, dtype=np.int32)

        self.radius = np.asarray(model.geom_size[self.geom_ids, 0], dtype=np.float64)
        self.half_length = np.asarray(
            model.geom_size[self.geom_ids, 1], dtype=np.float64
        )
        if np.any(~np.isfinite(self.radius)) or np.any(self.radius <= 0.0):
            raise ValueError("Every fingertip capsule radius must be positive.")
        if np.any(~np.isfinite(self.half_length)) or np.any(
            self.half_length <= 0.0
        ):
            raise ValueError("Every fingertip capsule half-length must be positive.")

        positions = np.empty(
            (self.num_fingertips, self.num_taxels, 3), dtype=np.float64
        )
        axes = np.empty(
            (self.num_fingertips, self.num_taxels, 3, 3), dtype=np.float64
        )
        spacing = np.empty(self.num_fingertips, dtype=np.float64)

        axial_fractions = np.linspace(-1.0, 1.0, self.rows)
        angles = np.linspace(0.0, 2.0 * np.pi, self.columns, endpoint=False)
        radial_directions = np.column_stack(
            (np.cos(angles), np.sin(angles), np.zeros(self.columns))
        )

        for fingertip_index, geom_id in enumerate(self.geom_ids):
            geom_to_body = _quaternion_matrix(model.geom_quat[geom_id])
            geom_position = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
            radius = self.radius[fingertip_index]
            half_length = self.half_length[fingertip_index]

            taxel_index = 0
            for axial_fraction in axial_fractions:
                axial_position = axial_fraction * half_length
                for radial_outward in radial_directions:
                    position_geom = radius * radial_outward
                    position_geom[2] = axial_position

                    # Taxel x follows the capsule's geom-z axis.  The inward
                    # surface normal is taxel z, and y completes a right-handed
                    # frame.
                    x_axis_geom = np.asarray([0.0, 0.0, 1.0])
                    z_axis_geom = -radial_outward
                    y_axis_geom = np.cross(z_axis_geom, x_axis_geom)
                    axes_geom = np.stack(
                        (x_axis_geom, y_axis_geom, z_axis_geom), axis=0
                    )

                    positions[fingertip_index, taxel_index] = (
                        geom_position + geom_to_body @ position_geom
                    )
                    axes[fingertip_index, taxel_index] = (
                        geom_to_body @ axes_geom.T
                    ).T
                    taxel_index += 1

            spacing[fingertip_index] = _nearest_neighbour_spacing(
                positions[fingertip_index]
            )

        # Layout arrays are immutable so a projector cannot silently change
        # geometry or coordinate conventions between otherwise identical runs.
        for value in (
            self.body_ids,
            self.geom_ids,
            self.radius,
            self.half_length,
            positions,
            axes,
            spacing,
        ):
            value.setflags(write=False)
        self.positions_local = positions
        self.centers_local = positions
        self.centers_body = positions
        self.axes_local = axes
        self.axes_body = axes.transpose(0, 1, 3, 2)
        self.axes_body.setflags(write=False)
        self.spacing = spacing

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        fingertip_body_names: Sequence[str],
        *,
        rows: int = 4,
        columns: int = 4,
    ) -> CapsuleTaxelLayout:
        """Construct a layout from fingertip collision capsules in ``model``."""

        return cls(
            model,
            fingertip_body_names,
            rows=rows,
            columns=columns,
        )

    @property
    def num_fingertips(self) -> int:
        return len(self.fingertip_body_names)

    @property
    def num_taxels(self) -> int:
        return self.rows * self.columns


class GaussianTaxelProjector:
    """Project MuJoCo contact forces onto a capsule taxel layout.

    A contact's Gaussian weights are normalized over all taxels on its
    fingertip.  Thus every contact contributes exactly its full force instead
    of losing load outside a finite kernel window.  Each contribution is then
    expressed in the receiving taxel's orthonormal frame.  Rotating the output
    forces back to the fingertip frame and summing over taxels reconstructs the
    sum of input contact forces (up to floating-point roundoff).

    Args:
        layout: Capsule layout defining taxel positions and frames.
        sigma: Gaussian standard deviation in metres.  ``None`` uses each
            fingertip layout's nearest-neighbour spacing.  A scalar applies to
            every fingertip; a one-dimensional value can configure each
            fingertip independently.
    """

    def __init__(
        self,
        layout: CapsuleTaxelLayout,
        *,
        sigma: float | Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if not isinstance(layout, CapsuleTaxelLayout):
            raise TypeError(
                "layout must be a CapsuleTaxelLayout, got "
                f"{type(layout).__name__}."
            )
        self.layout = layout
        if sigma is None:
            sigma_values = np.array(layout.spacing, dtype=np.float64, copy=True)
        else:
            raw_sigma = np.asarray(sigma, dtype=np.float64)
            if raw_sigma.ndim == 0:
                sigma_values = np.full(
                    layout.num_fingertips, float(raw_sigma), dtype=np.float64
                )
            elif raw_sigma.shape == (layout.num_fingertips,):
                sigma_values = np.array(raw_sigma, dtype=np.float64, copy=True)
            else:
                raise ValueError(
                    "sigma must be a scalar or have shape "
                    f"({layout.num_fingertips},), got {raw_sigma.shape}."
                )
        if np.any(~np.isfinite(sigma_values)) or np.any(sigma_values <= 0.0):
            raise ValueError("Every sigma value must be finite and positive.")
        sigma_values.setflags(write=False)
        self.sigma = sigma_values

    def weights(
        self,
        fingertip_index: int,
        position_local: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Return normalized Gaussian weights for one contact position."""

        fingertip_index = self._validate_fingertip_index(fingertip_index)
        position = _finite_vector3(position_local, name="position_local")
        offsets = self.layout.positions_local[fingertip_index] - position
        squared_distance = np.einsum("ij,ij->i", offsets, offsets)
        log_weights = -0.5 * squared_distance / self.sigma[fingertip_index] ** 2
        # Shifting by the maximum prevents every exponential from underflowing
        # for a contact far from the finite set of taxel centres.
        unnormalized = np.exp(log_weights - np.max(log_weights))
        weights = unnormalized / np.sum(unnormalized)
        # Put the final rounding residual on the strongest taxel.  This makes
        # the finite-array sum exactly one in normal IEEE-754 arithmetic while
        # changing no meaningful spatial ordering.
        strongest = int(np.argmax(weights))
        weights[strongest] += 1.0 - np.sum(weights)
        return np.asarray(weights, dtype=np.float64)

    def project(self, contacts: Iterable[ContactEvent]) -> np.ndarray:
        """Return per-taxel three-axis forces for ``contacts``.

        The result has shape ``(num_fingertips, num_taxels, 3)`` and dtype
        ``float32``.  Empty input produces an all-zero result; multiple contacts
        superpose linearly.
        """

        projected = np.zeros(
            (self.layout.num_fingertips, self.layout.num_taxels, 3),
            dtype=np.float64,
        )
        for contact in contacts:
            fingertip_index = self._validate_fingertip_index(
                contact.fingertip_index
            )
            position = _finite_vector3(
                contact.position_local, name="contact.position_local"
            )
            force = _finite_vector3(contact.force_local, name="contact.force_local")
            weights = self.weights(fingertip_index, position)
            force_in_taxel_frames = np.einsum(
                "tij,j->ti",
                self.layout.axes_local[fingertip_index],
                force,
            )
            projected[fingertip_index] += (
                weights[:, np.newaxis] * force_in_taxel_frames
            )
        return np.asarray(projected, dtype=np.float32)

    def _validate_fingertip_index(self, fingertip_index: int) -> int:
        if isinstance(fingertip_index, bool) or not isinstance(
            fingertip_index, (int, np.integer)
        ):
            raise TypeError(
                "fingertip_index must be an integer, got "
                f"{fingertip_index!r}."
            )
        index = int(fingertip_index)
        if index < 0 or index >= self.layout.num_fingertips:
            raise IndexError(
                f"fingertip_index {index} is outside [0, "
                f"{self.layout.num_fingertips})."
            )
        return index


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def _nearest_neighbour_spacing(positions: np.ndarray) -> float:
    offsets = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(offsets, axis=-1)
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(np.min(distances, axis=1)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Taxel layout must have a positive finite spacing.")
    return spacing


def _finite_vector3(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}.")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


__all__ = ["CapsuleTaxelLayout", "GaussianTaxelProjector"]
