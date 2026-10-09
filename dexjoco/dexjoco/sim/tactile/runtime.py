"""Shared MuJoCo contact-ground-truth runtime for DexJoCo environments.

The runtime owns the control-interval lifecycle shared by every task while
leaving task-specific control, rewards, rendering, and reset logic in the
environment.  It intentionally does not modify the MuJoCo model or contacts.
"""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
from gymnasium import spaces

from .contact import FingertipContactExtractor
from .taxel import CapsuleTaxelLayout, GaussianTaxelProjector


def make_tactile_ground_truth_observation_space(
    *,
    hand_count: int,
    fingertips_per_hand: int = 4,
    taxels_per_fingertip: int = 16,
) -> spaces.Dict:
    """Create the public hand/finger space for ideal contact observations."""

    if (
        hand_count <= 0
        or fingertips_per_hand <= 0
        or taxels_per_fingertip <= 0
    ):
        raise ValueError(
            "hand_count, fingertips_per_hand, and taxels_per_fingertip "
            "must be positive."
        )
    fingertip_shape = (hand_count, fingertips_per_hand)
    return spaces.Dict(
        {
            "fingertip_wrench_local": spaces.Box(
                -np.inf,
                np.inf,
                shape=(*fingertip_shape, 6),
                dtype=np.float32,
            ),
            "fingertip_normal": spaces.Box(
                0.0,
                np.inf,
                shape=fingertip_shape,
                dtype=np.float32,
            ),
            "fingertip_force_peak": spaces.Box(
                0.0,
                np.inf,
                shape=fingertip_shape,
                dtype=np.float32,
            ),
            "fingertip_impulse_local": spaces.Box(
                -np.inf,
                np.inf,
                shape=(*fingertip_shape, 3),
                dtype=np.float32,
            ),
            "contact_fraction": spaces.Box(
                0.0,
                1.0,
                shape=fingertip_shape,
                dtype=np.float32,
            ),
            "taxel_force_local": spaces.Box(
                -np.inf,
                np.inf,
                shape=(*fingertip_shape, taxels_per_fingertip, 3),
                dtype=np.float32,
            ),
            "sim_time": spaces.Box(
                0.0,
                np.inf,
                shape=(1,),
                dtype=np.float64,
            ),
        }
    )


class TactileRuntime:
    """Coordinate MuJoCo fingertip contact truth for one environment.

    Fingertips must be ordered by hand and then by the public finger order.
    For example, a single right hand uses ``index, middle, ring, thumb`` and a
    bimanual runtime uses the four right fingertips followed by the four left
    fingertips.  The runtime source is MuJoCo contact constraint force.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        fingertip_body_names_by_hand: Sequence[Sequence[str]],
        *,
        physics_dt: float,
    ) -> None:
        self.model = model
        self.data = data
        self.fingertip_body_names_by_hand = tuple(
            tuple(names) for names in fingertip_body_names_by_hand
        )
        self.hand_count = len(self.fingertip_body_names_by_hand)
        if self.hand_count == 0:
            raise ValueError("At least one hand is required.")
        fingertip_counts = {
            len(names) for names in self.fingertip_body_names_by_hand
        }
        if 0 in fingertip_counts:
            raise ValueError("Every hand must contain at least one fingertip.")
        if len(fingertip_counts) != 1:
            raise ValueError(
                "Every hand must use the same number of fingertips; got "
                f"{sorted(fingertip_counts)}."
            )
        self.fingertips_per_hand = next(iter(fingertip_counts))
        self.fingertip_body_names = tuple(
            name
            for hand_names in self.fingertip_body_names_by_hand
            for name in hand_names
        )
        if len(set(self.fingertip_body_names)) != len(
            self.fingertip_body_names
        ):
            raise ValueError("Fingertip body names must be unique across hands.")

        self.physics_dt = float(physics_dt)
        if not np.isfinite(self.physics_dt) or self.physics_dt <= 0.0:
            raise ValueError(
                f"physics_dt must be finite and positive, got {physics_dt!r}."
            )
        self.extractor = FingertipContactExtractor(
            model,
            data,
            self.fingertip_body_names,
            physics_dt=self.physics_dt,
        )
        self.taxel_layout = CapsuleTaxelLayout(
            model,
            self.fingertip_body_names,
        )
        self.taxel_projector = GaussianTaxelProjector(self.taxel_layout)
        self.taxels_per_fingertip = self.taxel_layout.num_taxels
        flat_taxel_shape = (
            len(self.fingertip_body_names),
            self.taxels_per_fingertip,
            3,
        )
        self._taxel_force_sum = np.zeros(flat_taxel_shape, dtype=np.float64)
        self._latest_taxel_force_local = np.zeros(
            flat_taxel_shape,
            dtype=np.float32,
        )
        self._taxel_substep_count = 0
        self.observation_space = spaces.Dict(
            {
                "tactile_gt": make_tactile_ground_truth_observation_space(
                    hand_count=self.hand_count,
                    fingertips_per_hand=self.fingertips_per_hand,
                    taxels_per_fingertip=self.taxels_per_fingertip,
                ),
            }
        )

    def reset(self) -> None:
        """Reset the control-interval ground-truth state."""

        self.extractor.reset()
        self._clear_taxel_interval()
        self._latest_taxel_force_local.fill(0.0)

    def begin(self) -> None:
        """Begin one control interval."""

        self.extractor.begin()
        self._clear_taxel_interval()

    def accumulate_substep(self) -> None:
        """Consume contacts after exactly one completed MuJoCo physics step."""

        self.extractor.accumulate_substep()
        self._taxel_force_sum += self.taxel_projector.project(
            self.extractor.current_contacts(copy=False)
        )
        self._taxel_substep_count += 1

    def finish(self) -> None:
        """Publish the current control-interval ground truth."""

        self.extractor.finish()
        # The extractor and taxel projection consume the same completed
        # physics substeps, so this count is necessarily positive whenever
        # extractor.finish() succeeds.
        self._latest_taxel_force_local = np.asarray(
            self._taxel_force_sum / float(self._taxel_substep_count),
            dtype=np.float32,
        )

    def observation(self) -> dict[str, dict[str, np.ndarray]]:
        """Return schema-stable MuJoCo contact-ground-truth observations."""

        tactile_gt = self.extractor.observation()
        fingertip_shape = (self.hand_count, self.fingertips_per_hand)
        formatted_gt = {
            "fingertip_wrench_local": tactile_gt[
                "fingertip_wrench_local"
            ].reshape(*fingertip_shape, 6),
            "fingertip_normal": tactile_gt["fingertip_normal"].reshape(
                fingertip_shape
            ),
            "fingertip_force_peak": tactile_gt[
                "fingertip_force_peak"
            ].reshape(fingertip_shape),
            "fingertip_impulse_local": tactile_gt[
                "fingertip_impulse_local"
            ].reshape(*fingertip_shape, 3),
            "contact_fraction": tactile_gt["contact_fraction"].reshape(
                fingertip_shape
            ),
            "taxel_force_local": self._latest_taxel_force_local.reshape(
                *fingertip_shape,
                self.taxels_per_fingertip,
                3,
            ).copy(),
            "sim_time": tactile_gt["sim_time"].copy(),
        }
        return {"tactile_gt": formatted_gt}

    def _clear_taxel_interval(self) -> None:
        """Clear unpublished taxel samples for one control interval."""

        self._taxel_force_sum.fill(0.0)
        self._taxel_substep_count = 0
