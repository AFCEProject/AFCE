"""Force and tactile sensing utilities for DexJoCo environments."""

from .contact import ContactEvent, FingertipContactExtractor
from .runtime import (
    TactileRuntime,
    make_tactile_ground_truth_observation_space,
)
from .taxel import CapsuleTaxelLayout, GaussianTaxelProjector

SINGLE_FINGERTIP_BODY_NAMES = (
    "ff_tip",
    "mf_tip",
    "rf_tip",
    "th_tip",
)
RIGHT_FINGERTIP_BODY_NAMES = (
    "ff_tip_right",
    "mf_tip_right",
    "rf_tip_right",
    "th_tip_right",
)
LEFT_FINGERTIP_BODY_NAMES = (
    "ff_tip_left",
    "mf_tip_left",
    "rf_tip_left",
    "th_tip_left",
)
BIMANUAL_FINGERTIP_BODY_NAMES = (
    *RIGHT_FINGERTIP_BODY_NAMES,
    *LEFT_FINGERTIP_BODY_NAMES,
)

__all__ = [
    "BIMANUAL_FINGERTIP_BODY_NAMES",
    "CapsuleTaxelLayout",
    "ContactEvent",
    "FingertipContactExtractor",
    "GaussianTaxelProjector",
    "LEFT_FINGERTIP_BODY_NAMES",
    "RIGHT_FINGERTIP_BODY_NAMES",
    "SINGLE_FINGERTIP_BODY_NAMES",
    "TactileRuntime",
    "make_tactile_ground_truth_observation_space",
]
