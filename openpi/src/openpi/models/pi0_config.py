import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Opt-in DexJoCo Action Expert input. The 46-D canonical state is decoded
    # into two palm tokens and 32 topology-aware Allegro joint tokens instead
    # of being available only as discretized language tokens.
    structured_hand_state: bool = False

    # Task-Effect + Robot Grounding (effect_vla). Off by default so every
    # existing DexJoCo π0.5 config is unchanged.
    effect_grounding: bool = False
    num_effect_anchors: int = 2
    effect_dim: int = 512
    grounding_dim: int = 512
    geo_dim: int = 42
    lambda_effect: float = 0.2
    lambda_grounding: float = 0.1
    condition_effect: bool = True
    use_grounding: bool = True

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.structured_hand_state and not self.pi05:
            raise ValueError("structured_hand_state currently supports π0.5 only.")
        if self.structured_hand_state and self.action_dim != 44:
            raise ValueError("structured_hand_state expects the DexJoCo 44-D action layout.")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        if self.effect_grounding:
            import sys
            from pathlib import Path

            dexjoco_root = Path(__file__).resolve().parents[4]
            if str(dexjoco_root) not in sys.path:
                sys.path.insert(0, str(dexjoco_root))
            from effect_vla.model.pi05_effect_policy import Pi0EffectPolicy  # noqa: PLC0415

            return Pi0EffectPolicy(self, rngs=nnx.Rngs(rng))
        from openpi.models.pi0 import Pi0  # noqa: PLC0415

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct(
                    [batch_size, 46 if self.structured_hand_state else self.action_dim], jnp.float32
                ),
                hand_presence=(
                    jax.ShapeDtypeStruct([batch_size, 2], jnp.bool_)
                    if self.structured_hand_state
                    else None
                ),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                effect_target=(
                    jax.ShapeDtypeStruct([batch_size, self.num_effect_anchors, self.effect_dim], jnp.float32)
                    if self.effect_grounding
                    else None
                ),
                grounding_target=(
                    jax.ShapeDtypeStruct([batch_size, self.num_effect_anchors, self.geo_dim], jnp.float32)
                    if self.effect_grounding
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def get_bootstrap_freeze_filter(self) -> nnx.filterlib.Filter:
        """Stage-1 freeze: VLM including its LoRA, Action Expert LoRA stays trainable.

        New Effect/Grounding/adapter params are not matched and remain trainable.
        """
        vlm_all = nnx.All(
            nnx_utils.PathRegex(".*llm.*"),
            nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
        )
        return nnx.Any(self.get_freeze_filter(), vlm_all)
