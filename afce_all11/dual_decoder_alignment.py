"""C01 E policy with a frozen supervisory decoder and an adaptive inference decoder."""
from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from afce_all11.jax_action_decoder import ActionDecoder, load_arrays
from afce_all11.readout_alignment import (
    ReadoutEffectPi,
    ReadoutEffectPiConfig,
    frozen_decoder,
    grouped_loss,
    time_weight,
)
from openpi.shared import array_typing as at


class DualDecoderEffectPi(ReadoutEffectPi):
    """Keep D0 fixed for E supervision and adapt D1 only for action inference."""

    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        arrays = load_arrays(config.decoder_init_path)
        self.adaptive_single_decoder = ActionDecoder(arrays, 22)
        self.adaptive_bimanual_decoder = ActionDecoder(arrays, 44)
        self.adaptation_weight = float(config.adaptation_weight)
        self.adaptation_anchor_weight = float(config.adaptation_anchor_weight)
        self.adaptation_rollout_weight = float(config.adaptation_rollout_weight)
        self.adaptation_start = int(config.adaptation_start)
        self.adaptation_ramp_steps = int(config.adaptation_ramp_steps)
        self.adaptation_rollout_interval = int(config.adaptation_rollout_interval)
        self.adaptation_rollout_batch = int(config.adaptation_rollout_batch)
        self.adaptation_rollout_stride = int(config.adaptation_rollout_stride)
        self.adaptive_decoder_lr_scale = float(config.adaptive_decoder_lr_scale)
        self.finger_delta_weight = float(config.finger_delta_weight)
        self.finger_delta_scale_floor = float(config.finger_delta_scale_floor)

    def adaptive_loss(self, estimate, observation):
        # D1 must fit the policy's actual E distribution without providing a
        # second, uncontrolled gradient path into the policy itself.
        physical = jax.lax.stop_gradient(
            estimate * self.effect_std.value + self.effect_mean.value
        )
        losses = []
        for decoder, dim, state_dim in (
            (self.adaptive_single_decoder, 22, 23),
            (self.adaptive_bimanual_decoder, 44, 46),
        ):
            prediction = decoder(physical, observation.effect_state[:, :state_dim])
            target = jax.lax.stop_gradient(
                (observation.effect_actions[..., :dim] - decoder.action_mean.value)
                / decoder.action_std.value
            )
            losses.append(grouped_loss(
                prediction, target, observation.effect_valid, dim
            ))
        single = observation.effect_action_dim == 22
        per_sample = jnp.where(single, losses[0], losses[1])
        return per_sample.mean(), (per_sample, single)

    def finger_delta_loss(self, estimate, observation, weights):
        """Match finger opening/closing amplitude and timing through frozen D0."""
        physical = estimate * self.effect_std.value + self.effect_mean.value
        losses = []
        for decoder, dim, state_dim in (
            (self.single_decoder, 22, 23),
            (self.bimanual_decoder, 44, 46),
        ):
            decoder = frozen_decoder(decoder)
            prediction = decoder(physical, observation.effect_state[:, :state_dim])
            target = jax.lax.stop_gradient(
                (observation.effect_actions[..., :dim] - decoder.action_mean.value)
                / decoder.action_std.value
            )
            finger_slices = ((6, 22),) if dim == 22 else ((6, 22), (28, 44))
            predicted_fingers = jnp.concatenate(
                [prediction[..., lo:hi] for lo, hi in finger_slices], axis=-1
            )
            target_fingers = jnp.concatenate(
                [target[..., lo:hi] for lo, hi in finger_slices], axis=-1
            )
            predicted_delta = predicted_fingers[:, 1:] - predicted_fingers[:, :-1]
            target_delta = target_fingers[:, 1:] - target_fingers[:, :-1]
            valid = observation.effect_valid[:, 1:] & observation.effect_valid[:, :-1]
            valid = valid & (observation.effect_action_dim == dim)[:, None]
            scale_mask = jnp.broadcast_to(valid[..., None], target_delta.shape)
            scale = jax.lax.stop_gradient(jnp.maximum(
                jnp.sqrt(
                    jnp.sum(jnp.where(scale_mask, jnp.square(target_delta), 0), axis=(0, 1), keepdims=True)
                    / jnp.maximum(scale_mask.sum(axis=(0, 1), keepdims=True), 1)
                    + 1e-8
                ),
                self.finger_delta_scale_floor,
            ))
            error = (predicted_delta - target_delta) / scale
            absolute = jnp.abs(error)
            huber = jnp.where(absolute < 1.0, 0.5 * jnp.square(error), absolute - 0.5)
            mask = jnp.broadcast_to(valid[..., None], huber.shape)
            losses.append(
                jnp.sum(jnp.where(mask, huber, 0), axis=(1, 2))
                / jnp.maximum(mask.sum(axis=(1, 2)), 1)
            )
        single = observation.effect_action_dim == 22
        per_sample = jnp.where(single, losses[0], losses[1])
        return jnp.mean(weights * per_sample), per_sample

    def compute_loss_and_metrics_at_step(
        self, rng, observation, actions, *, step, train=False
    ):
        if observation.effect_valid is None:
            raise ValueError("Aligned valid frame masks are required")
        fm, clean, target, velocity, truth, time = self.flow(rng, observation, train)

        # D0: frozen C01 decoder. This is the only auxiliary path that updates
        # the policy, and it uses the calibrated GT-AUX weight.
        def d0_objective(value):
            auxiliary, (per_sample, single_mask) = self.auxiliary(
                value, target, observation, time_weight(time), "ground_truth"
            )
            finger, finger_per_sample = self.finger_delta_loss(
                value, observation, time_weight(time)
            )
            total = self.alignment_weight * auxiliary + self.finger_delta_weight * finger
            return total, (auxiliary, finger, per_sample, finger_per_sample, single_mask)

        (d0_raw, (auxiliary, finger, d0_per_sample, finger_per_sample, single)), d0_gradient = jax.value_and_grad(
            d0_objective,
            has_aux=True,
        )(clean)
        d0_ramp = jnp.clip(
            (step - self.alignment_start) / self.alignment_ramp_steps, 0.0, 1.0
        )
        gtaux_scaled = self.alignment_weight * d0_ramp * auxiliary
        finger_scaled = self.finger_delta_weight * d0_ramp * finger
        d0_scaled = d0_ramp * d0_raw

        # D1: adaptive inference decoder. stop_gradient inside adaptive_loss
        # ensures this term only updates D1.
        adaptation, (d1_per_sample, _) = self.adaptive_loss(clean, observation)
        anchor, (anchor_per_sample, _) = self.adaptive_loss(target, observation)
        rollout_active = (step + 1) % self.adaptation_rollout_interval == 0

        def rollout_adaptation(_):
            rollout_observation = jax.tree.map(
                lambda value: value[:: self.adaptation_rollout_stride][
                    : self.adaptation_rollout_batch
                ],
                observation,
            )
            rollout_rng = jax.random.fold_in(rng, jnp.asarray(0xD1, dtype=jnp.uint32))
            sampled = self.sample_actions(
                rollout_rng, rollout_observation, num_steps=10
            )
            return self.adaptive_loss(sampled, rollout_observation)[0]

        rollout = jax.lax.cond(
            rollout_active,
            rollout_adaptation,
            lambda _: jnp.asarray(0.0, dtype=jnp.float32),
            operand=None,
        )
        d1_ramp = jnp.clip(
            (step + 1 - self.adaptation_start) / self.adaptation_ramp_steps,
            0.0,
            1.0,
        )
        prediction_scaled = self.adaptation_weight * d1_ramp * adaptation
        anchor_scaled = self.adaptation_anchor_weight * d1_ramp * anchor
        rollout_scaled = self.adaptation_rollout_weight * d1_ramp * rollout
        d1_scaled = prediction_scaled + anchor_scaled + rollout_scaled

        denominator = jnp.maximum(
            jnp.linalg.norm(2 * (velocity - truth) / velocity.size), 1e-12
        )
        ratio = (
            jnp.linalg.norm(
                time[:, None, None] * jax.lax.stop_gradient(d0_gradient)
            )
            / denominator
        )
        total = fm + d0_scaled + d1_scaled
        metrics = {
            "loss": total,
            "effect_loss": fm,
            "action_loss": d0_scaled,
            "gtaux_loss": gtaux_scaled,
            "finger_delta_loss": finger_scaled,
            "finger_delta_loss_raw": finger,
            "auxiliary_loss": gtaux_scaled,
            "auxiliary_loss_raw": auxiliary,
            "action_loss_raw": d0_per_sample.mean(),
            "adaptation_loss": d1_scaled,
            "adaptation_prediction_loss": prediction_scaled,
            "adaptation_anchor_loss": anchor_scaled,
            "adaptation_rollout_loss": rollout_scaled,
            "adaptation_loss_raw": adaptation,
            "adaptation_anchor_loss_raw": anchor,
            "adaptation_rollout_loss_raw": rollout,
            "adaptation_rollout_active": rollout_active.astype(jnp.float32),
            "decoder_train_enabled": jnp.asarray(0.0),
            "ramp": d0_ramp,
            "adaptation_ramp": d1_ramp,
            "aux_to_fm_output_grad_ratio": d0_ramp * ratio,
            "aux_to_fm_output_grad_ratio_unramped": ratio,
            "single_samples": single.sum().astype(jnp.float32),
            "bimanual_samples": (~single).sum().astype(jnp.float32),
            "valid_fraction": observation.effect_valid.mean(),
            "time_mean": time.mean(),
            "action_single_loss": jnp.sum(jnp.where(single, d0_per_sample, 0))
            / jnp.maximum(single.sum(), 1),
            "action_bimanual_loss": jnp.sum(jnp.where(~single, d0_per_sample, 0))
            / jnp.maximum((~single).sum(), 1),
            "adaptation_single_loss": jnp.sum(jnp.where(single, d1_per_sample, 0))
            / jnp.maximum(single.sum(), 1),
            "adaptation_bimanual_loss": jnp.sum(jnp.where(~single, d1_per_sample, 0))
            / jnp.maximum((~single).sum(), 1),
            "anchor_single_loss": jnp.sum(jnp.where(single, anchor_per_sample, 0))
            / jnp.maximum(single.sum(), 1),
            "anchor_bimanual_loss": jnp.sum(jnp.where(~single, anchor_per_sample, 0))
            / jnp.maximum((~single).sum(), 1),
            "finger_single_loss": jnp.sum(jnp.where(single, finger_per_sample, 0))
            / jnp.maximum(single.sum(), 1),
            "finger_bimanual_loss": jnp.sum(jnp.where(~single, finger_per_sample, 0))
            / jnp.maximum((~single).sum(), 1),
        }
        for label, mask in (
            ("low", time < 0.2),
            ("middle", (time >= 0.2) & (time <= 0.7)),
            ("high", time > 0.7),
        ):
            metrics["count_noise_" + label] = mask.sum().astype(jnp.float32)
            metrics["action_noise_" + label] = jnp.sum(
                jnp.where(mask, d0_per_sample, 0)
            ) / jnp.maximum(mask.sum(), 1)
            metrics["adaptation_noise_" + label] = jnp.sum(
                jnp.where(mask, d1_per_sample, 0)
            ) / jnp.maximum(mask.sum(), 1)
        return total, metrics


@dataclasses.dataclass(frozen=True)
class DualDecoderEffectPiConfig(ReadoutEffectPiConfig):
    adaptation_weight: float = 1.0
    adaptation_anchor_weight: float = 0.25
    adaptation_rollout_weight: float = 0.1
    adaptation_start: int = 0
    adaptation_ramp_steps: int = 10_000
    adaptation_rollout_interval: int = 20
    adaptation_rollout_batch: int = 8
    adaptation_rollout_stride: int = 4
    adaptive_decoder_lr_scale: float = 0.25
    finger_delta_weight: float = 0.05
    finger_delta_scale_floor: float = 0.1
    joint_objective: str = "c01_gtaux_finger_d0_plus_anchored_rollout_d1_v1"

    def create(self, rng):
        return DualDecoderEffectPi(self, nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        observation, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation = dataclasses.replace(
                observation,
                effect_valid=jax.ShapeDtypeStruct((batch_size, 30), jnp.bool_),
            )
        return observation, actions


def dual_decoder_config(
    config,
    *,
    adaptation_weight=1.0,
    adaptation_anchor_weight=0.25,
    adaptation_rollout_weight=0.1,
    adaptation_start=0,
    adaptation_ramp_steps=10_000,
    adaptation_rollout_interval=20,
    adaptation_rollout_batch=8,
    adaptation_rollout_stride=4,
    adaptive_decoder_lr_scale=0.25,
    finger_delta_weight=0.05,
    finger_delta_scale_floor=0.1,
):
    values = dataclasses.asdict(config.model)
    values.update(
        joint_objective="c01_gtaux_finger_d0_plus_anchored_rollout_d1_v1",
        adaptation_weight=adaptation_weight,
        adaptation_anchor_weight=adaptation_anchor_weight,
        adaptation_rollout_weight=adaptation_rollout_weight,
        adaptation_start=adaptation_start,
        adaptation_ramp_steps=adaptation_ramp_steps,
        adaptation_rollout_interval=adaptation_rollout_interval,
        adaptation_rollout_batch=adaptation_rollout_batch,
        adaptation_rollout_stride=adaptation_rollout_stride,
        adaptive_decoder_lr_scale=adaptive_decoder_lr_scale,
        finger_delta_weight=finger_delta_weight,
        finger_delta_scale_floor=finger_delta_scale_floor,
    )
    return dataclasses.replace(config, model=DualDecoderEffectPiConfig(**values))


class DualDecodedEffectPolicy:
    """Decode predicted C01 E with the trained adaptive D1 decoder."""

    def __init__(self, latent_policy, model, *, task):
        self.policy = latent_policy
        self.state_dim = 46 if task.startswith("bimanual_") else 23
        decoder = (
            model.adaptive_bimanual_decoder
            if self.state_dim == 46
            else model.adaptive_single_decoder
        )
        self.decode = jax.jit(
            lambda effect, state: decoder.physical(
                effect * model.effect_std.value + model.effect_mean.value, state
            )
        )

    @property
    def metadata(self):
        return self.policy.metadata

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def infer(self, observation, **kwargs):
        import time

        import numpy as np

        state = np.asarray(observation["state"], dtype=np.float32)[: self.state_dim]
        if state.shape != (self.state_dim,):
            raise ValueError("Adaptive decoder requires the current unnormalized state")
        started = time.monotonic()
        result = self.policy.infer(observation, **kwargs)
        effect = np.asarray(result["actions"], dtype=np.float32)
        if effect.shape != (30, 256):
            raise ValueError("Dual-decoder policy must return the complete 30x256 E")
        result["actions"] = np.asarray(self.decode(effect[None], state[None]))[0]
        result["policy_timing"] = {"infer_ms": 1000 * (time.monotonic() - started)}
        return result
