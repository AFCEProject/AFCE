"""C01 single decoder jointly trained with pi; calibrated GT-AUX and unchanged finger delta."""
from __future__ import annotations
import dataclasses
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from afce_all11.readout_alignment import ReadoutEffectPi, ReadoutEffectPiConfig, grouped_loss, time_weight

GTAUX_WEIGHT = 0.25628781345139295
OBJECTIVE = "c01_single_joint_gtaux_finger_fixed_v1"

class SingleFingerEffectPi(ReadoutEffectPi):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.finger_delta_weight = config.finger_delta_weight
        self.finger_delta_scale_floor = config.finger_delta_scale_floor
        self.joint_decoder_lr_scale = config.joint_decoder_lr_scale

    def auxiliary(self, estimate, reference, observation, weights, mode):
        if mode != "ground_truth":
            raise ValueError("Trainable GT-AUX only supports the ground-truth action target")
        del reference
        physical = estimate * self.effect_std.value + self.effect_mean.value
        losses = []
        for decoder, dim, state_dim in (
            (self.single_decoder, 22, 23),
            (self.bimanual_decoder, 44, 46),
        ):
            state = observation.effect_state[:, :state_dim]
            prediction = decoder(physical, state)
            target = jax.lax.stop_gradient(
                (observation.effect_actions[..., :dim] - decoder.action_mean.value)
                / decoder.action_std.value
            )
            losses.append(grouped_loss(prediction, target, observation.effect_valid, dim))
        single = observation.effect_action_dim == 22
        per_sample = jnp.where(single, losses[0], losses[1])
        return jnp.mean(weights * per_sample), (per_sample, single)

    def finger_delta_loss(self, estimate, observation, weights):
        """Match finger opening/closing amplitude and timing through the jointly trained decoder."""
        physical = estimate * self.effect_std.value + self.effect_mean.value
        losses = []
        for decoder, dim, state_dim in (
            (self.single_decoder, 22, 23),
            (self.bimanual_decoder, 44, 46),
        ):
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

    def compute_loss_and_metrics_at_step(self, rng, observation, actions, *, step, train=False):
        if observation.effect_valid is None:
            raise ValueError("Aligned valid frame masks are required")
        fm, clean, target, velocity, truth, time = self.flow(rng, observation, train)
        def auxiliary_objective(value):
            gtaux, (per_sample, single) = self.auxiliary(value, target, observation, time_weight(time), "ground_truth")
            finger, _ = self.finger_delta_loss(value, observation, time_weight(time))
            return self.alignment_weight * gtaux + self.finger_delta_weight * finger, (gtaux, finger, per_sample, single)
        (raw, (gtaux, finger, per_sample, single)), grad = jax.value_and_grad(auxiliary_objective, has_aux=True)(clean)
        ramp = jnp.clip((step-self.alignment_start)/self.alignment_ramp_steps, 0., 1.)
        gtaux_scaled = self.alignment_weight*ramp*gtaux
        finger_scaled = self.finger_delta_weight*ramp*finger
        action = ramp*raw
        denom = jnp.maximum(jnp.linalg.norm(2*(velocity-truth)/velocity.size),1e-12)
        ratio = jnp.linalg.norm(time[:,None,None]*jax.lax.stop_gradient(grad))/denom
        eligible = (time>=.2)&(time<=.7)&observation.effect_valid.any(axis=-1)
        metrics = {
            "loss":fm+action, "effect_loss":fm, "action_loss":action,
            "gtaux_loss":gtaux_scaled, "finger_delta_loss":finger_scaled,
            "finger_delta_loss_raw":finger, "auxiliary_loss":gtaux_scaled,
            "auxiliary_loss_raw":gtaux, "action_loss_raw":per_sample.mean(),
            "decoder_train_enabled":jnp.asarray(1.), "ramp":ramp,
            "joint_decoder_lr_scale":jnp.asarray(self.joint_decoder_lr_scale),
            "aux_to_fm_output_grad_ratio":ramp*ratio,
            "aux_to_fm_output_grad_ratio_unramped":ratio,
            "single_samples":single.sum().astype(jnp.float32),
            "bimanual_samples":(~single).sum().astype(jnp.float32),
            "single_aux_samples":(single&eligible).sum().astype(jnp.float32),
            "bimanual_aux_samples":((~single)&eligible).sum().astype(jnp.float32),
            "valid_fraction":observation.effect_valid.mean(), "time_mean":time.mean(),
            "action_single_loss":jnp.sum(jnp.where(single,per_sample,0))/jnp.maximum(single.sum(),1),
            "action_bimanual_loss":jnp.sum(jnp.where(~single,per_sample,0))/jnp.maximum((~single).sum(),1),
        }
        return fm+action, metrics

@dataclasses.dataclass(frozen=True)
class SingleFingerEffectPiConfig(ReadoutEffectPiConfig):
    decoder_warmup_steps: int = 0
    joint_objective: str = OBJECTIVE
    alignment_target_mode: str = "ground_truth"
    alignment_weight: float = GTAUX_WEIGHT
    alignment_start: int = 0
    alignment_ramp_steps: int = 10_000
    finger_delta_weight: float = .05
    finger_delta_scale_floor: float = .1
    # Retain the previous adaptive D1's optimizer update multiplier.
    joint_decoder_lr_scale: float = .25
    def create(self, rng):
        return SingleFingerEffectPi(self, nnx.Rngs(rng))

def single_finger_config(config):
    values = dataclasses.asdict(config.model)
    values.update(decoder_warmup_steps=0, joint_objective=OBJECTIVE,
        alignment_target_mode="ground_truth", alignment_weight=GTAUX_WEIGHT,
        alignment_start=0, alignment_ramp_steps=10_000,
        finger_delta_weight=.05, finger_delta_scale_floor=.1, joint_decoder_lr_scale=.25)
    return dataclasses.replace(config, model=SingleFingerEffectPiConfig(**values))
