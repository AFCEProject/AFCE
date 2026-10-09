"""Frozen functional-readout supervision, with a paired ground-truth control."""
from __future__ import annotations
import dataclasses
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from afce.pi_bridge_joint import JointEffectPi, JointEffectPiConfig
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at

HUBER_DELTA = 1.0  # Same unit-threshold grouped normalized Huber as the parent.
MID_PROBABILITY = ((.7-.001)/.999)**1.5 - ((.2-.001)/.999)**1.5


def frozen_decoder(decoder):
    graph, state = nnx.split(decoder)
    return nnx.merge(graph, jax.tree.map(jax.lax.stop_gradient, state))


def grouped_loss(prediction, target_normalized, valid, dimension):
    groups = []
    for offset in range(0, dimension, 22):
        for lo, hi in ((0,3), (3,6), (6,22)):
            error = prediction[..., offset+lo:offset+hi] - target_normalized[..., offset+lo:offset+hi]
            absolute = jnp.abs(error)
            values = jnp.where(absolute < HUBER_DELTA, .5*error**2, absolute-.5)
            weights = jnp.broadcast_to(valid[...,None], values.shape)
            groups.append(jnp.sum(jnp.where(weights, values, 0), (1,2))/jnp.maximum(weights.sum((1,2)),1))
    return jnp.stack(groups).mean(0)


def decoder_feature_loss(decoder, prediction, target, valid):
    """Measure E in the frozen decoder's own input representation.

    The target RMS removes the arbitrary scale of each pretrained decoder while
    retaining token directions and relative magnitudes.  Invalid padded tokens
    never contribute.  Both decoder parameters and target features are fixed;
    therefore every useful gradient must change the policy's predicted E.
    """
    predicted_features = decoder.linear(prediction.astype(jnp.float32), 'e_proj')
    target_features = jax.lax.stop_gradient(
        decoder.linear(target.astype(jnp.float32), 'e_proj'))
    mask = jnp.broadcast_to(valid[..., None], target_features.shape)
    count = jnp.maximum(mask.sum((1, 2)), 1)
    target_scale = jax.lax.stop_gradient(jnp.sqrt(
        jnp.sum(jnp.where(mask, jnp.square(target_features), 0), axis=(1, 2)) / count + 1e-6))
    error = (predicted_features - target_features) / target_scale[:, None, None]
    absolute = jnp.abs(error)
    values = jnp.where(absolute < HUBER_DELTA, .5 * jnp.square(error), absolute - .5)
    return jnp.sum(jnp.where(mask, values, 0), axis=(1, 2)) / count


def time_weight(time):
    return ((time >= .2) & (time <= .7)).astype(jnp.float32)/MID_PROBABILITY


class ReadoutEffectPi(JointEffectPi):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.alignment_target_mode = config.alignment_target_mode
        self.alignment_weight = config.alignment_weight
        self.alignment_start = config.alignment_start
        self.alignment_ramp_steps = config.alignment_ramp_steps

    def flow(self, rng, observation, train):
        target = jax.lax.stop_gradient(observation.effect_target)
        pre_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model_lib.preprocess_observation(pre_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, target.shape)
        time = jax.random.beta(time_rng, 1.5, 1, target.shape[:-2])*.999+.001
        t = time[...,None,None]
        noisy = t*noise+(1-t)*target
        truth_velocity = noise-target
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, condition = self.embed_suffix(observation, noisy, time)
        mask = jnp.concatenate([prefix_mask,suffix_mask],1)
        ar = jnp.concatenate([prefix_ar,suffix_ar],0)
        (_, suffix_out), _ = self.PaliGemma.llm([prefix,suffix], mask=make_attn_mask(mask,ar),
            positions=jnp.cumsum(mask,axis=1)-1, adarms_cond=[None,condition])
        velocity = self.action_out_proj(suffix_out[:,-self.action_horizon:]).astype(jnp.float32)
        fm = jnp.mean(jnp.square(velocity-truth_velocity))
        return fm, noisy-t*velocity, target, velocity, truth_velocity, time

    def auxiliary(self, estimate, reference, observation, weights, mode):
        physical = estimate*self.effect_std.value+self.effect_mean.value
        real = reference*self.effect_std.value+self.effect_mean.value
        losses = []
        for decoder, dim, state_dim in ((self.single_decoder,22,23),(self.bimanual_decoder,44,46)):
            decoder = frozen_decoder(decoder)
            if mode == 'feature_alignment':
                losses.append(decoder_feature_loss(
                    decoder, physical, real, observation.effect_valid))
            else:
                state = observation.effect_state[:,:state_dim]
                predicted = decoder(physical,state)
                if mode == 'reference_readout':
                    target = jax.lax.stop_gradient(decoder(real,state))
                elif mode == 'ground_truth':
                    target = jax.lax.stop_gradient((observation.effect_actions[...,:dim]-decoder.action_mean.value)/decoder.action_std.value)
                else:
                    raise ValueError(mode)
                losses.append(grouped_loss(predicted,target,observation.effect_valid,dim))
        single = observation.effect_action_dim == 22
        per_sample = jnp.where(single,losses[0],losses[1])
        return jnp.mean(weights*per_sample), (per_sample,single)

    def component_loss(self, rng, observation, kind):
        fm, clean, target, _, _, time = self.flow(rng,observation,True)
        if kind == 'fm':
            return fm
        return self.auxiliary(clean,target,observation,time_weight(time),kind)[0]

    def calibration_metrics(self, rng, observation):
        fm, clean, target, velocity, truth, time = self.flow(rng,observation,True)
        denominator = jnp.maximum(jnp.linalg.norm(2*(velocity-truth)/velocity.size),1e-12)
        values = {'fm':fm,'single_samples':(observation.effect_action_dim==22).sum(),
                  'mid_samples':((time>=.2)&(time<=.7)).sum()}
        for mode in ('reference_readout','ground_truth','feature_alignment'):
            loss, grad = jax.value_and_grad(lambda z:self.auxiliary(z,target,observation,time_weight(time),mode)[0])(clean)
            values[mode+'_loss'] = loss
            values[mode+'_unweighted_output_grad_ratio'] = jnp.linalg.norm(time[:,None,None]*grad)/denominator
        return values

    def compute_loss_and_metrics_at_step(self,rng,observation,actions,*,step,train=False):
        # Preserve the parent's exact BF16/optimizer path when lambda is zero.
        # Merely multiplying a different traced auxiliary graph by zero can
        # change XLA fusion and last-bit gradients, despite equal scalar logs.
        if self.alignment_target_mode == 'fm_reference' or self.alignment_weight == 0:
            loss, values = JointEffectPi.compute_loss_and_metrics_at_step(self,rng,observation,actions,step=step,train=train)
            values.update(auxiliary_loss=jnp.asarray(0.),auxiliary_loss_raw=jnp.asarray(0.),
                          aux_to_fm_output_grad_ratio=jnp.asarray(0.),ramp=jnp.asarray(0.))
            return loss,values
        if observation.effect_valid is None:
            raise ValueError('Aligned valid frame masks are required')
        fm, clean, target, velocity, truth, time = self.flow(rng,observation,train)
        mode = self.alignment_target_mode
        (auxiliary,(per_sample,single)), gradient = jax.value_and_grad(
            lambda z:self.auxiliary(z,target,observation,time_weight(time),mode),has_aux=True)(clean)
        ramp = jnp.clip((step-self.alignment_start)/self.alignment_ramp_steps,0.,1.)
        scaled = self.alignment_weight*ramp*auxiliary
        denominator = jnp.maximum(jnp.linalg.norm(2*(velocity-truth)/velocity.size),1e-12)
        ratio = self.alignment_weight*jnp.linalg.norm(time[:,None,None]*jax.lax.stop_gradient(gradient))/denominator
        metrics = {'loss':fm+scaled,'effect_loss':fm,'action_loss':scaled,'auxiliary_loss':scaled,
                   'action_loss_raw':per_sample.mean(),'auxiliary_loss_raw':auxiliary,
                   'decoder_train_enabled':jnp.asarray(0.),'ramp':ramp,
                   'aux_to_fm_output_grad_ratio':ramp*ratio,'aux_to_fm_output_grad_ratio_unramped':ratio,
                   'single_samples':single.sum().astype(jnp.float32),'bimanual_samples':(~single).sum().astype(jnp.float32),
                   'valid_fraction':observation.effect_valid.mean(),'time_mean':time.mean(),
                   'action_single_loss':jnp.sum(jnp.where(single,per_sample,0))/jnp.maximum(single.sum(),1),
                   'action_bimanual_loss':jnp.sum(jnp.where(~single,per_sample,0))/jnp.maximum((~single).sum(),1)}
        for label,mask in [('low',time<.2),('middle',(time>=.2)&(time<=.7)),('high',time>.7)]:
            metrics['count_noise_'+label]=mask.sum().astype(jnp.float32)
            metrics['action_noise_'+label]=jnp.sum(jnp.where(mask,per_sample,0))/jnp.maximum(mask.sum(),1)
        return fm+scaled,metrics


@dataclasses.dataclass(frozen=True)
class ReadoutEffectPiConfig(JointEffectPiConfig):
    alignment_target_mode: str = 'reference_readout'
    alignment_weight: float = .1
    alignment_start: int = 30000
    alignment_ramp_steps: int = 1000
    alignment_time_weight: str = 'I[.2<=t<=.7]/population_probability; original Beta time unchanged'
    alignment_huber_delta: float = HUBER_DELTA
    alignment_valid_mask: str = 'auxiliary only; sample mean includes zero for all-invalid sample'

    def create(self,rng):
        return ReadoutEffectPi(self,nnx.Rngs(rng))

    def inputs_spec(self,*,batch_size=1):
        observation,actions=super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation=dataclasses.replace(observation,effect_valid=jax.ShapeDtypeStruct((batch_size,30),jnp.bool_))
        return observation,actions


def readout_config(config,variant,weight=.1,alignment_start=30000,alignment_ramp_steps=1000):
    modes={'readout':'reference_readout','gtaux':'ground_truth','feature':'feature_alignment',
           'fm':'reference_readout','reference':'fm_reference'}
    values=dataclasses.asdict(config.model)
    values.update(decoder_warmup_steps=60001,joint_objective='frozen_decoder_alignment_v2',
                  alignment_target_mode=modes[variant],
                  alignment_weight=weight if variant in ('readout','gtaux','feature') else 0.,
                  alignment_start=alignment_start, alignment_ramp_steps=alignment_ramp_steps)
    return dataclasses.replace(config,model=ReadoutEffectPiConfig(**values))
