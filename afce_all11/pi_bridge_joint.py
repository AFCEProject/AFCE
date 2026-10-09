"""Frozen E supervision with jointly optimized policy and pretrained action decoders."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from afce_all11.jax_action_decoder import ActionDecoder, grouped_action_loss, load_arrays
from afce_all11.pi_bridge import EffectCacheDataset, EffectDataConfig, EffectPi, EffectPiConfig
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at
from openpi.training import weight_loaders


class JointEffectPi(EffectPi):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        arrays = load_arrays(config.decoder_init_path)
        self.single_decoder = ActionDecoder(arrays, 22)
        self.bimanual_decoder = ActionDecoder(arrays, 44)
        self.effect_mean = nnx.Variable(jnp.asarray(arrays['effect_mean']))
        self.effect_std = nnx.Variable(jnp.asarray(arrays['effect_std']))
        self.decoder_warmup_steps = int(config.decoder_warmup_steps)

    def compute_loss_and_metrics_at_step(self, rng, observation, actions, *, step, train=False):
        if any(value is None for value in (observation.effect_target, observation.effect_state,
                                           observation.effect_actions, observation.effect_action_dim)):
            raise ValueError('Joint training requires frozen E, raw state/actions, and embodiment identity')
        target = jax.lax.stop_gradient(observation.effect_target)
        raw_state = observation.effect_state
        raw_actions = observation.effect_actions
        action_dims = observation.effect_action_dim
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model_lib.preprocess_observation(preprocess_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, target.shape)
        time = jax.random.beta(time_rng, 1.5, 1, target.shape[:-2])*.999+.001
        time_expanded = time[..., None, None]
        noisy_effect = time_expanded*noise+(1-time_expanded)*target
        target_velocity = noise-target
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, condition = self.embed_suffix(observation, noisy_effect, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attention_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1)-1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix, suffix], mask=attention_mask, positions=positions, adarms_cond=[None, condition])
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon:]).astype(jnp.float32)
        effect_loss = jnp.mean(jnp.square(velocity-target_velocity))
        clean_effect = noisy_effect-time_expanded*velocity
        physical_effect = clean_effect*self.effect_std.value+self.effect_mean.value
        single = self.single_decoder(physical_effect, raw_state[:, :23])
        bimanual = self.bimanual_decoder(physical_effect, raw_state)
        single_loss = grouped_action_loss(single, raw_actions[..., :22], self.single_decoder)
        bimanual_loss = grouped_action_loss(bimanual, raw_actions, self.bimanual_decoder)
        is_single = action_dims == 22
        action_loss_raw = jnp.mean(jnp.where(is_single, single_loss, bimanual_loss))
        decoder_train_enabled = jnp.asarray(step >= self.decoder_warmup_steps, dtype=jnp.float32)
        action_loss = decoder_train_enabled*action_loss_raw
        total = effect_loss+action_loss
        return total, {'loss': total, 'effect_loss': effect_loss, 'action_loss': action_loss,
            'action_loss_raw': action_loss_raw, 'decoder_train_enabled': decoder_train_enabled,
            'action_single_loss': jnp.sum(jnp.where(is_single, single_loss, 0))/jnp.maximum(is_single.sum(), 1),
            'action_bimanual_loss': jnp.sum(jnp.where(~is_single, bimanual_loss, 0))/jnp.maximum((~is_single).sum(), 1)}

    def compute_loss_and_metrics(self, rng, observation, actions, *, train=False):
        return self.compute_loss_and_metrics_at_step(
            rng, observation, actions, step=jnp.asarray(self.decoder_warmup_steps), train=train)

    def compute_loss(self, rng, observation, actions, *, train=False):
        total, _ = self.compute_loss_and_metrics(rng, observation, actions, train=train)
        return jnp.broadcast_to(total, observation.effect_target.shape[:-1])


@dataclasses.dataclass(frozen=True)
class JointEffectPiConfig(EffectPiConfig):
    decoder_init_path: str = ''
    decoder_init_sha256: str = ''
    joint_objective: str = 'effect_flow_matching_plus_grouped_action_huber_1_to_1_v1'
    decoder_warmup_steps: int = 0

    def create(self, rng):
        return JointEffectPi(self, nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        observation, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation = dataclasses.replace(observation,
                effect_state=jax.ShapeDtypeStruct((batch_size, 46), jnp.float32),
                effect_actions=jax.ShapeDtypeStruct((batch_size, 30, 44), jnp.float32),
                effect_action_dim=jax.ShapeDtypeStruct((batch_size,), jnp.int32))
        return observation, actions


@dataclasses.dataclass(frozen=True)
class PreserveJointTargets(transforms.DataTransformFn):
    inputs: tuple

    def __call__(self, data):
        extra = {name: data[name] for name in ('effect_state', 'effect_actions', 'effect_action_dim', 'effect_valid') if name in data}
        for transform in self.inputs:
            data = transform(data)
        return {**data, **extra}


@dataclasses.dataclass(frozen=True)
class JointEffectDataConfig(EffectDataConfig):
    def create(self, assets_dirs, model_config):
        base = super().create(assets_dirs, model_config)
        repack = base.repack_transforms.inputs[0]
        fields = {name: name for name in ('effect_state', 'effect_actions', 'effect_action_dim', 'effect_valid')}
        return dataclasses.replace(base, afce_joint_decoder=True,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({**repack.structure, **fields})]),
            data_transforms=dataclasses.replace(base.data_transforms,
                inputs=[PreserveJointTargets(tuple(base.data_transforms.inputs))]))


class JointEffectCacheDataset(EffectCacheDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        state = np.asarray(sample['observation.state'], dtype=np.float32)
        actions = np.asarray(sample['action'], dtype=np.float32)
        action_dim = actions.shape[-1]
        if action_dim not in (22, 44) or state.shape != (23*(action_dim//22),) or actions.shape != (30, action_dim):
            raise ValueError(f'Unexpected raw joint supervision: {state.shape}, {actions.shape}')
        sample['effect_state'] = np.pad(state, (0, 46-len(state)))
        sample['effect_actions'] = np.pad(actions, ((0, 0), (0, 44-action_dim)))
        sample['effect_action_dim'] = np.int32(action_dim)
        episode, frame = int(sample['episode_index']), int(sample['frame_index'])
        sample['effect_valid'] = frame + np.arange(30) < len(self.cache[episode])
        return sample


@dataclasses.dataclass(frozen=True)
class JointEffectWeights:
    params_path: str

    def load(self, params):
        loaded = model_lib.restore_params(self.params_path, restore_type=np.ndarray)
        loaded = {name: value for name, value in loaded.items() if name not in ('action_in_proj', 'action_out_proj')}
        return weight_loaders._merge_params(loaded, params,
            missing_regex=r'.*(lora|action_in_proj|action_out_proj|single_decoder|bimanual_decoder|effect_mean|effect_std).*')


def joint_effect_config(config, cache, base, decoder_init, *, decoder_warmup_steps=0):
    import hashlib
    import json
    cache, decoder_init = Path(cache), Path(decoder_init).resolve()
    manifest = json.loads((cache/'manifest.json').read_text())
    receipt = json.loads(decoder_init.with_suffix('.json').read_text())
    digest = hashlib.sha256(decoder_init.read_bytes()).hexdigest()
    if not manifest.get('complete') or manifest.get('exported_frames') != 523763:
        raise ValueError('Joint policy needs complete all-frame E targets')
    if receipt['codec_sha256'] != manifest['checkpoint_sha256'] or receipt['initialization_sha256'] != digest:
        raise ValueError('Decoder initialization does not match the frozen E encoder')
    if receipt['normalization_sha256'] != hashlib.sha256((cache/'normalization.npz').read_bytes()).hexdigest():
        raise ValueError('Effect normalization changed')
    if decoder_warmup_steps < 0:
        raise ValueError('decoder_warmup_steps must be nonnegative')
    objective = ('effect_only_then_effect_plus_grouped_action_huber_1_to_1_v1'
                 if decoder_warmup_steps else 'effect_flow_matching_plus_grouped_action_huber_1_to_1_v1')
    model = JointEffectPiConfig(**dataclasses.asdict(config.model),
        decoder_init_path=str(decoder_init), decoder_init_sha256=digest,
        decoder_warmup_steps=decoder_warmup_steps, joint_objective=objective)
    data = JointEffectDataConfig(**{field.name: getattr(config.data, field.name) for field in dataclasses.fields(config.data)},
        afce_cache_root=cache)
    return dataclasses.replace(config, model=model, data=data, weight_loader=JointEffectWeights(str(base)))


class JointDecodedEffectPolicy:
    def __init__(self, latent_policy, model, *, task):
        self.policy = latent_policy
        self.state_dim = 46 if task.startswith('bimanual_') else 23
        decoder = model.bimanual_decoder if self.state_dim == 46 else model.single_decoder
        self.decode = jax.jit(lambda effect, state: decoder.physical(
            effect*model.effect_std.value+model.effect_mean.value, state))

    @property
    def metadata(self):
        return self.policy.metadata

    def reset(self):
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def infer(self, observation, **kwargs):
        import time
        state = np.asarray(observation['state'], dtype=np.float32)[:self.state_dim]
        if state.shape != (self.state_dim,):
            raise ValueError('Decoder requires the current unnormalized state')
        started = time.monotonic()
        result = self.policy.infer(observation, **kwargs)
        effect = np.asarray(result['actions'], dtype=np.float32)
        if effect.shape != (30, 256):
            raise ValueError('Joint policy must return the complete 30x256 E before decoding')
        result['actions'] = np.asarray(self.decode(effect[None], state[None]))[0]
        result['policy_timing'] = {'infer_ms': 1000*(time.monotonic()-started)}
        return result
