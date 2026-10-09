"""Differentiable JAX equivalent of the pretrained, complete action decoder."""
from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


class ActionDecoder(nnx.Module):
    def __init__(self, arrays, action_dim):
        self.action_dim = action_dim
        prefix = f'decoder_{action_dim}/'
        self.weights = {
            name[len(prefix):]: nnx.Param(jnp.asarray(value, dtype=jnp.float32))
            for name, value in arrays.items() if name.startswith(prefix)
        }
        if not self.weights or self.weights['out.weight'].value.shape != (action_dim, 256):
            raise ValueError('Unexpected decoder output shape')
        for name in ('state_mean', 'state_std', 'action_mean', 'action_std'):
            setattr(self, name, nnx.Variable(jnp.asarray(arrays[f'{name}_{action_dim}'], dtype=jnp.float32)))

    def linear(self, values, name):
        weight = self.weights[name+'.weight'].value
        bias = self.weights[name+'.bias'].value
        return jnp.matmul(values, weight.T, precision=jax.lax.Precision.HIGHEST)+bias

    def norm(self, values, name):
        mean = jnp.mean(values, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(values-mean), axis=-1, keepdims=True)
        return (values-mean)*jax.lax.rsqrt(variance+1e-5)*self.weights[name+'.weight'].value+self.weights[name+'.bias'].value

    def attention(self, queries, memory, name):
        weight = self.weights[name+'.in_proj_weight'].value
        bias = self.weights[name+'.in_proj_bias'].value
        width = queries.shape[-1]
        query = jnp.matmul(queries, weight[:width].T, precision=jax.lax.Precision.HIGHEST)+bias[:width]
        key = jnp.matmul(memory, weight[width:2*width].T, precision=jax.lax.Precision.HIGHEST)+bias[width:2*width]
        value = jnp.matmul(memory, weight[2*width:].T, precision=jax.lax.Precision.HIGHEST)+bias[2*width:]
        heads = 8
        head_dim = width//heads
        query = query.reshape(query.shape[:-1]+(heads, head_dim)).transpose(0, 2, 1, 3)
        key = key.reshape(key.shape[:-1]+(heads, head_dim)).transpose(0, 2, 1, 3)
        value = value.reshape(value.shape[:-1]+(heads, head_dim)).transpose(0, 2, 1, 3)
        scores = jnp.matmul(query/head_dim**.5, key.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST)
        attended = jnp.matmul(jax.nn.softmax(scores, axis=-1), value, precision=jax.lax.Precision.HIGHEST)
        attended = attended.transpose(0, 2, 1, 3).reshape(queries.shape)
        return self.linear(attended, name+'.out_proj')

    def __call__(self, effect, raw_state):
        state = (raw_state.astype(jnp.float32)-self.state_mean.value)/self.state_std.value
        state = self.norm(jax.nn.gelu(self.linear(state, 'state_mlp.0'), approximate=False), 'state_mlp.2')
        memory = jnp.concatenate([state[:, None], self.linear(effect.astype(jnp.float32), 'e_proj')], axis=1)
        queries = self.weights['query'].value+self.weights['time_emb.weight'].value[None]
        queries = jnp.broadcast_to(queries, (effect.shape[0], 30, 256))
        for index in range(4):
            prefix = f'decoder.layers.{index}'
            normalized = self.norm(queries, prefix+'.norm1')
            queries = queries+self.attention(normalized, normalized, prefix+'.self_attn')
            queries = queries+self.attention(self.norm(queries, prefix+'.norm2'), memory, prefix+'.multihead_attn')
            hidden = jax.nn.relu(self.linear(self.norm(queries, prefix+'.norm3'), prefix+'.linear1'))
            queries = queries+self.linear(hidden, prefix+'.linear2')
        return self.linear(queries, 'out')

    def physical(self, effect, raw_state):
        return self(effect, raw_state)*self.action_std.value+self.action_mean.value


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def grouped_action_loss(predicted_normalized, target_physical, decoder):
    target = (target_physical-decoder.action_mean.value)/decoder.action_std.value
    groups = []
    for offset in range(0, decoder.action_dim, 22):
        for start, stop in ((0, 3), (3, 6), (6, 22)):
            error = jnp.abs(predicted_normalized[..., offset+start:offset+stop]-target[..., offset+start:offset+stop])
            huber = jnp.where(error < 1., .5*jnp.square(error), error-.5)
            groups.append(jnp.mean(huber, axis=(-1, -2)))
    return jnp.mean(jnp.stack(groups), axis=0)
