"""Learnable Effect queries over π0.5 VLM prefix tokens.

Q_E = [q1, q2]
Z_E = CrossAttn(Q_E, H_VLM)
Ê   = MLP(Z_E)  → (B, K, 512)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from effect_vla.constants import EFFECT_DIM, NUM_ANCHORS


class CrossAttention(nnx.Module):
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs):
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, query, context, context_mask=None):
        b, qn, _ = query.shape
        n = context.shape[1]
        h = self.num_heads
        d = self.head_dim
        q = self.q_proj(query).reshape(b, qn, h, d).transpose(0, 2, 1, 3)
        k = self.k_proj(context).reshape(b, n, h, d).transpose(0, 2, 1, 3)
        v = self.v_proj(context).reshape(b, n, h, d).transpose(0, 2, 1, 3)
        scale = d ** -0.5
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if context_mask is not None:
            mask = context_mask[:, None, None, :].astype(logits.dtype)
            logits = jnp.where(mask > 0, logits, jnp.asarray(-1e9, dtype=logits.dtype))
        attn = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, qn, h * d)
        return self.out_proj(out)


class EffectPredictor(nnx.Module):
    def __init__(self, vlm_dim: int, rngs: nnx.Rngs, *, k: int = NUM_ANCHORS, effect_dim: int = EFFECT_DIM):
        self.k = k
        self.queries = nnx.Param(jax.random.normal(rngs.params(), (k, vlm_dim)) * 0.02)
        heads = 8 if vlm_dim % 8 == 0 else 1
        self.cross = CrossAttention(vlm_dim, num_heads=heads, rngs=rngs)
        self.fc1 = nnx.Linear(vlm_dim, vlm_dim, rngs=rngs)
        self.fc2 = nnx.Linear(vlm_dim, effect_dim, rngs=rngs)

    def __call__(self, h_vlm, h_mask=None):
        b = h_vlm.shape[0]
        q = jnp.broadcast_to(jnp.asarray(self.queries)[None, :, :], (b, self.k, h_vlm.shape[-1]))
        z = self.cross(q, h_vlm, h_mask)
        e = self.fc2(nnx.swish(self.fc1(z)))
        e = e / jnp.maximum(jnp.linalg.norm(e, axis=-1, keepdims=True), 1e-6)
        return e
