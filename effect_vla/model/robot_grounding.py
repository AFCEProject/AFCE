"""Robot grounding: (Ê, S_t) → G, plus a geometry head for wrist/fingertip targets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from effect_vla.constants import EFFECT_DIM, GEO_DIM, GROUNDING_DIM, NUM_ANCHORS
from effect_vla.model.effect_query import CrossAttention


class RobotGrounding(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        rngs: nnx.Rngs,
        *,
        k: int = NUM_ANCHORS,
        effect_dim: int = EFFECT_DIM,
        grounding_dim: int = GROUNDING_DIM,
        geo_dim: int = GEO_DIM,
    ):
        self.k = k
        self.state_fc1 = nnx.Linear(state_dim, grounding_dim, rngs=rngs)
        self.state_fc2 = nnx.Linear(grounding_dim, grounding_dim, rngs=rngs)
        self.effect_proj = nnx.Linear(effect_dim, grounding_dim, rngs=rngs)
        self.queries = nnx.Param(jax.random.normal(rngs.params(), (k, grounding_dim)) * 0.02)
        self.cross = CrossAttention(grounding_dim, num_heads=8, rngs=rngs)
        self.geo_fc1 = nnx.Linear(grounding_dim, grounding_dim, rngs=rngs)
        self.geo_fc2 = nnx.Linear(grounding_dim, geo_dim, rngs=rngs)

    def __call__(self, effect_hat, state):
        """effect_hat: (B, K, d_E), state: (B, S). Returns G (B,K,d_G) and geo (B,K,geo)."""
        b = state.shape[0]
        h_s = nnx.swish(self.state_fc2(nnx.swish(self.state_fc1(state))))[:, None, :]
        e = self.effect_proj(effect_hat)
        ctx = jnp.concatenate([e, h_s], axis=1)
        q = jnp.broadcast_to(jnp.asarray(self.queries)[None, :, :], (b, self.k, e.shape[-1]))
        g = self.cross(q, ctx)
        geo = self.geo_fc2(nnx.swish(self.geo_fc1(g)))
        return g, geo
