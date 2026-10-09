"""π0.5 + Task Effect + Robot Grounding.

Keeps the original Flow Matching Action Expert. Extra E/G tokens are
prepended to the suffix through zero-init adapters so the initial policy
matches the loaded DexJoCo π0.5 checkpoint.
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at

from effect_vla.constants import EFFECT_DIM, GEO_DIM, GROUNDING_DIM, LAMBDA_EFFECT, LAMBDA_GROUNDING, NUM_ANCHORS
from effect_vla.loss.effect_loss import effect_loss_jax
from effect_vla.loss.grounding_loss import grounding_loss_jax
from effect_vla.model.action_condition_adapter import ActionConditionAdapter
from effect_vla.model.effect_query import EffectPredictor
from effect_vla.model.robot_grounding import RobotGrounding


class Pi0EffectPolicy(_pi0.Pi0):
    """Subclass of the DexJoCo π0.5 model; new params are isolated by name prefix."""

    def __init__(self, config: pi0_config.Pi0Config, rngs):
        super().__init__(config, rngs)
        import openpi.models.gemma as _gemma

        paligemma_width = _gemma.get_config(config.paligemma_variant).width
        action_width = _gemma.get_config(config.action_expert_variant).width
        k = int(getattr(config, "num_effect_anchors", NUM_ANCHORS))
        effect_dim = int(getattr(config, "effect_dim", EFFECT_DIM))
        grounding_dim = int(getattr(config, "grounding_dim", GROUNDING_DIM))
        geo_dim = int(getattr(config, "geo_dim", GEO_DIM))
        state_dim = int(config.action_dim if not config.structured_hand_state else 46)

        self.effect_k = k
        self.condition_effect = bool(getattr(config, "condition_effect", True))
        self.use_grounding = bool(getattr(config, "use_grounding", True))
        self.lambda_effect = float(getattr(config, "lambda_effect", LAMBDA_EFFECT))
        self.lambda_grounding = float(getattr(config, "lambda_grounding", LAMBDA_GROUNDING))

        self.effect_predictor = EffectPredictor(paligemma_width, rngs, k=k, effect_dim=effect_dim)
        self.robot_grounding = RobotGrounding(
            state_dim, rngs, k=k, effect_dim=effect_dim, grounding_dim=grounding_dim, geo_dim=geo_dim
        )
        self.effect_act_adapter = ActionConditionAdapter(effect_dim, action_width, rngs)
        self.grounding_act_adapter = ActionConditionAdapter(grounding_dim, action_width, rngs)
        # Eval-only: "zero" | "shuffle" | None. Training never sets this.
        self.intervention_mode = None

    def predict_effect_and_grounding(self, prefix_tokens, prefix_mask, state):
        effect_hat = self.effect_predictor(prefix_tokens, prefix_mask)
        g_lat, g_geo = self.robot_grounding(effect_hat, state)
        return effect_hat, g_lat, g_geo

    def _condition_tokens(self, effect_hat, g_lat):
        e_tok = self.effect_act_adapter(effect_hat)
        g_tok = self.grounding_act_adapter(g_lat)
        if not self.condition_effect:
            e_tok = jnp.zeros_like(e_tok)
        if not self.use_grounding:
            g_tok = jnp.zeros_like(g_tok)
        tokens = jnp.concatenate([e_tok, g_tok], axis=1)
        return tokens

    def embed_suffix_with_condition(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        condition_tokens,
    ):
        tokens, input_mask, ar_mask, adarms_cond = super().embed_suffix(obs, noisy_actions, timestep)
        b = condition_tokens.shape[0]
        n_extra = condition_tokens.shape[1]
        extra_mask = jnp.ones((b, n_extra), dtype=jnp.bool_)
        extra_ar = jnp.array([True] + [False] * (n_extra - 1), dtype=jnp.bool_)
        # Prepend E/G tokens so action tokens can attend to them. Insert before
        # the trailing action-horizon block by concatenating at the front of
        # the suffix (after optional structured-state tokens would require
        # splitting; putting them first is equivalent for FM decoding because
        # the action head always reads the last H tokens).
        tokens = jnp.concatenate([condition_tokens, tokens], axis=1)
        input_mask = jnp.concatenate([extra_mask, input_mask], axis=1)
        ar_mask = jnp.concatenate([extra_ar, ar_mask], axis=0)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        total, _metrics = self.compute_loss_and_metrics(rng, observation, actions, train=train)
        # Broadcast scalar total into the per-step FM shape so existing
        # train_step code (`mean(chunked_loss)`) stays valid.
        return jnp.broadcast_to(total, actions.shape[:-1])

    def compute_loss_and_metrics(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        effect_hat, g_lat, g_geo = self.predict_effect_and_grounding(
            prefix_tokens, prefix_mask, observation.state
        )
        cond = self._condition_tokens(effect_hat, g_lat)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_with_condition(
            observation, x_t, time, cond
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        fm = jnp.mean(jnp.square(v_t - u_t))

        metrics = {"loss_fm": fm}
        total = fm
        if observation.effect_target is not None:
            l_e, e_m = effect_loss_jax(effect_hat, observation.effect_target)
            total = total + self.lambda_effect * l_e
            metrics["loss_effect"] = l_e
            metrics.update(e_m)
        if self.use_grounding and observation.grounding_target is not None:
            l_g, g_m = grounding_loss_jax(g_geo, observation.grounding_target)
            total = total + self.lambda_grounding * l_g
            metrics["loss_grounding"] = l_g
            metrics.update(g_m)
        metrics["loss"] = total
        return total, metrics

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        effect_hat, g_lat, _g_geo = self.predict_effect_and_grounding(
            prefix_tokens, prefix_mask, observation.state
        )
        if self.intervention_mode == "zero":
            effect_hat = jnp.zeros_like(effect_hat)
        elif self.intervention_mode == "shuffle":
            effect_hat = jnp.roll(effect_hat, 1, axis=0)
        cond = self._condition_tokens(effect_hat, g_lat)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_with_condition(
                observation, x_t, jnp.broadcast_to(time, batch_size), cond
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_s = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_s, suffix_attn_mask], axis=-1)
            positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_s,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond_fn(carry):
            _x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond_fn, step, (noise, 1.0))
        return x_0
