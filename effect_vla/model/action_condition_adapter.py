"""Zero-init adapters that inject Ê / G into the Action Expert context.

At step 0: ActionExpert(H, E, G) ≈ ActionExpert(H) because the output
projection is all zeros.
"""

from __future__ import annotations

from flax import nnx


def _zero_linear(in_dim: int, out_dim: int, rngs: nnx.Rngs) -> nnx.Linear:
    return nnx.Linear(
        in_dim,
        out_dim,
        kernel_init=nnx.initializers.zeros,
        bias_init=nnx.initializers.zeros,
        rngs=rngs,
    )


class ActionConditionAdapter(nnx.Module):
    def __init__(self, in_dim: int, action_width: int, rngs: nnx.Rngs):
        self.proj = _zero_linear(in_dim, action_width, rngs)

    def __call__(self, tokens):
        return self.proj(tokens)
