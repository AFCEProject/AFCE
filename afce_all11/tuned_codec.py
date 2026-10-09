"""Optional time-aligned residual decoder and physical state perturbations."""
from __future__ import annotations

import torch
from torch import nn

from afce_all11.calibrated_codec import CalibratedCodec


class TunedCodec(CalibratedCodec):
    def __init__(self, fusion, conditioning, statistics, aligned_residual=False):
        super().__init__(fusion, conditioning, statistics)
        self.aligned_residual = aligned_residual
        if aligned_residual:
            # Local RNG makes new heads identical across fusion architectures.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(73042)
                self.aligned_heads = nn.ModuleDict({str(d): nn.Sequential(
                    nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d)
                ) for d in (22, 44)})
                for head in self.aligned_heads.values():
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)

    def decode(self, e, state):
        action = super().decode(e, state)
        if self.aligned_residual:
            d = action.shape[-1]
            action = action + self.aligned_heads[str(d)](e) * getattr(self, f'action_std_{d}')
        return action


def perturb_state(state, generator=None, position_std=.002, rotation_std_deg=.5, joint_std_deg=.5):
    """Perturb official [poseR, poseL, jointsR, jointsL] state, retaining unit rotations."""
    if state.shape[-1] not in (23, 46):
        raise ValueError('Expected official 23D or 46D state')
    out = state.clone()
    radians = torch.pi / 180
    def noise(shape):
        return torch.randn(shape, dtype=state.dtype, device=state.device, generator=generator).clamp(-3, 3)
    layout = [(0, 7)] if state.shape[-1] == 23 else [(0, 14), (7, 30)]
    for off, joints in layout:
        out[..., off:off+3] += position_std * noise(state[..., off:off+3].shape)
        rv = rotation_std_deg * radians * noise(state[..., off:off+3].shape)
        angle = rv.norm(dim=-1, keepdim=True)
        dw = torch.cos(angle/2)
        dv = .5 * torch.sinc(angle/(2*torch.pi)) * rv
        q = state[..., off+3:off+7]
        qw, qv = q[..., :1], q[..., 1:]
        new = torch.cat([dw*qw - (dv*qv).sum(-1, keepdim=True),
                         dw*qv + qw*dv + torch.cross(dv, qv, dim=-1)], -1)
        out[..., off+3:off+7] = new / new.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out[..., joints:joints+16] += joint_std_deg * radians * noise(state[..., joints:joints+16].shape)
    return out
