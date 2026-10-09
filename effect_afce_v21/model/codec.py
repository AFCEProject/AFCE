"""AFCE v2.1 codec: tri-modal mixer → E ∈ R^{H×256}; decoders read only E."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from effect_afce_v21.constants import (
    ACTION_DIM,
    ACTION_HORIZON,
    BRANCH_DIM,
    DELTA_S_DIM,
    EFFECT_DIM,
    LAMBDA_RAMP_STEPS,
    LAMBDA_START,
    LOSS_W_S,
    LOSS_W_V,
    N_WORLD_REGIONS,
    NUM_WORLD_INTERVALS,
    STATE_DIM,
    W_RAW_DIM,
    Y_W_DIM,
)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None):
        super().__init__()
        h = hidden or max(out_dim, in_dim)
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionBranch(nn.Module):
    def __init__(self, action_dim: int = ACTION_DIM, out_dim: int = BRANCH_DIM):
        super().__init__()
        self.proj = MLP(action_dim, out_dim, hidden=64)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.proj(actions)  # (B,H,64)


class RobotBranch(nn.Module):
    def __init__(self, out_dim: int = BRANCH_DIM):
        super().__init__()
        self.root = MLP(9, out_dim, hidden=64)
        self.tip = MLP(3, out_dim, hidden=32)
        self.joint = MLP(16, out_dim, hidden=64)
        self.pool = nn.Linear(out_dim * 3, out_dim)

    def forward(self, root_feat, tip_feat, joint_feat) -> torch.Tensor:
        root = self.root(root_feat)
        tips = self.tip(tip_feat).mean(dim=2)
        joint = self.joint(joint_feat)
        return self.pool(torch.cat([root, tips, joint], dim=-1))


class SparseWorldReadout(nn.Module):
    """Z_W sparse → per-timestep z_i^V via CA(q_i, Z_W)."""

    def __init__(
        self,
        *,
        raw_dim: int = W_RAW_DIM,
        dim: int = BRANCH_DIM,
        horizon: int = ACTION_HORIZON,
        heads: int = 4,
    ):
        super().__init__()
        self.proj = MLP(raw_dim, dim, hidden=128)
        self.query = nn.Parameter(torch.randn(1, horizon, dim) * 0.02)
        self.time_emb = nn.Embedding(horizon, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, w_raw: torch.Tensor, w_valid: torch.Tensor | None = None) -> torch.Tensor:
        b, kw, m, d = w_raw.shape
        z = self.proj(w_raw.reshape(b, kw * m, d))
        q = self.query.expand(b, -1, -1) + self.time_emb.weight.unsqueeze(0)
        key_pad = None
        if w_valid is not None:
            key_pad = ~(w_valid.reshape(b, kw * m) > 0.5)
        out, _ = self.attn(q, z, z, key_padding_mask=key_pad, need_weights=False)
        return out  # (B,H,64)


class TriModalMixer(nn.Module):
    def __init__(self, dim: int = BRANCH_DIM, layers: int = 2, heads: int = 4, ffn: int = 256):
        super().__init__()
        self.type_emb = nn.Embedding(3, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ffn, dropout=0.0, batch_first=True, norm_first=True
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, z_a, z_s, z_v) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, d = z_a.shape
        te = self.type_emb.weight  # (3,D)
        x = torch.stack([z_a + te[0], z_s + te[1], z_v + te[2]], dim=2)  # (B,H,3,D)
        x = x.reshape(b * h, 3, d)
        y = self.tf(x).view(b, h, 3, d)
        return y[:, :, 0], y[:, :, 1], y[:, :, 2]


class TemporalEffect(nn.Module):
    def __init__(self, dim: int = EFFECT_DIM, layers: int = 2, heads: int = 8, ffn: int = 1024, horizon: int = ACTION_HORIZON):
        super().__init__()
        self.pos = nn.Embedding(horizon, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ffn, dropout=0.0, batch_first=True, norm_first=True
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, e0: torch.Tensor) -> torch.Tensor:
        h = e0.shape[1]
        return self.tf(e0 + self.pos.weight[:h].unsqueeze(0))


class ActionDecoder(nn.Module):
    def __init__(
        self,
        *,
        effect_dim: int = EFFECT_DIM,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        horizon: int = ACTION_HORIZON,
        dim: int = 256,
        layers: int = 4,
        heads: int = 8,
        ffn: int = 1024,
    ):
        super().__init__()
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.e_proj = nn.Linear(effect_dim, dim)
        self.query = nn.Parameter(torch.randn(1, horizon, dim) * 0.02)
        self.time_emb = nn.Embedding(horizon, dim)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ffn, dropout=0.0, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.out = nn.Linear(dim, action_dim)

    def forward(self, e: torch.Tensor, state_t: torch.Tensor) -> torch.Tensor:
        b = e.shape[0]
        s = self.state_mlp(state_t).unsqueeze(1)
        ctx = torch.cat([s, self.e_proj(e)], dim=1)
        q = self.query.expand(b, -1, -1) + self.time_emb.weight.unsqueeze(0)
        return self.out(self.decoder(q, ctx))


class ResponseDecoder(nn.Module):
    def __init__(self, effect_dim: int = EFFECT_DIM, out_dim: int = DELTA_S_DIM):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(effect_dim, effect_dim), nn.GELU(), nn.Linear(effect_dim, out_dim))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.mlp(e)


class WorldDecoder(nn.Module):
    def __init__(self, effect_dim: int = EFFECT_DIM, query_dim: int = W_RAW_DIM, out_dim: int = Y_W_DIM):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, effect_dim)
        self.attn = nn.MultiheadAttention(effect_dim, 8, batch_first=True)
        self.out = nn.Sequential(nn.Linear(effect_dim, effect_dim), nn.GELU(), nn.Linear(effect_dim, out_dim))

    def forward(self, e: torch.Tensor, w_raw: torch.Tensor) -> torch.Tensor:
        b, kw, m, qd = w_raw.shape
        q = self.q_proj(w_raw.reshape(b, kw * m, qd))
        h, _ = self.attn(q, e, e, need_weights=False)
        return self.out(h).view(b, kw, m, -1)


class AFCECodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_branch = ActionBranch()
        self.robot_branch = RobotBranch()
        self.world_readout = SparseWorldReadout()
        self.mixer = TriModalMixer()
        self.p_e = nn.Linear(BRANCH_DIM * 3, EFFECT_DIM)
        self.temporal = TemporalEffect()
        self.d_a = ActionDecoder()
        self.d_s = ResponseDecoder()
        self.d_v = WorldDecoder()

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        *,
        drop_a: bool = False,
        drop_s: bool = False,
        drop_v: bool = False,
    ) -> dict[str, torch.Tensor]:
        z_a = self.action_branch(batch["action"])
        z_s = self.robot_branch(batch["root_feat"], batch["tip_feat"], batch["joint_feat"])
        z_v = self.world_readout(batch["W_raw"], batch.get("W_valid"))
        if drop_a:
            z_a = torch.zeros_like(z_a)
        if drop_s:
            z_s = torch.zeros_like(z_s)
        if drop_v:
            z_v = torch.zeros_like(z_v)
        ta, ts, tv = self.mixer(z_a, z_s, z_v)
        e0 = self.p_e(torch.cat([ta, ts, tv], dim=-1))
        e = self.temporal(e0)
        return {"Z_A": z_a, "Z_S": z_s, "Z_V": z_v, "E0": e0, "E": e}

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        drop_a: bool = False,
        drop_s: bool = False,
        drop_v: bool = False,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode(batch, drop_a=drop_a, drop_s=drop_s, drop_v=drop_v)
        return {
            **enc,
            "action_pred": self.d_a(enc["E"], batch["state_t"]),
            "delta_s_pred": self.d_s(enc["E"]),
            "y_w_pred": self.d_v(enc["E"], batch["W_raw"]),
        }


def grouped_action_loss(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    def _g(a, b):
        return F.smooth_l1_loss(a, b, reduction="mean", beta=1.0)

    l_xyz = _g(pred[..., :3], target[..., :3])
    l_rot = _g(pred[..., 3:6], target[..., 3:6])
    l_hand = _g(pred[..., 6:], target[..., 6:])
    l_a = (l_xyz + l_rot + l_hand) / 3.0
    with torch.no_grad():
        pos_l2 = torch.linalg.norm(pred[..., :3] - target[..., :3], dim=-1).mean()
    return {"loss_action": l_a, "action_pos_l2": pos_l2}


def world_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid > 0.5
    if mask.sum() == 0:
        return pred.new_zeros(())
    p, t = pred[mask], target[mask]
    return F.smooth_l1_loss(p[..., :64], t[..., :64], beta=1.0) + F.smooth_l1_loss(
        p[..., 64:], t[..., 64:], beta=1.0
    )


def afce_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_s: float = LOSS_W_S,
    lambda_v: float = LOSS_W_V,
) -> dict[str, torch.Tensor]:
    ga = grouped_action_loss(out["action_pred"], batch["action"])
    l_s = F.smooth_l1_loss(out["delta_s_pred"], batch["delta_s"], beta=1.0)
    l_v = world_loss(out["y_w_pred"], batch["Y_W"], batch["W_valid"])
    loss = ga["loss_action"] + float(lambda_s) * l_s + float(lambda_v) * l_v
    return {
        "loss": loss,
        "loss_action": ga["loss_action"],
        "loss_s": l_s,
        "loss_v": l_v,
        "action_pos_l2": ga["action_pos_l2"],
    }


def schedule_lambdas(step: int, ramp: int = LAMBDA_RAMP_STEPS, lam0: float = LAMBDA_START, lam1: float = LOSS_W_S) -> tuple[float, float]:
    if step >= ramp:
        return float(lam1), float(lam1)
    alpha = float(step) / float(max(ramp, 1))
    lam = lam0 + (lam1 - lam0) * alpha
    return lam, lam
