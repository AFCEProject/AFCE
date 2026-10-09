"""Differentiable action geometry, using the actual Allegro hand transforms."""
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
import torch.nn.functional as F

from afce.evidence import hand_model


def rotvec_matrix(vector):
    """Rodrigues with finite values and derivatives at the identity."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(vector.shape[:-1]+(3, 3))
    angle = torch.linalg.vector_norm(vector, dim=-1)
    a = torch.sinc(angle/torch.pi)[..., None, None]
    b = .5*torch.sinc(angle/(2*torch.pi)).square()[..., None, None]
    return torch.eye(3, device=vector.device, dtype=vector.dtype)+a*skew+b*(skew@skew)


def matvec(matrix, vector):
    return (matrix@vector[..., None]).squeeze(-1)


class HandKinematics(nn.Module):
    """Finger collision-center positions in the controller's attachment-site frame."""
    def __init__(self, side):
        super().__init__()
        model, qpos_ids, tips, site = hand_model(side)
        root = int(model.site_bodyid[site])
        needed = set()
        for geom in tips:
            body = int(model.geom_bodyid[geom])
            while body != root:
                if body == 0:
                    raise ValueError('Fingertip is not a descendant of attachment site')
                needed.add(body); body = int(model.body_parentid[body])
        self.root = root; self.nodes = []
        self.tip_bodies = [int(model.geom_bodyid[g]) for g in tips]

        def buffer(name, value):
            self.register_buffer(name, torch.tensor(np.asarray(value).copy(), dtype=torch.float64))

        def quat_matrix(q):
            return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()

        for body in sorted(needed):
            buffer(f'p{body}', model.body_pos[body])
            buffer(f'r{body}', quat_matrix(model.body_quat[body]))
            count = int(model.body_jntnum[body]); index = None
            if count:
                if count != 1:
                    raise ValueError('Only one hinge per finger body is supported')
                joint = int(model.body_jntadr[body])
                if int(model.jnt_type[joint]) != 3:
                    raise ValueError('Expected hinge joint')
                qpos = int(model.jnt_qposadr[joint])
                index = list(qpos_ids).index(qpos)
                buffer(f'axis{body}', model.jnt_axis[joint])
                buffer(f'anchor{body}', model.jnt_pos[joint])
                buffer(f'ref{body}', model.qpos0[qpos])
            self.nodes.append((body, int(model.body_parentid[body]), index))
        buffer('tip_local', model.geom_pos[tips])
        buffer('site_position', model.site_pos[site])
        buffer('site_rotation', quat_matrix(model.site_quat[site]))

    def forward(self, joints):
        if joints.shape[-1] != 16:
            raise ValueError('Expected ff/mf/rf/th joint order, four hinges each')
        shape = joints.shape[:-1]
        poses = {self.root: (torch.zeros(shape+(3,), dtype=joints.dtype, device=joints.device),
                            torch.eye(3, dtype=joints.dtype, device=joints.device).expand(shape+(3, 3)))}
        for body, parent, index in self.nodes:
            pp, pr = poses[parent]
            position = pp+matvec(pr, getattr(self, f'p{body}'))
            rotation = pr@getattr(self, f'r{body}')
            if index is not None:
                angle = joints[..., index]-getattr(self, f'ref{body}')
                delta = rotvec_matrix(angle[..., None]*getattr(self, f'axis{body}'))
                anchor = getattr(self, f'anchor{body}')
                position = position+matvec(rotation, anchor-matvec(delta, anchor))
                rotation = rotation@delta
            poses[body] = position, rotation
        tips = torch.stack([poses[body][0]+matvec(poses[body][1], self.tip_local[i])
                            for i, body in enumerate(self.tip_bodies)], -2)
        return (tips-self.site_position)@self.site_rotation


class ActionGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.hands = nn.ModuleDict({side: HandKinematics(side) for side in ('single', 'right', 'left')})

    def forward(self, action):
        if action.shape[-1] not in (22, 44):
            raise ValueError('Official rotvec actions must be A22 or A44')
        sides = ('single',) if action.shape[-1] == 22 else ('right', 'left')
        tips = []
        for i, side in enumerate(sides):
            a = action[..., i*22:(i+1)*22]
            local = self.hands[side](a[..., 6:22])
            tips.append(local@rotvec_matrix(a[..., 3:6]).transpose(-1, -2)+a[..., None, :3])
        return torch.cat(tips, -2)


def rotation_objective(pred, target):
    # Chordal SO(3) distance, invariant to equivalent rotation-vector representations.
    return torch.stack([(rotvec_matrix(pred[..., off+3:off+6])-rotvec_matrix(target[..., off+3:off+6]))
                        .square().sum((-1, -2))/6 for off in range(0, pred.shape[-1], 22)]).mean()


def relative_pose(action):
    if action.shape[-1] != 44:
        raise ValueError('Relative hand pose requires bimanual A44')
    right = rotvec_matrix(action[..., 3:6]); left = rotvec_matrix(action[..., 25:28])
    return matvec(right.transpose(-1, -2), action[..., 22:25]-action[..., :3]), right.transpose(-1, -2)@left


def relative_objective(pred, target):
    if pred.shape[-1] == 22:
        return pred.sum()*0
    pp, pr = relative_pose(pred); tp, tr = relative_pose(target)
    return .5*(F.smooth_l1_loss(pp/.05, tp/.05)+(pr-tr).square().sum((-1, -2)).mean()/6)


def temporal_objective(pred, target, scale):
    # Preserve demonstrated motion; no penalty for a correctly reproduced abrupt contact.
    # Rotation-vector differences are excluded because of the representation's branch cut.
    groups = []
    for off in range(0, pred.shape[-1], 22):
        for lo, hi in ((0, 3), (6, 22)):
            sl = slice(off+lo, off+hi)
            groups.append(F.smooth_l1_loss(torch.diff(pred[..., sl], dim=-2)/scale[sl],
                                          torch.diff(target[..., sl], dim=-2)/scale[sl]))
    return torch.stack(groups).mean()


def auxiliary_objectives(pred, target, scale, config, geometry=None):
    values = {}
    if config['fingertip_weight']:
        if geometry is None:
            raise ValueError('Fingertip loss requires verified differentiable kinematics')
        values['fingertip'] = F.smooth_l1_loss(geometry(pred)/.05, geometry(target)/.05)
    if config['relative_weight']:
        values['relative'] = relative_objective(pred, target)
    if config['temporal_weight']:
        values['temporal'] = temporal_objective(pred, target, scale)
    if config['rotation_weight']:
        values['rotation'] = rotation_objective(pred, target)
    return values
