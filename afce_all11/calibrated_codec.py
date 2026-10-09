"""Versioned codec with train-only input and output standardization.

All public tensors stay in physical units. Decoders internally predict centered,
unit-scale values; constant relative-rotation components no longer amplify the
gradient through division by a 1e-3 physical standard deviation.
"""
from __future__ import annotations
import numpy as np
import torch
from afce_all11.codec import Codec


def fit_statistics(data):
    stats = data.action_stats()
    for d in (22, 44):
        episodes = [ep for ep in data.episodes if ep['actions'].shape[-1] == d]
        for key, source in [('action', 'actions'), ('state', 'states')]:
            x = np.concatenate([ep[source] for ep in episodes]).astype(np.float64)
            stats[str(d)][key+'_mean'] = torch.from_numpy(x.mean(0).astype(np.float32))
            stats[str(d)][key+'_std'] = torch.from_numpy(x.std(0).clip(1e-3).astype(np.float32))
    response = {22: [], 44: []}
    for i in np.random.default_rng(192).integers(0, len(data), 2048):
        b = data.sample(int(i)); response[b['action'].shape[-1]].append(b['response'][b['response_valid'] > 0])
    for d, arrays in response.items():
        x = np.concatenate(arrays).astype(np.float64)
        stats[str(d)]['response_mean'] = torch.from_numpy(x.mean(0).astype(np.float32))
        stats[str(d)]['response_std'] = torch.from_numpy(x.std(0).clip(1e-3).astype(np.float32))
    count = 0; total = np.zeros(136, np.float64); squared = total.copy()
    for ep in data.episodes:
        with np.load(data.evidence/ep['task']/f"episode_{ep['episode']:06d}"/'world_intervals.npz') as z:
            x = z['raw'][z['valid'] > 0].astype(np.float64)
            count += len(x); total += x.sum(0); squared += np.square(x).sum(0)
    if not count: raise ValueError('No valid visual targets in training data')
    mean = total/count; std = np.sqrt(np.maximum(squared/count-mean*mean, 0)).clip(1e-3)
    # These columns are recomputed relative to each sampled window.
    mean[70:72] = 0.; std[70:72] = 1.
    stats['world'] = {'mean': torch.from_numpy(mean.astype(np.float32)),
                      'std': torch.from_numpy(std.astype(np.float32)), 'valid_regions': count}
    return stats


class CalibratedCodec(Codec):
    def __init__(self, fusion, conditioning, statistics):
        super().__init__(fusion, conditioning)
        for d in (22, 44):
            for key in ('action', 'state', 'response'):
                for kind in ('mean', 'std'):
                    self.register_buffer(f'{key}_{kind}_{d}', statistics[str(d)][key+'_'+kind].clone())
        self.register_buffer('world_mean', statistics['world']['mean'].clone())
        self.register_buffer('world_std', statistics['world']['std'].clone())

    def encode(self, batch, drop=None):
        d = batch['action'].shape[-1]
        b = dict(batch)
        b['action'] = (batch['action']-getattr(self, f'action_mean_{d}'))/getattr(self, f'action_std_{d}')
        # Reassemble each hand's response without depending on the target field.
        normalized = []
        for hand in range(d//22):
            r = torch.cat([batch['root'][..., hand*9:(hand+1)*9],
                           batch['tips'][..., hand*4:(hand+1)*4, :].flatten(-2),
                           batch['joints'][..., hand*16:(hand+1)*16]], -1)
            sl = slice(hand*37, (hand+1)*37)
            normalized.append((r-getattr(self, f'response_mean_{d}')[sl])/getattr(self, f'response_std_{d}')[sl])
        b['root'] = torch.cat([r[..., :9] for r in normalized], -1)
        b['tips'] = torch.cat([r[..., 9:21].unflatten(-1, (4, 3)) for r in normalized], -2)
        b['joints'] = torch.cat([r[..., 21:37] for r in normalized], -1)
        b['world'] = (batch['world']-self.world_mean)/self.world_std
        return super().encode(b, drop)

    def decode(self, e, state):
        d = 22 if state.shape[-1] == 23 else 44
        s = (state-getattr(self, f'state_mean_{d}'))/getattr(self, f'state_std_{d}')
        a = self.action_decoders[str(d)](e, s)
        return a*getattr(self, f'action_std_{d}')+getattr(self, f'action_mean_{d}')

    def decode_outputs(self, e, batch):
        d = batch['action'].shape[-1]
        response = self.response_decoders[str(d)](e)
        world = self.world_decoder(e, batch['world_query'])
        return {'E': e, 'action': self.decode(e, batch['state']),
                'response': response*getattr(self, f'response_std_{d}')+getattr(self, f'response_mean_{d}'),
                'world': world*self.world_std[:66]+self.world_mean[:66]}

    def forward(self, batch, drop=None):
        return self.decode_outputs(self.encode(batch, drop), batch)
