"""Extract the frozen codec's pretrained action decoders for joint policy training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from afce.prepare import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--effect-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = args.effect_cache.resolve()
    manifest = json.loads((cache/'manifest.json').read_text())
    checkpoint = cache/'codec.pt'
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not manifest['complete'] or checkpoint_sha != manifest['checkpoint_sha256']:
        raise ValueError('Frozen E cache and codec do not match')
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if blob['config']['fusion'] != 'query' or blob['config'].get('aligned_residual', False):
        raise ValueError('This version requires the selected Query codec without an aligned residual head')
    arrays = {}
    for action_dim in (22, 44):
        prefix = f'action_decoders.{action_dim}.'
        for name, value in blob['model'].items():
            if name.startswith(prefix):
                arrays[f'decoder_{action_dim}/'+name[len(prefix):]] = value.detach().cpu().numpy()
        for name in ('state_mean', 'state_std', 'action_mean', 'action_std'):
            arrays[f'{name}_{action_dim}'] = blob['model'][f'{name}_{action_dim}'].detach().cpu().numpy()
    with np.load(cache/'normalization.npz', allow_pickle=False) as normalization:
        arrays['effect_mean'] = normalization['mean'].copy()
        arrays['effect_std'] = normalization['std'].copy()
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite decoder initialization')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        with np.load(args.output, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[name], value) for name, value in arrays.items()):
                raise ValueError('Existing initialization differs; choose a new output')
    else:
        with args.output.open('xb') as stream:
            np.savez(stream, **arrays)
    receipt = {'codec_sha256': checkpoint_sha, 'codec_step': blob['step'], 'fusion': 'query',
               'initialization_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
               'normalization_sha256': hashlib.sha256((cache/'normalization.npz').read_bytes()).hexdigest(),
               'independent_action_dims': [22, 44], 'parameter_arrays': len(arrays)}
    atomic_json(args.output.with_suffix('.json'), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
