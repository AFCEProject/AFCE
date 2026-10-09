"""Verify saved policy checkpoints contain updated decoders and frozen normalization."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from afce.jax_action_decoder import load_arrays
from afce.prepare import atomic_json
from openpi.models.model import restore_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--decoder-init', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    initial = load_arrays(args.decoder_init)
    params = restore_params(args.checkpoint/'params', restore_type=np.ndarray)
    if any(name.startswith('adaptive_') for name in params):
        raise AssertionError('Single-decoder checkpoint unexpectedly contains adaptive D1')
    report = {'passed': False, 'checkpoint': str(args.checkpoint.resolve()),
              'decoder_init_sha256': hashlib.sha256(args.decoder_init.read_bytes()).hexdigest(), 'decoders': {}}
    for name, action_dim in (('single_decoder', 22), ('bimanual_decoder', 44)):
        decoder = params[name]
        differences = []
        for parameter, value in decoder['weights'].items():
            before = initial[f'decoder_{action_dim}/'+parameter]
            delta = float(np.max(np.abs(np.asarray(value)-before)))
            if delta > 0:
                differences.append({'parameter': parameter, 'max_abs_update': delta})
        if not differences:
            raise AssertionError(f'{name} has no saved parameter updates')
        for field in ('state_mean', 'state_std', 'action_mean', 'action_std'):
            np.testing.assert_array_equal(np.asarray(decoder[field]), initial[f'{field}_{action_dim}'])
        report['decoders'][name] = {'changed_parameter_tensors': len(differences),
            'max_abs_update': max(row['max_abs_update'] for row in differences), 'normalization_unchanged': True}
    for name in ('effect_mean', 'effect_std'):
        np.testing.assert_array_equal(np.asarray(params[name]), initial[name])
    if any('encoder' in name for name in params if name != 'PaliGemma'):
        raise AssertionError('Unexpected codec encoder parameters inside the policy checkpoint')
    report.update(passed=True, effect_normalization_unchanged=True, codec_encoder_not_in_policy=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
