"""Compare JAX decoding and gradients against the original pretrained Torch decoder."""
import argparse
import hashlib
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as functional

from afce.export_effect import load_codec
from afce.jax_action_decoder import ActionDecoder, grouped_action_loss, load_arrays
from afce.prepare import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--effect-cache', type=Path, required=True)
    parser.add_argument('--decoder-init', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    codec, _ = load_codec(args.effect_cache/'codec.pt', 'cpu')
    arrays = load_arrays(args.decoder_init)
    generator = np.random.default_rng(19742)
    report = {'passed': False, 'init_sha256': hashlib.sha256(args.decoder_init.read_bytes()).hexdigest(), 'embodiments': {}}
    for action_dim in (22, 44):
        decoder = ActionDecoder(arrays, action_dim)
        source = codec.action_decoders[str(action_dim)]
        source.requires_grad_(True)
        if set(source.state_dict()) != set(decoder.weights):
            raise AssertionError('Parameter mapping is incomplete')
        effect = (generator.normal(size=(2, 30, 256)).astype(np.float32)*arrays['effect_std']+arrays['effect_mean'])
        state = (generator.normal(size=(2, 23*(action_dim//22))).astype(np.float32)*arrays[f'state_std_{action_dim}']
                 +arrays[f'state_mean_{action_dim}'])
        reference_effect = torch.tensor(effect, requires_grad=True)
        reference_state = torch.tensor(state)
        normalized_state = (reference_state-getattr(codec, f'state_mean_{action_dim}'))/getattr(codec, f'state_std_{action_dim}')
        source_prediction = source(reference_effect, normalized_state)
        predicted = decoder(jnp.asarray(effect), jnp.asarray(state))
        np.testing.assert_allclose(np.asarray(predicted), source_prediction.detach().numpy(), rtol=3e-4, atol=1e-5)
        target = source_prediction.detach().numpy()*arrays[f'action_std_{action_dim}']+arrays[f'action_mean_{action_dim}']
        target = target+.05*arrays[f'action_std_{action_dim}']
        normalized_target = (torch.tensor(target)-getattr(codec, f'action_mean_{action_dim}'))/getattr(codec, f'action_std_{action_dim}')
        source_groups = []
        for offset in range(0, action_dim, 22):
            for start, stop in ((0, 3), (3, 6), (6, 22)):
                source_groups.append(functional.smooth_l1_loss(source_prediction[..., offset+start:offset+stop],
                    normalized_target[..., offset+start:offset+stop]))
        source_loss = torch.stack(source_groups).mean()
        source_loss.backward()

        def objective(module, values):
            return grouped_action_loss(module(values, jnp.asarray(state)), jnp.asarray(target), module).mean()

        loss, parameter_grads = nnx.value_and_grad(objective)(decoder, jnp.asarray(effect))
        effect_grads = jax.grad(lambda values: objective(decoder, values))(jnp.asarray(effect))
        np.testing.assert_allclose(float(loss), float(source_loss.detach()), rtol=3e-4, atol=1e-7)
        np.testing.assert_allclose(np.asarray(effect_grads), reference_effect.grad.numpy(), rtol=3e-3, atol=2e-7)
        output_grads = parameter_grads['weights']['out.weight'].value
        np.testing.assert_allclose(np.asarray(output_grads), source.out.weight.grad.numpy(), rtol=3e-3, atol=2e-6)
        velocity_grads = jax.grad(lambda velocity: objective(decoder, jnp.asarray(effect)-.5*velocity))(
            jnp.zeros_like(jnp.asarray(effect)))
        if not float(jnp.linalg.norm(effect_grads)) > 0 or not float(jnp.linalg.norm(velocity_grads)) > 0:
            raise AssertionError('Action loss does not reach the predicted E/flow velocity')
        before = np.asarray(decoder.weights['out.weight'].value).copy()
        decoder.weights['out.weight'].value -= 1e-3*output_grads
        if np.array_equal(before, np.asarray(decoder.weights['out.weight'].value)):
            raise AssertionError('Decoder parameters did not update')
        report['embodiments'][str(action_dim)] = {
            'forward_max_abs': float(np.max(np.abs(np.asarray(predicted)-source_prediction.detach().numpy()))),
            'loss': float(loss), 'effect_gradient_norm': float(jnp.linalg.norm(effect_grads)),
            'velocity_gradient_norm': float(jnp.linalg.norm(velocity_grads)),
            'decoder_parameter_gradient_norm': float(jnp.linalg.norm(output_grads)),
            'pretrained_forward_and_gradients_match': True, 'decoder_updates': True}
    report['passed'] = True
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
