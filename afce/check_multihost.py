"""Audit topology migration and actual eight-GPU interrupted training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
import torch

from afce.exact_train import restore_runtime
from afce.prepare import atomic_json
from afce.resumable_data import CursorBatchSampler


def numpy_restore(checkpointer, path):
    metadata = checkpointer.metadata(str(path))
    tree = getattr(metadata, 'tree', metadata)
    restore_args = jax.tree.map(lambda value: ocp.ArrayRestoreArgs(restore_type=np.ndarray), tree)
    return checkpointer.restore(str(path), args=ocp.args.PyTreeRestore(item=tree, restore_args=restore_args))


def compare_checkpoints(left, right):
    result = {}
    for item in ('params', 'train_state'):
        with ocp.PyTreeCheckpointer() as checkpointer:
            left_tree = numpy_restore(checkpointer, left/item)
            right_tree = numpy_restore(checkpointer, right/item)
        left_leaves, left_structure = jax.tree.flatten(left_tree)
        right_leaves, right_structure = jax.tree.flatten(right_tree)
        if left_structure != right_structure:
            raise AssertionError(f'{item}: tree structure changed')
        differences = []
        for index, (left_value, right_value) in enumerate(zip(left_leaves, right_leaves, strict=True)):
            if not np.array_equal(left_value, right_value):
                differences.append(index)
        result[item] = {'leaves': len(left_leaves), 'bitwise_equal': not differences, 'differences': differences[:20]}
        del left_tree, right_tree, left_leaves, right_leaves
    return result


def equal_nested(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_nested(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_nested(first, second) for first, second in zip(left, right, strict=True))
    return left == right


def compare_runtime(left, right, *, migration=False):
    left_runtime, right_runtime = restore_runtime(left), restore_runtime(right)
    cursor = left_runtime['loader']
    batch = next(iter(CursorBatchSampler(cursor['length'], cursor['batch_size'], cursor['seed'], cursor['committed_batches'])))
    rank_batches = (batch[:16], batch[16:])
    if migration:
        host_rng_equal = all(equal_nested(left_runtime['host_rng'], host) for host in right_runtime['host_rng_by_process'])
    else:
        host_rng_equal = equal_nested(left_runtime['host_rng_by_process'], right_runtime['host_rng_by_process'])
    return {
        'updates_equal': left_runtime['optimizer_updates'] == right_runtime['optimizer_updates'],
        'loader_state_equal': equal_nested(left_runtime['loader'], right_runtime['loader']),
        'jax_rng_equal': equal_nested(left_runtime['jax_train_key'], right_runtime['jax_train_key']) and
                         left_runtime['jax_key_impl'] == right_runtime['jax_key_impl'],
        'all_host_rng_equal': host_rng_equal,
        'next_global_indices': [entry[0] for entry in batch],
        'next_process_indices': [[entry[0] for entry in entries] for entries in rank_batches],
        'partition_complete': rank_batches[0]+rank_batches[1] == batch and
                              set(entry[1] for entry in rank_batches[0]).isdisjoint(entry[1] for entry in rank_batches[1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    source = Path(plan['source_checkpoint'])
    initial = int(plan['start_step'])
    formal = Path(plan['formal_checkpoint_root'])
    check = Path(plan['check_root'])
    continuous = check/'checkpoints/afce_official/continuous'
    interrupted = check/'checkpoints/afce_official/interrupted'
    migration = compare_checkpoints(source, formal/str(initial-1))
    migration_runtime = compare_runtime(source, formal/str(initial-1), migration=True)
    continuation = compare_checkpoints(continuous/str(initial+1), interrupted/str(initial+1))
    continuation_runtime = compare_runtime(continuous/str(initial+1), interrupted/str(initial+1))
    def rows(root):
        return [row for line in (root/'exact_metrics.jsonl').read_text().splitlines()
                if (row := json.loads(line))['event'] == 'train']
    left_rows, right_rows = rows(continuous), rows(interrupted)
    keys = ('step', 'batch_indices', 'batch_sha256', 'loss', 'grad_norm', 'param_norm',
            'effect_loss', 'action_loss', 'action_single_loss', 'action_bimanual_loss',
            'single_decoder_grad_norm', 'bimanual_decoder_grad_norm')
    rows_equal = ([{key: row[key] for key in keys} for row in left_rows] ==
                  [{key: row[key] for key in keys} for row in right_rows])
    correct_steps = [row['step'] for row in left_rows] == [initial+1, initial+2]
    decoder_gradients = all(any(row[name] > 0 and np.isfinite(row[name]) for row in left_rows)
                           for name in ('single_decoder_grad_norm', 'bimanual_decoder_grad_norm'))
    loss_weights = all(np.isclose(row['loss'], row['effect_loss']+row['action_loss'], rtol=1e-6, atol=1e-7)
                       for row in left_rows)
    runtime_keys = ('updates_equal', 'loader_state_equal', 'jax_rng_equal', 'partition_complete')
    passed = (all(item['bitwise_equal'] for item in [*migration.values(), *continuation.values()]) and
              all(runtime[key] for runtime in (migration_runtime, continuation_runtime) for key in runtime_keys) and
              rows_equal and correct_steps and decoder_gradients and loss_weights)
    report = {'passed': passed, 'source_checkpoint': str(source), 'start_step': initial,
              'global_batch': 32, 'process_count': 2, 'device_count': 8, 'per_device_batch': 4,
              'migration': migration, 'migration_runtime': migration_runtime,
              'continuation': continuation, 'continuation_runtime': continuation_runtime,
              'training_rows_equal': rows_equal, 'correct_steps': correct_steps,
              'both_decoders_receive_gradients': decoder_gradients, 'loss_weights_1_to_1': loss_weights,
              'host_rng_difference_is_blocking': False,
              'acceptance_basis': 'User accepts host Python RNG bookkeeping difference after identical model, optimizer, data, JAX RNG, losses and gradients; host RNG equality is still reported',
              'contract': json.loads((formal/'training_contract.json').read_text())}
    original_report = check/'result_before_host_rng_acceptance.json'
    if (check/'result.json').exists() and not original_report.exists():
        atomic_json(original_report, json.loads((check/'result.json').read_text()))
    atomic_json(check/'result.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise AssertionError('Eight-GPU migration/restart validation failed; formal continuation is blocked')
    atomic_json(Path(plan['query_root'])/'eight_gpu_ready.json', report)


if __name__ == '__main__':
    main()
