"""Two-host finger-Δ single decoder with calibrated GT-AUX and fixed finger-delta."""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import signal
import random
import shutil
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
from jax.experimental import multihost_utils

from afce import exact_train
from afce.multihost_data import batch_digest, create_loader
from afce.prepare import atomic_json
from openpi.models import model as model_lib
from openpi.shared import array_typing as at, normalize
from openpi.training import checkpoints as checkpoints_lib, sharding


SOURCE_FILES = ('multihost_finger_delta_pi.py', 'multihost_data.py', 'readout_alignment.py',
                'finger_delta_alignment.py')


def primary_json(path, value):
    if jax.process_index() == 0:
        atomic_json(path, value)


def log(path, row):
    if jax.process_index() == 0:
        exact_train.append(path, row)


def gather_objects(value):
    payload = pickle.dumps(value, protocol=5)
    sizes = np.asarray(multihost_utils.process_allgather(np.asarray(len(payload), dtype=np.int32))).reshape(-1)
    buffer = np.zeros(int(sizes.max()), dtype=np.uint8)
    buffer[:len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    buffers = np.asarray(multihost_utils.process_allgather(buffer)).reshape(jax.process_count(), -1)
    return [pickle.loads(buffers[index, :int(size)].tobytes()) for index, size in enumerate(sizes)]


def runtime_fix_compatible(previous, current, root):
    return previous == current


def restore_state(path, shape, state_sharding):
    def with_sharding(value, placement):
        return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=placement)
    shape = jax.tree.map(with_sharding, shape, state_sharding)
    with at.disable_typechecking():
        train_shape, params_shape = checkpoints_lib._split_params(shape)
    restored = {}
    for name, template in (('train_state', train_shape), ('params', {'params': params_shape})):
        restore_args = jax.tree.map(lambda value: ocp.ArrayRestoreArgs(
            restore_type=jax.Array, sharding=value.sharding, global_shape=value.shape, dtype=value.dtype), template)
        with ocp.PyTreeCheckpointer() as checkpointer:
            restored[name] = checkpointer.restore(
                str(Path(path)/name), args=ocp.args.PyTreeRestore(item=template, restore_args=restore_args))
    return checkpoints_lib._merge_params(restored['train_state'], restored['params'])


def save(checkpointer, state, loader, train_rng, run_contract, reason, metrics):
    jax.block_until_ready(state)
    updates = int(state.step)
    if loader.committed != updates:
        raise ValueError('Optimizer and data progress differ')
    local_host_rng = exact_train.host_rng_state()
    host_states = gather_objects(local_host_rng)
    runtime = {
        'schema': 'afce_multihost_training_state_v1', 'optimizer_updates': updates,
        'loader': loader.state_dict(), 'host_rng': host_states[0], 'host_rng_by_process': host_states,
        'jax_train_key': np.asarray(jax.random.key_data(train_rng)),
        'jax_key_impl': str(jax.random.key_impl(train_rng)), 'contract': run_contract, 'reason': reason,
    }
    payload = pickle.dumps(runtime, protocol=5)
    data_config = loader.data_config()
    def assets(directory):
        if data_config.norm_stats is not None and data_config.asset_id is not None:
            normalize.save(directory/data_config.asset_id, data_config.norm_stats)
        (directory/'training_runtime.pkl').write_bytes(payload)
        (directory/'training_runtime.sha256').write_text(hashlib.sha256(payload).hexdigest()+'\n')
        (directory/'resume_metadata.json').write_text(json.dumps({
            'optimizer_updates': updates, 'next_global_batch': loader.committed,
            'process_count': jax.process_count(), 'reason': reason, 'schema': runtime['schema']}, indent=2))
    with at.disable_typechecking():
        train_state, params = checkpoints_lib._split_params(state)
    started = time.monotonic()
    checkpointer.save(updates-1, {'assets': assets, 'train_state': train_state, 'params': {'params': params}})
    checkpointer.wait_until_finished()
    checkpointer.check_for_errors()
    # Checkpoint bookkeeping must not advance training host random streams.
    exact_train.restore_host_rng(local_host_rng)
    if updates in (32000,35000,40000):
        if jax.process_index()==0:
            milestone=Path(metrics).parent/'milestones'/str(updates)
            if not milestone.exists():
                milestone.parent.mkdir(exist_ok=True)
                shutil.copytree(Path(metrics).parent/str(updates-1),milestone,copy_function=os.link)
        multihost_utils.sync_global_devices('preserved_'+str(updates))
    log(metrics, {'event': 'checkpoint_complete', 'step': updates, 'directory_index': updates-1,
                  'reason': reason, 'save_seconds': time.monotonic()-started})
    primary_json(Path(metrics).parent/'latest_complete.json', {
        'optimizer_updates': updates, 'directory_index': updates-1, 'complete': True})


def train(config, args):
    import train as original_train
    entered = time.monotonic()
    root = Path(config.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = None
    if jax.process_index() == 0:
        lock = (root/'training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    multihost_utils.sync_global_devices('training_lock')
    metrics = root/'exact_metrics.jsonl'
    signal_seen = []
    def request_stop(signum, frame):
        signal_seen.append(signal.Signals(signum).name)
    for signum in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, request_stop)
    random.seed(config.seed+jax.process_index())
    np.random.seed(config.seed+jax.process_index())
    torch.manual_seed(config.seed+jax.process_index())
    torch.set_num_threads(4)
    jax.config.update('jax_compilation_cache_dir', str(root.parent/'jax-cache-8gpu'))
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = create_loader(config, data_sharding)
    run_contract = exact_train.contract(config, loader)
    run_contract.update(schema='afce_multihost_pi_v1', process_count=jax.process_count(),
                        multihost_code={name: exact_train.sha(Path(__file__).parent/name) for name in SOURCE_FILES})
    run_contract = json.loads(json.dumps(run_contract))
    contract_hashes = gather_objects(hashlib.sha256(json.dumps(run_contract, sort_keys=True).encode()).hexdigest())
    if len(set(contract_hashes)) != 1:
        raise ValueError('Processes have different training contracts')
    contract_path = root/'training_contract.json'
    if contract_path.exists() and not runtime_fix_compatible(
            json.loads(contract_path.read_text()), run_contract, root):
        raise ValueError('Eight-GPU training contract changed')
    primary_json(contract_path, run_contract)
    existing = exact_train.complete_steps(root)
    source = root/str(existing[-1]) if existing else args.source
    if source is None:
        if not args.initialize_from_base:
            raise ValueError('A complete source checkpoint is required unless --initialize-from-base is explicit')
        state, state_sharding = original_train.init_train_state(config, init_rng, mesh, resume=False)
        resumed = False
    else:
        source = source.resolve()
        runtime = exact_train.restore_runtime(source)
        if runtime['contract'] != run_contract:
            raise ValueError('Training contract changed; inexact resume is forbidden')
        state_shape, state_sharding = original_train.init_train_state(config, init_rng, mesh, resume=True)
        state = restore_state(source, state_shape, state_sharding)
        loader.load_state_dict(runtime['loader'])
        # DataLoader iterator construction consumes process-local host RNG.  Do
        # that bookkeeping before restoring the saved RNG snapshot so a restart
        # begins the next update from exactly the checkpointed host state.
        loader.start()
        if int(state.step) != runtime['optimizer_updates'] or int(state.step) != loader.committed:
            raise ValueError('Restored optimizer/model/data progress differs')
        global_key_data = jax.make_array_from_process_local_data(replicated, runtime['jax_train_key'])
        train_rng = jax.random.wrap_key_data(global_key_data, impl=runtime['jax_key_impl'])
        host_states = runtime.get('host_rng_by_process')
        if host_states is not None and len(host_states) != jax.process_count():
            raise ValueError('Saved host RNG process count changed')
        exact_train.restore_host_rng(host_states[jax.process_index()] if host_states else runtime['host_rng'])
        resumed = True
    # Fresh JIT state.step is weakly typed; Orbax restores it strongly typed.
    # Use the same explicit int32 scalar on both paths before tracing pstep.
    original_step_weak_type = bool(getattr(state.step, "weak_type", False))
    state = jax.tree.map(lambda value: jnp.asarray(value, dtype=value.dtype), state)
    jax.block_until_ready(state)
    start_step = int(state.step)
    if start_step >= config.num_train_steps:
        raise ValueError('The source experiment has already finished')
    if args.calibrate:
        from afce.calibrate_readout import calibrate
        calibrate(config,state,loader,train_rng,mesh,data_sharding,state_sharding,replicated,args.calibrate)
        return 0
    before_manager_rng = exact_train.host_rng_state()
    checkpointer = exact_train.manager(root, 1000)
    exact_train.restore_host_rng(before_manager_rng)
    primary_json(root/'active_process.json', {
        'pid': os.getpid(), 'host': platform.node(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'start_step': start_step, 'process_count': 2, 'device_count': 8,
        'deadline_unix': args.deadline_unix, 'source_checkpoint': str(source) if source else None,
        'decoder_warmup_steps': config.model.decoder_warmup_steps,
        'original_step_weak_type': original_step_weak_type,
        'canonical_step_weak_type': bool(state.step.weak_type)})
    log(metrics, {'event': 'exact_training_started', 'step': start_step, 'resumed': resumed,
                  'global_batch': 32, 'device_count': 8, 'process_count': 2,
                  'per_device_batch': 4, 'target_updates': config.num_train_steps,
                  'decoder_warmup_steps': config.model.decoder_warmup_steps,
                  'source_checkpoint': str(source) if source else None})
    pstep = jax.jit(functools.partial(original_train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    last_saved = start_step if existing else None
    last_save_time = time.monotonic()
    stopped = False
    try:
        if args.restore_only:
            if source is None:
                raise ValueError('--restore-only requires an existing source checkpoint')
            if not existing:
                save(checkpointer, state, loader, train_rng, run_contract, 'four_to_eight_gpu_migration', metrics)
            primary_json(root/'migration_source.json', {
                'checkpoint': str(source), 'optimizer_updates': start_step,
                'source_runtime_sha256': exact_train.sha(source/'assets/training_runtime.pkl'),
                'source_contract': runtime['contract'], 'destination_contract': run_contract})
            return 0
        if args.stop_at_update and start_step >= args.stop_at_update:
            return 0
        while int(state.step) < config.num_train_steps:
            payload, metadata = loader.next()
            batch_hash = None
            batch_indices = None
            if args.trace:
                digests = np.asarray(multihost_utils.process_allgather(batch_digest(payload)))
                batch_hash = hashlib.sha256(digests.tobytes()).hexdigest()
                batch_indices = np.asarray(multihost_utils.process_allgather(
                    np.asarray(metadata['indices'], dtype=np.int64), tiled=True)).tolist()
            observation = model_lib.Observation.from_dict(payload)
            # JAX/Flax tracing is host bookkeeping; the training randomness is
            # the explicit JAX key and per-visit DataLoader seeds.
            before_trace_rng = exact_train.host_rng_state()
            with sharding.set_mesh(mesh):
                state, info = pstep(train_rng, state, (observation, payload['actions']))
            jax.block_until_ready(state)
            exact_train.restore_host_rng(before_trace_rng)
            info = jax.device_get(info)
            if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(info)):
                raise FloatingPointError('Nonfinite training metrics')
            updates = int(state.step)
            loader.commit(updates)
            if float(np.asarray(info['decoder_train_enabled'])) != 1:
                raise AssertionError('Both embodiment heads of the single decoder must remain trainable')
            if float(np.asarray(info['ramp'])) > 0:
                for arm in ('single', 'bimanual'):
                    if float(np.asarray(info[arm+'_aux_samples'])) > 0 and not float(np.asarray(info[arm+'_decoder_grad_norm'])) > 0:
                        raise AssertionError('Eligible jointly trained decoder has zero gradient: '+arm)
            expected = sum(float(np.asarray(info[name])) for name in
                           ('effect_loss', 'action_loss'))
            if not np.isclose(float(np.asarray(info['loss'])), expected, rtol=1e-5, atol=1e-6):
                raise AssertionError('Alignment loss weights or logging changed')
            if args.trace or updates % 10 == 0 or updates == start_step+1:
                row = {'event': 'train', 'step': updates, 'elapsed_s': time.monotonic()-entered,
                       **{name: float(np.asarray(value)) for name, value in info.items()}}
                if args.trace:
                    row.update(batch_indices=batch_indices, batch_sha256=batch_hash)
                log(metrics, row)
            control = np.asarray(multihost_utils.process_allgather(np.asarray([
                bool(signal_seen), bool(args.deadline_unix and time.time() >= args.deadline_unix),
                bool(args.stop_at_update and updates >= args.stop_at_update),
                bool(updates % 500 == 0 or updates in (32000,35000,40000) or updates == start_step+100 or time.monotonic()-last_save_time >= 600),
            ], dtype=np.int32))).reshape(jax.process_count(), 4)
            flags = control.any(axis=0)
            reasons = [name for name, enabled in zip(
                ('slurm_signal', 'allocation_deadline', 'bounded_resume_validation'), flags[:3], strict=True) if enabled]
            finished = updates == config.num_train_steps
            if flags.any() or finished:
                save(checkpointer, state, loader, train_rng, run_contract,
                     '+'.join(reasons) or ('finished' if finished else 'periodic'), metrics)
                last_saved = updates
                last_save_time = time.monotonic()
            if reasons and not finished:
                primary_json(root/'continuation_needed.json', {
                    'optimizer_updates': updates, 'target': config.num_train_steps, 'reason': reasons})
                stopped = True
                break
        if int(state.step) == config.num_train_steps:
            primary_json(root/'complete.json', {'complete': True, 'optimizer_updates': int(state.step),
                'checkpoint_index': int(state.step)-1, 'status': 'pi_training_complete_not_rollout_evaluated'})
            if jax.process_index() == 0:
                (root/'continuation_needed.json').unlink(missing_ok=True)
    finally:
        loader.close()
        checkpointer.wait_until_finished()
        checkpointer.close()
    log(metrics, {'event': 'exact_training_stopped' if stopped else 'exact_training_complete',
                  'step': int(state.step), 'last_saved_update': last_saved})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-root', type=Path, required=True)
    parser.add_argument('--decoder-init', type=Path, required=True)
    parser.add_argument('--assets-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--name', default='finger_delta_seed42_60000')
    parser.add_argument('--restore-only', action='store_true')
    parser.add_argument('--stop-at-update', type=int, default=0)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--deadline-unix', type=int, default=0)
    parser.add_argument('--initialize-from-base', action='store_true')
    parser.add_argument('--decoder-warmup-steps', type=int, default=0)
    parser.add_argument('--variant', choices=('finger_delta',), default='finger_delta')
    parser.add_argument('--calibrate', type=Path)
    parser.add_argument('--weight', type=float, default=0.25628781345139295)
    args = parser.parse_args()
    jax.distributed.initialize(coordinator_address=os.environ['AFCE_COORDINATOR'],
        num_processes=2, process_id=int(os.environ['SLURM_PROCID']), local_device_ids=list(range(4)),
        initialization_timeout=180)
    if jax.process_count() != 2 or jax.device_count() != 8 or jax.local_device_count() != 4:
        raise ValueError('Expected exactly two JAX processes and eight GPUs, four GPUs on each host')
    if any(device.platform != 'gpu' for device in jax.devices()):
        raise ValueError('All eight devices must be GPUs')
    repository = Path(__file__).resolve().parents[1]
    openpi_root = Path(os.environ.get('AFCE_OPENPI_ROOT', repository/'openpi')).resolve()
    if not (openpi_root/'config.yaml').is_file() or not (openpi_root/'scripts').is_dir():
        raise FileNotFoundError(f'Invalid AFCE_OPENPI_ROOT: {openpi_root}')
    os.chdir(openpi_root)
    sys.path.insert(0, str(openpi_root/'scripts'))
    import dexjoco_multitask as base
    from afce.pi_bridge_joint import joint_effect_config
    query_root = args.query_root.resolve()
    decoder_init = args.decoder_init.resolve()
    assets_dir = args.assets_dir.resolve()
    runtime_root = query_root.parent
    base_params = Path(os.environ.get(
        'AFCE_PI05_BASE_PARAMS',
        str(repository / 'checkpoints' / 'pi05_base_action_dim_44' / 'params'),
    )).resolve()
    config, init, tasks = base.build_config(base.Args(
        operation='inspect', data_root=runtime_root/'datasets_hf/dexjoco_lerobot_datasets',
        init_params_path=base_params, exp_name=args.name, checkpoint_base_dir=args.output.resolve()/'checkpoints',
        assets_base_dir=assets_dir, batch_size=32, num_workers=4, num_train_steps=60000,
        resume=True, save_interval=500, keep_period=10000, fsdp_devices=1, wandb_enabled=False))
    config = dataclasses.replace(config, name='afce_official', seed=42,
                                 data=dataclasses.replace(config.data, balance='proportional'))
    config = joint_effect_config(config, query_root/'effect_cache', init, decoder_init,
                                 decoder_warmup_steps=args.decoder_warmup_steps)
    if args.weight != 0.25628781345139295 or args.decoder_warmup_steps != 0:
        raise ValueError('finger-Δ fixed-weight ablation requires original GT-AUX weight and no decoder freeze')
    from afce.finger_delta_alignment import finger_delta_config
    config = finger_delta_config(config)
    base._validate_sources(config)
    if not base._norm_stats_path(config).is_file():
        raise ValueError('The original normalization statistics must already exist')
    exit_code = train(config, args)
    multihost_utils.sync_global_devices('training_finished')
    jax.distributed.shutdown()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
