"""π training with complete checkpoints and graceful time-budget boundaries."""
from __future__ import annotations

import dataclasses
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import random
import re
import signal
import time

import numpy as np
import torch
import jax
import orbax.checkpoint as ocp
from etils import epath
from openpi.models import model as model_lib
from openpi.training import checkpoints as checkpoints_lib, sharding
from openpi.shared import array_typing as at, normalize
from afce_all11.prepare import atomic_json
from afce_all11.resumable_data import create_loader


def host_rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_host_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None: torch.cuda.set_rng_state_all(state['torch_cuda'])


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def contract(config,loader):
    dc=loader.data_config();sources=[]
    for s in dc.sources:
        root=Path(s.root)
        metadata={str(p.relative_to(root)):sha(p) for p in sorted((root/'meta').rglob('*')) if p.is_file()}
        sources.append({'task':root.name,'metadata':metadata})
    cache=Path(dc.afce_cache_root) if dc.afce_cache_root else None
    return {'schema':'afce_exact_pi_v1','seed':config.seed,'batch_size':config.batch_size,
            'steps':config.num_train_steps,'model':dataclasses.asdict(config.model),
            'optimizer':dataclasses.asdict(config.optimizer),'schedule':dataclasses.asdict(config.lr_schedule),
            'ema_decay':config.ema_decay,'freeze_filter':re.sub(r'0x[0-9a-fA-F]+','<address>',repr(config.freeze_filter)),
            'fsdp_devices':config.fsdp_devices,'jax_device_count':jax.device_count(),
            'devices':[d.device_kind for d in jax.devices()],
            'packages':{n:importlib.metadata.version(n) for n in ['jax','jaxlib','flax','optax','orbax-checkpoint','torch','numpy','lerobot','av']},
            'sampling':'committed cursor; PCG64 per-epoch permutation; drop_last; frame-proportional',
            'sources':sources,'norm_stats':hashlib.sha256(pickle.dumps(dc.norm_stats,protocol=5)).hexdigest(),
            'effect_manifest':sha(cache/'manifest.json') if cache else None,
            'effect_normalization':sha(cache/'normalization.npz') if cache else None,
            'training_code':{name:sha(Path(__file__).parent/name) for name in
                (['exact_train.py','resumable_data.py','pi_bridge.py']+
                 (['pi_bridge_joint.py','jax_action_decoder.py','readout_alignment.py',
                   'dual_decoder_alignment.py'] if dc.afce_joint_decoder else []))},
            'openpi_code':{str(p.relative_to(Path(__file__).parents[1])):sha(p)
                for folder in ['openpi/src/openpi','openpi/scripts/train.py']
                for p in ([Path(__file__).parents[1]/folder] if folder.endswith('.py') else sorted((Path(__file__).parents[1]/folder).rglob('*.py')))}}


def manager(path,keep_period):
    return ocp.CheckpointManager(epath.Path(path),item_handlers={
        'assets':checkpoints_lib.CallbackHandler(),'train_state':ocp.PyTreeCheckpointHandler(),'params':ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=2,keep_period=keep_period,create=True,
                async_options=ocp.AsyncOptions(timeout_secs=600)))


def complete_steps(path):
    # Orbax's atomic commit directory + the captured runtime must both exist.
    result=[]
    for p in Path(path).glob('[0-9]*'):
        if p.is_dir() and p.name.isdigit() and (p/'_CHECKPOINT_METADATA').exists() and (p/'assets/training_runtime.pkl').exists():
            result.append(int(p.name))
    return sorted(result)


def save(manager_,state,loader,train_rng,run_contract,reason,metrics_path):
    jax.block_until_ready(state)
    updates=int(state.step)
    if loader.committed!=updates: raise ValueError('Checkpoint optimizer and data progress differ')
    runtime={'schema':'afce_complete_training_state_v1','optimizer_updates':updates,
             'loader':loader.state_dict(),'host_rng':host_rng_state(),
             'jax_train_key':np.asarray(jax.random.key_data(train_rng)),
             'jax_key_impl':str(jax.random.key_impl(train_rng)),'contract':run_contract,'reason':reason}
    payload=pickle.dumps(runtime,protocol=5)
    dc=loader.data_config()
    def assets(directory):
        if dc.norm_stats is not None and dc.asset_id is not None: normalize.save(directory/dc.asset_id,dc.norm_stats)
        (directory/'training_runtime.pkl').write_bytes(payload)
        (directory/'training_runtime.sha256').write_text(hashlib.sha256(payload).hexdigest()+'\n')
        (directory/'resume_metadata.json').write_text(json.dumps({'optimizer_updates':updates,
            'next_global_batch':loader.committed,'reason':reason,'schema':runtime['schema']},indent=2))
    with at.disable_typechecking(): train_state,params=checkpoints_lib._split_params(state)
    started=time.monotonic()
    manager_.save(updates-1,{'assets':assets,'train_state':train_state,'params':{'params':params}})
    manager_.wait_until_finished()
    manager_.check_for_errors()
    row={'event':'checkpoint_complete','step':updates,'directory_index':updates-1,'reason':reason,'save_seconds':time.monotonic()-started}
    append(metrics_path,row)
    return row


def restore_runtime(path):
    path=Path(path);payload=(path/'assets/training_runtime.pkl').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=(path/'assets/training_runtime.sha256').read_text().strip():
        raise ValueError('Runtime state checksum mismatch')
    return pickle.loads(payload)


def append(path,row):
    row={'utc':time.time(),**row}
    with Path(path).open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def main(config,*,save_interval=500,save_seconds=600,max_runtime_seconds=4800,stop_file=None,
         stop_after_updates=0,trace=False):
    import train as original_train
    entered=time.monotonic();root=Path(config.checkpoint_dir);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metrics=root/'exact_metrics.jsonl';signal_seen={'value':None}
    def request_stop(signum,_): signal_seen['value']=signal.Signals(signum).name
    for sig in (signal.SIGUSR1,signal.SIGTERM): signal.signal(sig,request_stop)
    if jax.process_count()!=1: raise ValueError('Only single-process JAX supported for exact cursor resume')
    if config.batch_size%jax.device_count(): raise ValueError('Global batch must divide device count')
    torch.set_num_threads(4)
    jax.config.update('jax_compilation_cache_dir',os.environ.get('AFCE_JAX_CACHE',str(root.parent/'jax-cache')))
    rng=jax.random.key(config.seed);train_rng,init_rng=jax.random.split(rng)
    mesh=sharding.make_mesh(config.fsdp_devices)
    ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    loader=create_loader(config,ds);run_contract=contract(config,loader)
    # Normalize JSON tuples consistently before comparison with on-disk metadata.
    run_contract=json.loads(json.dumps(run_contract))
    contract_file=root/'training_contract.json'
    if contract_file.exists() and json.loads(contract_file.read_text())!=run_contract:
        raise ValueError('Training/data/software contract changed; refusing an inexact resume')
    atomic_json(contract_file,run_contract)
    steps=complete_steps(root)
    if steps and not config.resume: raise ValueError('Existing full checkpoint: use --resume')
    if not steps and any(p.is_dir() and p.name.isdigit() for p in root.iterdir()):
        raise ValueError('Only legacy/incomplete checkpoints found; exact resume cannot be assumed')
    ckpt=manager(root,config.keep_period)
    resuming=bool(steps)
    state,state_sharding=original_train.init_train_state(config,init_rng,mesh,resume=resuming)
    if resuming:
        last=steps[-1];runtime=restore_runtime(root/str(last))
        if runtime['contract']!=run_contract: raise ValueError('Saved runtime contract changed')
        with at.disable_typechecking(): state=checkpoints_lib.restore_state(ckpt,state,loader,step=last)
        loader.load_state_dict(runtime['loader'])
        if int(state.step)!=runtime['optimizer_updates'] or int(state.step)!=loader.committed:
            raise ValueError('Restored model and data progress differ')
        train_rng=jax.random.wrap_key_data(runtime['jax_train_key'],impl=runtime['jax_key_impl'])
        restore_host_rng(runtime['host_rng'])
    jax.block_until_ready(state)
    pstep=jax.jit(functools.partial(original_train.train_step,config),
        in_shardings=(replicated,state_sharding,ds),out_shardings=(state_sharding,replicated),donate_argnums=(1,))
    start_step=int(state.step);last_saved=start_step if resuming else 0;last_save_time=time.monotonic()
    atomic_json(root/'active_process.json',{'pid':os.getpid(),'host':platform.node(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
                'start_step':start_step,'max_runtime_seconds':max_runtime_seconds})
    append(metrics,{'event':'exact_training_started','step':start_step,'resumed':resuming,
                    'global_batch':config.batch_size,'device_count':jax.device_count(),'target_updates':config.num_train_steps})
    stopped=False
    try:
        while int(state.step)<config.num_train_steps:
            # No first-batch fetch before restoration; augmentation is keyed by
            # the restored global update, not a newly seeded worker iterator.
            payload,batch_meta=loader.next()
            observation=model_lib.Observation.from_dict(payload);actions=payload['actions']
            batch_hash=None
            if trace:
                h=hashlib.sha256()
                for leaf in jax.tree.leaves((observation,actions)):
                    value=np.asarray(jax.device_get(leaf));h.update(str(value.shape).encode());h.update(str(value.dtype).encode());h.update(value.tobytes())
                batch_hash=h.hexdigest()
            step_before=int(state.step)
            with sharding.set_mesh(mesh): state,info=pstep(train_rng,state,(observation,actions))
            info=jax.device_get(info);jax.block_until_ready(state)
            if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(info)):
                raise FloatingPointError(f'Nonfinite training metrics after update {step_before+1}')
            updates=int(state.step);loader.commit(updates)
            if trace or updates%10==0 or updates==start_step+1:
                row={'event':'train','step':updates,'elapsed_s':time.monotonic()-entered,
                     **{k:float(np.asarray(v)) for k,v in info.items()}}
                if trace: row.update(batch_indices=batch_meta['indices'],batch_sha256=batch_hash)
                append(metrics,row)
            reasons=[]
            if signal_seen['value']: reasons.append(signal_seen['value'])
            if stop_file and Path(stop_file).exists(): reasons.append('scheduler_stop_file')
            if max_runtime_seconds and time.monotonic()-entered>=max_runtime_seconds: reasons.append('walltime_budget')
            if stop_after_updates and updates>=stop_after_updates: reasons.append('bounded_resume_validation')
            finished=updates==config.num_train_steps
            due=updates%save_interval==0 or (save_seconds and time.monotonic()-last_save_time>=save_seconds)
            if due or reasons or finished:
                save(ckpt,state,loader,train_rng,run_contract,'+'.join(reasons) or ('finished' if finished else 'periodic'),metrics)
                last_saved=updates;last_save_time=time.monotonic()
                atomic_json(root/'latest_complete.json',{'optimizer_updates':updates,'directory_index':updates-1,'complete':True})
            if reasons and not finished:
                atomic_json(root/'continuation_needed.json',{'optimizer_updates':updates,'target':config.num_train_steps,'reason':reasons})
                stopped=True;break
        if int(state.step)==config.num_train_steps:
            atomic_json(root/'complete.json',{'optimizer_updates':int(state.step),'checkpoint_index':int(state.step)-1,'status':'pi_training_complete_not_rollout_evaluated'})
            (root/'continuation_needed.json').unlink(missing_ok=True)
    finally:
        loader.close();ckpt.wait_until_finished();ckpt.close()
    append(metrics,{'event':'exact_training_stopped' if stopped else 'exact_training_complete','step':int(state.step),'last_saved_update':last_saved})
    return 75 if stopped else 0
