"""Auditable recipe forks with exact restart within each fixed recipe."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from afce.codec import losses
from afce.evidence import EvidenceData
from afce.lowdim import grouped_batch
from afce.prepare import atomic_json
from afce.train_calibrated import atomic_save
from afce.train_codec import validate
from afce.tuned_codec import TunedCodec, perturb_state


def source_sha():
    root = Path(__file__).resolve().parents[1]
    files = [root/'afce'/name for name in (
        'train_tuning.py', 'tuned_codec.py', 'calibrated_codec.py', 'codec.py',
        'evidence.py', 'lowdim.py', 'train_codec.py')]
    files += sorted((root/'effect_codec/model').glob('*.py'))
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def configure_torch():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def restore_rng(blob, rng):
    rng.bit_generator.state = blob['numpy_rng']
    np.random.set_state(blob['numpy_global_rng'])
    random.setstate(blob['python_rng'])
    torch.set_rng_state(blob['torch_rng'])
    torch.cuda.set_rng_state_all(blob['cuda_rng_all'])


def make_optimizer(model, blob, lr):
    parameters = dict(model.named_parameters())
    names = blob.get('optimizer_parameter_names')
    if names is None:
        names = [[name for name in parameters if not name.startswith('aligned_heads.')]]
    groups = [{'params': [parameters[name] for name in group]} for group in names]
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=.01)
    optimizer.load_state_dict(blob['optimizer'])
    covered = {name for group in names for name in group}
    added = [name for name in parameters if name not in covered]
    if added:
        if not all(name.startswith('aligned_heads.') for name in added):
            raise ValueError('Unexpected new parameters')
        optimizer.add_param_group({'params': [parameters[name] for name in added]})
        names = names + [added]
    for group in optimizer.param_groups:
        group['lr'] = lr
    return optimizer, names


def initialize(blob, config, device='cuda:0'):
    parent_config = blob['config']
    if parent_config['fusion'] != config['fusion'] or parent_config['condition'] != 'ASV':
        raise ValueError('Parent fusion or conditioning differs')
    model = TunedCodec(config['fusion'], 'ASV', blob['stats'], config['aligned_residual']).to(device)
    incompatible = model.load_state_dict(blob['model'], strict=False)
    adding = config['aligned_residual'] and not parent_config.get('aligned_residual', False)
    allowed = {name for name in model.state_dict() if name.startswith('aligned_heads.')} if adding else set()
    if set(incompatible.missing_keys) != allowed or incompatible.unexpected_keys:
        raise ValueError(f'Incompatible checkpoint: {incompatible}')
    optimizer, names = make_optimizer(model, blob, config['lr'])
    rng = np.random.default_rng()
    restore_rng(blob, rng)
    return model, optimizer, names, rng


def train_step(model, optimizer, batch, stats, config, noise_rng):
    optimizer.zero_grad(set_to_none=True)
    metrics = {}
    for b in batch.values():
        e = model.encode(b)
        decoded_batch = b
        if config['state_noise']:
            decoded_batch = dict(b)
            decoded_batch['state'] = perturb_state(b['state'], noise_rng)
        out = model.decode_outputs(e, decoded_batch)
        values = losses(out, b, stats, config['lambda_s'], config['lambda_v'])
        values['loss'] = values['loss'] + (config['action_weight']-1) * values['LA']
        if not all(torch.isfinite(v) for v in values.values()):
            raise FloatingPointError('Nonfinite training loss')
        weight = len(b['action'])/config['batch_size']
        (values['loss']*weight).backward()
        for key, value in values.items():
            metrics[key] = metrics.get(key, 0.) + float(value.detach())*weight
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient')
    optimizer.step()
    return {**metrics, 'grad_norm': float(norm)}


def checkpoint(model, optimizer, names, rng, noise_rng, step, best, score, config, parent):
    return {'schema': 'afce_tuned_v3', 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'optimizer_parameter_names': names,
            'step': step, 'best': best, 'val': score, 'stats': parent['stats'], 'config': config,
            'statistics_sha256': parent['statistics_sha256'], 'numpy_rng': rng.bit_generator.state,
            'numpy_global_rng': np.random.get_state(), 'python_rng': random.getstate(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng_all': torch.cuda.get_rng_state_all(),
            'noise_rng': noise_rng.get_state()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--init-from', type=Path, required=True)
    p.add_argument('--fusion', choices=['query', 'poe'], required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--action-weight', type=float, default=1.)
    p.add_argument('--lambda-s', type=float, default=.2)
    p.add_argument('--lambda-v', type=float, default=.2)
    p.add_argument('--aligned-residual', action='store_true')
    p.add_argument('--state-noise', action='store_true')
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--save-seconds', type=int, default=300)
    p.add_argument('--max-runtime-seconds', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.lr <= 0 or args.action_weight <= 0 or args.batch_size != 32:
        p.error('Positive learning rate/weight and fixed batch32 are required')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configure_torch()
    parent = torch.load(args.init_from, map_location='cpu', weights_only=False)
    if parent['config'].get('codec_version') not in ('calibrated_v2', 'tuned_v3'):
        raise ValueError('Use a calibrated or tuned parent')
    if args.steps <= parent['step']:
        raise ValueError('Total steps must exceed parent steps')
    if args.output.resolve() == args.init_from.resolve().parent:
        raise ValueError('Fork must use a new output directory')
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()
              if key not in ('resume', 'save_every', 'save_seconds', 'max_runtime_seconds')}
    config.update(codec_version='tuned_v3', condition='ASV', action_only=False, mask_mode='legacy-none',
                  effect_shape=[30, 256], independent_action_modules=[22, 44],
                  parent_sha256=hashlib.sha256(args.init_from.read_bytes()).hexdigest(),
                  parent_step=parent['step'], source_sha256=source_sha(),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(0), selection='validation_LA_512_fixed_windows',
                  noise_position_std_m=.002, noise_rotation_std_deg=.5, noise_joint_std_deg=.5,
                  deterministic_algorithms=True, initialization='full_optimizer_and_rng_fork')
    path = args.output/'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Run contract changed')
    atomic_json(path, config)
    if (args.output/'last.pt').exists() and not args.resume:
        raise ValueError('Use --resume for an existing experiment')
    saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False) if args.resume else parent
    if args.resume and saved['config'] != config:
        raise ValueError('Checkpoint configuration changed')
    started = time.monotonic()
    stop = []
    def request_stop(sig, _):
        stop.append(signal.Signals(sig).name)
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    print(json.dumps({'event': 'loading_data', 'fusion': args.fusion}), flush=True)
    train = EvidenceData(args.data, args.evidence, 'train', 'legacy-none')
    val = EvidenceData(args.data, args.evidence, 'val', 'legacy-none')
    model, optimizer, names, rng = initialize(saved, config)
    noise_rng = torch.Generator(device='cuda:0').manual_seed(9042)
    if 'noise_rng' in saved:
        noise_rng.set_state(saved['noise_rng'])
    step = saved['step']
    vi = np.random.default_rng(103).integers(0, len(val), 512)
    score = validate(model, val, parent['stats'], 'cuda:0', vi, False)
    best = saved['best'] if args.resume else score['LA']
    def log(row):
        row = {'unix_time': time.time(), 'elapsed_s': time.monotonic()-started, **row}
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    def save(name):
        blob = checkpoint(model, optimizer, names, rng, noise_rng, step, best, score, config, parent)
        atomic_save(blob, args.output/name)
    if not args.resume:
        save('best.pt'); save('last.pt')
    log({'event': 'train_started', 'step': step, 'fusion': args.fusion,
         'parent_sha256': config['parent_sha256'], 'validation': score,
         'lr': args.lr, 'batch_size': args.batch_size})
    last_save = time.monotonic()
    model.train()
    for step in range(step+1, args.steps+1):
        indices = rng.integers(0, len(train), args.batch_size)
        metrics = train_step(model, optimizer, grouped_batch(train, indices, 'cuda:0'), parent['stats'], config, noise_rng)
        if step % 50 == 0:
            log({'event': 'train', 'step': step, **metrics})
        improved = False
        if step % args.eval_every == 0 or step == args.steps:
            score = validate(model, val, parent['stats'], 'cuda:0', vi, False)
            improved = score['LA'] < best
            best = min(best, score['LA'])
            log({'event': 'validation', 'step': step, **score})
        if args.max_runtime_seconds and time.monotonic()-started >= args.max_runtime_seconds:
            stop.append('walltime_budget')
        if improved or stop or step % args.save_every == 0 or time.monotonic()-last_save >= args.save_seconds or step == args.steps:
            save('last.pt')
            if improved:
                save('best.pt')
            last_save = time.monotonic()
            log({'event': 'checkpoint_complete', 'step': step})
        if stop and step < args.steps:
            atomic_json(args.output/'continuation_needed.json', {'step': step, 'reasons': stop})
            raise SystemExit(75)
    atomic_json(args.output/'complete.json', {'steps': step, 'best_validation_LA': best,
                'status': 'tuning_completed_execution_acceptance_pending'})


if __name__ == '__main__':
    main()
