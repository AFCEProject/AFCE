"""Independent from-scratch codec trials; deterministic full-state continuation."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
import mujoco
import scipy
import pandas
from afce.codec import losses
from afce.evidence import EvidenceData
from afce.lowdim import grouped_batch
from afce.prepare import atomic_json
from afce.train_calibrated import atomic_save
from afce.train_codec import validate
from afce.train_tuning import configure_torch, restore_rng
from afce.tuned_codec import TunedCodec, perturb_state


def source_hash():
    root = Path(__file__).resolve().parents[1]
    names = ('train_sweep.py', 'codec.py', 'calibrated_codec.py', 'tuned_codec.py',
             'evidence.py', 'lowdim.py', 'train_codec.py', 'train_calibrated.py', 'train_tuning.py')
    files = [root/'afce'/name for name in names]
    files.append(root/'effect_codec/constants.py')
    files += sorted((root/'effect_codec/model').glob('*.py'))
    files += sorted((root/'dexjoco/dexjoco/sim/envs/xmls').glob('panda_allegro_*.xml'))
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def learning_rate(step, c):
    if c['schedule'] == 'two_stage':
        return c['lr'] if step <= c['switch_step'] else c['min_lr']
    if c['warmup'] and step <= c['warmup']:
        return c['lr']*step/c['warmup']
    if c['schedule'] == 'cosine':
        progress = min(1., max(0., (step-c['warmup'])/max(1, c['steps']-c['warmup'])))
        return c['min_lr'] + .5*(c['lr']-c['min_lr'])*(1+math.cos(math.pi*progress))
    return c['lr']


def make_optimizer(model, c):
    if c['exclude_norm_decay']:
        named = list(model.named_parameters())
        groups = [{'params': [p for _, p in named if p.ndim >= 2], 'weight_decay': c['weight_decay']},
                  {'params': [p for _, p in named if p.ndim < 2], 'weight_decay': 0.}]
    else:
        groups = model.parameters()
    return torch.optim.AdamW(groups, lr=c['lr'], betas=(.9, c['beta2']), eps=1e-8,
                             weight_decay=c['weight_decay'])


def step_once(model, optimizer, batch, stats, c, noise_rng, step):
    optimizer.zero_grad(set_to_none=True)
    lam_s = .05+(c['lambda_s']-.05)*min(step/1000, 1.)
    lam_v = .05+(c['lambda_v']-.05)*min(step/1000, 1.)
    metrics = {}
    for b in batch.values():
        e = model.encode(b)
        decoded = dict(b)
        if c['state_noise']:
            decoded['state'] = perturb_state(b['state'], noise_rng)
        out = model.decode_outputs(e, decoded)
        v = losses(out, b, stats, lam_s, lam_v)
        # Preserve the original per-physical-group Huber objective as a logged metric.
        # Position weighting changes only the training objective, never the evaluator.
        scale = stats[str(b['action'].shape[-1])]['action_std'].to(e.device)
        pos = torch.stack([F.smooth_l1_loss(out['action'][..., off:off+3]/scale[off:off+3],
                              b['action'][..., off:off+3]/scale[off:off+3])
                           for off in range(0, b['action'].shape[-1], 22)]).mean()
        v['loss'] += (c['action_weight']-1)*v['LA']
        v['loss'] += c['action_weight']*(c['position_weight']-1)*pos/3
        if not all(torch.isfinite(x) for x in v.values()):
            raise FloatingPointError('Nonfinite training objective')
        weight = len(b['action'])/c['batch_size']
        (v['loss']*weight).backward()
        for key, value in v.items():
            metrics[key] = metrics.get(key, 0.)+float(value.detach())*weight
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(grad):
        raise FloatingPointError('Nonfinite gradient')
    optimizer.step()
    return {**metrics, 'grad_norm': float(grad)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'evidence', 'statistics', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fusion', choices=['query', 'poe'], required=True)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min-lr', type=float, default=1e-5)
    p.add_argument('--schedule', choices=['constant', 'two_stage', 'cosine'], default='constant')
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--switch-step', type=int, default=20000)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--beta2', type=float, default=.999)
    p.add_argument('--exclude-norm-decay', action='store_true')
    p.add_argument('--action-weight', type=float, default=1.)
    p.add_argument('--position-weight', type=float, default=1.)
    p.add_argument('--lambda-s', type=float, default=.2)
    p.add_argument('--lambda-v', type=float, default=.2)
    p.add_argument('--aligned-residual', action='store_true')
    p.add_argument('--state-noise', action='store_true')
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--save-seconds', type=int, default=300)
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--max-runtime-seconds', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if (a.batch_size != 32 or a.steps < 1 or a.lr <= 0 or a.min_lr < 0 or a.min_lr > a.lr
            or a.weight_decay < 0 or not 0 < a.beta2 < 1 or a.warmup >= a.steps
            or min(a.action_weight, a.position_weight) <= 0):
        p.error('Invalid fixed-batch32 experiment configuration')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU for each trial')
    configure_torch()
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output/'training.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verified_path = a.evidence.parents[1]/'inputs_verified.json'
    verified = json.loads(verified_path.read_text())
    manifest_path = verified_path.parent/'input_manifest.json'
    if not verified['passed'] or verified['manifest_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError('Input bundle was not verified')
    parent = torch.load(a.statistics, map_location='cpu', weights_only=False)
    stats = parent['stats']
    c = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()
         if k not in ('resume', 'save_every', 'save_seconds', 'stop_after', 'max_runtime_seconds')}
    c.update(codec_version='tuned_v3', sweep_version='scratch_v1', condition='ASV', action_only=False,
             mask_mode='legacy-none', effect_shape=[30, 256], independent_action_modules=[22, 44],
             source_sha256=source_hash(), input_manifest_sha256=verified['manifest_sha256'],
             statistics_sha256=hashlib.sha256(a.statistics.read_bytes()).hexdigest(),
             torch_version=torch.__version__, cuda_version=torch.version.cuda,
             numpy_version=np.__version__, mujoco_version=mujoco.__version__,
             scipy_version=scipy.__version__, pandas_version=pandas.__version__,
             device=torch.cuda.get_device_name(0),
             initialization='random_from_step_zero', selection='fixed_validation_LA_not_execution')
    path = a.output/'config.json'
    if path.exists() and json.loads(path.read_text()) != c:
        raise ValueError('Experiment contract changed')
    if (a.output/'last.pt').exists() and not a.resume:
        raise ValueError('Existing checkpoint; use --resume')
    atomic_json(path, c)
    stop = []
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda sig, frame: stop.append(signal.Signals(sig).name))
    started = time.monotonic()
    def log(row):
        row = {'unix_time': time.time(), 'elapsed_s': time.monotonic()-started, **row}
        with (a.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    log({'event': 'loading_data', 'fusion': a.fusion})
    train = EvidenceData(a.data, a.evidence, 'train', 'legacy-none')
    val = EvidenceData(a.data, a.evidence, 'val', 'legacy-none')
    if (len(train), len(val), len(train.episodes), len(val.episodes)) != (469908, 53855, 990, 110):
        raise ValueError('Dataset split differs')
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    model = TunedCodec(a.fusion, 'ASV', stats, a.aligned_residual).cuda()
    optimizer = make_optimizer(model, c)
    names = {id(p): n for n, p in model.named_parameters()}
    parameter_names = [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]
    rng = np.random.default_rng(a.seed)
    noise_rng = torch.Generator(device='cuda:0').manual_seed(9042+a.seed)
    step = 0; best = float('inf'); score = None
    initial = {}
    for name in ('action_encoders', 'action_decoders'):
        for d, module in getattr(model, name).items():
            h = hashlib.sha256()
            for value in module.state_dict().values():
                h.update(value.cpu().numpy().tobytes())
            initial[name+'/'+d] = h.hexdigest()
    if a.resume:
        blob = torch.load(a.output/'last.pt', map_location='cpu', weights_only=False)
        if blob['config'] != c:
            raise ValueError('Saved configuration differs')
        model.load_state_dict(blob['model']); optimizer.load_state_dict(blob['optimizer'])
        restore_rng(blob, rng); noise_rng.set_state(blob['noise_rng'])
        step, best, score = blob['step'], blob['best'], blob['val']
    atomic_json(a.output/'initial_modules.json', initial)
    vi = np.random.default_rng(103).integers(0, len(val), 512)
    def save(name):
        atomic_save({'schema': 'afce_scratch_sweep_v1', 'config': c, 'stats': stats,
                     'statistics_sha256': c['statistics_sha256'], 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'optimizer_parameter_names': parameter_names,
                     'step': step, 'best': best, 'val': score, 'numpy_rng': rng.bit_generator.state,
                     'numpy_global_rng': np.random.get_state(), 'python_rng': random.getstate(),
                     'torch_rng': torch.get_rng_state(), 'cuda_rng_all': torch.cuda.get_rng_state_all(),
                     'noise_rng': noise_rng.get_state(), 'schedule': {'kind': a.schedule, 'completed_updates': step}},
                    a.output/name)
    if not a.resume:
        save('last.pt')
    log({'event': 'train_started', 'step': step, 'batch_size': 32, 'initial_modules': initial})
    last_save = time.monotonic(); model.train()
    for step in range(step+1, a.steps+1):
        lr = learning_rate(step, c)
        for group in optimizer.param_groups:
            group['lr'] = lr
        ids = rng.integers(0, len(train), 32)
        metrics = step_once(model, optimizer, grouped_batch(train, ids, 'cuda:0'), stats, c, noise_rng, step)
        if step == 1 or step % 50 == 0:
            log({'event': 'train', 'step': step, 'lr': lr, **metrics})
        improved = False
        if step % a.eval_every == 0 or step == a.steps:
            score = validate(model, val, stats, 'cuda:0', vi, False)
            if not all(math.isfinite(v) for v in score.values()):
                raise FloatingPointError('Nonfinite validation metric')
            improved = score['LA'] < best; best = min(best, score['LA'])
            log({'event': 'validation', 'step': step, **score})
        if a.stop_after and step >= a.stop_after:
            stop.append('requested_step_boundary')
        if a.max_runtime_seconds and time.monotonic()-started >= a.max_runtime_seconds:
            stop.append('walltime_budget')
        if (improved or stop or step % a.save_every == 0 or time.monotonic()-last_save >= a.save_seconds
                or step == a.steps):
            save('last.pt')
            if improved:
                save('best.pt')
            if step in (10000, 20000, 30000, 60000):
                save(f'step_{step}.pt')
            last_save = time.monotonic()
            log({'event': 'checkpoint_complete', 'step': step})
        if stop and step < a.steps:
            atomic_json(a.output/'continuation_needed.json', {'step': step, 'reasons': stop})
            raise SystemExit(75)
    atomic_json(a.output/'complete.json', {'steps': step, 'best_validation_LA': best,
                'approved_for_pi': False, 'status': 'training_complete_offline_and_execution_audit_pending'})


if __name__ == '__main__':
    main()
