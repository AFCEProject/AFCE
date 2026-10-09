"""Isolated v2 normalization repair; preserves the completed v1 experiments."""
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
import numpy as np
import torch
from afce_all11.calibrated_codec import CalibratedCodec, fit_statistics
from afce_all11.codec import losses
from afce_all11.evidence import EvidenceData
from afce_all11.lowdim import grouped_batch
from afce_all11.prepare import atomic_json
from afce_all11.train_codec import validate


def atomic_save(blob, path):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(blob, f); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--statistics', type=Path, required=True)
    p.add_argument('--fusion', choices=['query', 'poe'], required=True)
    p.add_argument('--steps', type=int, default=20000); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-every', type=int, default=500); p.add_argument('--eval-windows', type=int, default=512)
    p.add_argument('--save-every', type=int, default=250); p.add_argument('--save-seconds', type=int, default=300)
    p.add_argument('--max-runtime-seconds', type=int, default=0); p.add_argument('--resume', action='store_true')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'training.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('resume', 'save_every', 'save_seconds', 'max_runtime_seconds')}
    config.update(codec_version='calibrated_v2', condition='ASV', action_only=False, mask_mode='legacy-none',
                  effect_shape=[30, 256], independent_action_modules=[22, 44], selection='validation_LA')
    cp = args.output/'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config: raise ValueError('Run configuration changed')
    if (args.output/'last.pt').exists() and not args.resume: raise ValueError('Use --resume for existing checkpoint')
    atomic_json(cp, config)
    started = time.monotonic(); stop = []
    def signal_stop(sig, _): stop.append(signal.Signals(sig).name)
    for sig in (signal.SIGTERM, signal.SIGUSR1): signal.signal(sig, signal_stop)
    print(json.dumps({'event': 'loading_data', 'fusion': args.fusion}), flush=True)
    train = EvidenceData(args.data, args.evidence, 'train', 'legacy-none')
    val = EvidenceData(args.data, args.evidence, 'val', 'legacy-none')
    args.statistics.parent.mkdir(parents=True, exist_ok=True)
    statistics_lock = args.statistics.with_suffix('.lock').open('a')
    fcntl.flock(statistics_lock, fcntl.LOCK_EX)
    stat_contract = {'data': str(args.data.resolve()), 'evidence': str(args.evidence.resolve()),
                     'split': 'episodes0-89', 'frames': len(train), 'version': 'calibrated_v2'}
    if args.statistics.exists():
        saved = torch.load(args.statistics, map_location='cpu', weights_only=False)
        if saved['contract'] != stat_contract: raise ValueError('Statistics contract differs')
        stats = saved['stats']
    else:
        stats = fit_statistics(train); atomic_save({'stats': stats, 'contract': stat_contract}, args.statistics)
    fcntl.flock(statistics_lock, fcntl.LOCK_UN); statistics_lock.close()
    stats_sha = hashlib.sha256(args.statistics.read_bytes()).hexdigest()
    torch.manual_seed(args.seed)
    model = CalibratedCodec(args.fusion, 'ASV', stats).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rng = np.random.default_rng(args.seed); step = 0; best = float('inf'); score = None
    if args.resume:
        blob = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
        if blob['statistics_sha256'] != stats_sha: raise ValueError('Frozen statistics changed')
        model.load_state_dict(blob['model']); optimizer.load_state_dict(blob['optimizer'])
        step, best, score = blob['step'], blob['best'], blob['val']
        rng.bit_generator.state = blob['numpy_rng']; random.setstate(blob['python_rng'])
        np.random.set_state(blob['numpy_global_rng']); torch.set_rng_state(blob['torch_rng'])
        torch.cuda.set_rng_state_all(blob['cuda_rng_all'])
    vi = np.random.default_rng(103).integers(0, len(val), args.eval_windows)
    model.train(); last_save = time.monotonic()
    def log(row):
        row = {'unix_time': time.time(), 'elapsed_s': time.monotonic()-started, **row}
        with (args.output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    log({'event': 'train_started', 'step': step, 'fusion': args.fusion, 'statistics_sha256': stats_sha,
         'train_frames': len(train), 'val_frames': len(val), 'batch_size': args.batch_size})
    for step in range(step+1, args.steps+1):
        optimizer.zero_grad(set_to_none=True); metrics = {}
        indices = rng.integers(0, len(train), args.batch_size)
        lam = .05 + .15*min(step/1000, 1)
        for d, b in grouped_batch(train, indices, 'cuda:0').items():
            out = model(b); values = losses(out, b, stats, lam, lam)
            if not all(torch.isfinite(v) for v in values.values()): raise FloatingPointError(f'Loss at {step}')
            weight = len(b['action'])/args.batch_size
            (values['loss']*weight).backward()
            for key, value in values.items(): metrics[key] = metrics.get(key, 0.)+float(value.detach())*weight
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad): raise FloatingPointError(f'Gradient at {step}')
        optimizer.step()
        if step == 1 or step % 50 == 0: log({'event': 'train', 'step': step, **metrics, 'grad_norm': float(grad)})
        improved = False
        if step % args.eval_every == 0 or step == args.steps:
            score = validate(model, val, stats, 'cuda:0', vi, False)
            improved = score['LA'] < best; best = min(best, score['LA'])
            log({'event': 'validation', 'step': step, **score})
        if args.max_runtime_seconds and time.monotonic()-started >= args.max_runtime_seconds: stop.append('walltime_budget')
        if improved or stop or step % args.save_every == 0 or time.monotonic()-last_save >= args.save_seconds or step == args.steps:
            blob = {'schema': 'afce_all11_calibrated_v2', 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'step': step, 'best': best, 'val': score, 'stats': stats,
                    'config': config, 'statistics_sha256': stats_sha, 'numpy_rng': rng.bit_generator.state,
                    'numpy_global_rng': np.random.get_state(), 'python_rng': random.getstate(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng_all': torch.cuda.get_rng_state_all()}
            saved_at = time.monotonic(); atomic_save(blob, args.output/'last.pt')
            if improved: atomic_save(blob, args.output/'best.pt')
            last_save = time.monotonic(); log({'event': 'checkpoint_complete', 'step': step, 'save_seconds': last_save-saved_at})
        if stop and step < args.steps:
            atomic_json(args.output/'continuation_needed.json', {'step': step, 'reasons': stop})
            raise SystemExit(75)
    atomic_json(args.output/'complete.json', {'steps': step, 'best_validation_LA': best,
        'status': 'codec_training_completed_not_rollout_validated'})


if __name__ == '__main__': main()
