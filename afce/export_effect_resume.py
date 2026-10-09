"""Episode-atomic export of frozen E targets, with verified restart support."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from afce.evidence import EvidenceData
from afce.export_effect import load_codec
from afce.prepare import atomic_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'data', 'evidence', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=128)
    a = p.parse_args(); a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output/'export.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    source = Path(__file__).parent
    contract = {'checkpoint_sha256': sha(a.checkpoint), 'data': str(a.data.resolve()),
                'evidence': str(a.evidence.resolve()), 'dtype': 'float16', 'shape': [30, 256],
                'normalization_split': 'episode0-89', 'batch_size': a.batch_size,
                'source': {n: sha(source/n) for n in ('export_effect_resume.py', 'export_effect.py',
                    'codec.py', 'calibrated_codec.py', 'tuned_codec.py', 'evidence.py', 'lowdim.py')}}
    cp = a.output/'export_contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Export contract changed; use a new output directory')
    atomic_json(cp, contract)
    manifest_path = a.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('complete'):
        print('Frozen export already complete', flush=True); return
    model, blob = load_codec(a.checkpoint, 'cuda:0')
    data = EvidenceData(a.data, a.evidence, 'all', blob['config'].get('mask_mode', 'verified'))
    if len(data) != 523763 or len(data.episodes) != 1100:
        raise ValueError('Official all11 frame/episode count changed')
    record = {'complete': False, 'total_frames': len(data), 'checkpoint_sha256': contract['checkpoint_sha256'],
              'codec_step': blob['step'], 'shape': [30, 256], 'dtype': 'float16',
              'mask_mode': blob['config'].get('mask_mode'), 'normalization_split': 'episode0-89', 'episodes': []}
    offset = 0; total = 0; sum1 = np.zeros(256, np.float64); sum2 = sum1.copy(); started = time.time()
    for ei, ep in enumerate(data.episodes):
        n = len(ep['actions']); dest = a.output/ep['task']/f"episode_{ep['episode']:06d}.npy"
        dest.parent.mkdir(exist_ok=True); receipt = dest.with_suffix('.json')
        if dest.exists() and receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['checkpoint_sha256'] != contract['checkpoint_sha256'] or saved['sha256'] != sha(dest):
                raise ValueError('Cache checksum mismatch: '+str(dest))
            value = np.load(dest, mmap_mode='r')
            if value.shape != (n, 30, 256) or value.dtype != np.float16:
                raise ValueError('Invalid existing E shape/dtype')
        else:
            tmp = dest.with_suffix('.tmp.npy')
            value = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=(n, 30, 256))
            with torch.inference_mode():
                for lo in range(0, n, a.batch_size):
                    rows = [data.sample(offset+i) for i in range(lo, min(n, lo+a.batch_size))]
                    batch = {k: torch.as_tensor(np.stack([r[k] for r in rows]), device='cuda:0') for k in rows[0]}
                    encoded = model.encode(batch).float().cpu().numpy()
                    value[lo:lo+len(rows)] = encoded.astype(np.float16)
                    if not np.isfinite(value[lo:lo+len(rows)]).all():
                        raise FloatingPointError('Nonfinite float16 E target')
            value.flush(); del value; tmp.replace(dest)
            atomic_json(receipt, {'sha256': sha(dest), 'checkpoint_sha256': contract['checkpoint_sha256'],
                                 'frames': n, 'task': ep['task'], 'episode': ep['episode']})
            value = np.load(dest, mmap_mode='r')
        if ep['episode'] < 90:
            for lo in range(0, n, 128):
                x = value[lo:lo+128].astype(np.float64).reshape(-1, 256)
                sum1 += x.sum(0); sum2 += (x*x).sum(0); total += len(x)
        del value
        data.geo.pop(ei, None); data.world.pop(ei, None)
        offset += n
        record['episodes'].append({'task': ep['task'], 'episode': ep['episode'], 'frames': n})
        record.update(exported_frames=offset, updated_unix_time=time.time())
        atomic_json(manifest_path, record)
        print(json.dumps({'event': 'episode_exported', 'episodes': len(record['episodes']), 'frames': offset,
                          'total_frames': len(data), 'elapsed_seconds': time.time()-started}), flush=True)
    mean = sum1/total; std = np.sqrt(np.maximum(sum2/total-mean*mean, 0)).clip(1e-4)
    np.savez(a.output/'normalization.tmp.npz', mean=mean.astype(np.float32), std=std.astype(np.float32), training_tokens=total)
    (a.output/'normalization.tmp.npz').replace(a.output/'normalization.npz')
    shutil.copy2(a.checkpoint, a.output/'codec.tmp.pt'); (a.output/'codec.tmp.pt').replace(a.output/'codec.pt')
    record.update(complete=True, completed_unix_time=time.time())
    atomic_json(manifest_path, record)


if __name__ == '__main__':
    main()
