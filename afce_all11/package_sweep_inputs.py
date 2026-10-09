"""Make a small, hash-verified copy of the exact existing codec training inputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    entries = []
    for pattern in ('afce_all11/*.py', 'effect_afce_v21/model/*.py',
                    'effect_afce_v21/__init__.py', 'effect_afce_v21/constants.py',
                    'dexjoco/dexjoco/sim/envs/xmls/panda_allegro_*.xml',
                    'runtime/data/*/episode_*/world_intervals.npz', 'runtime/data/world_complete.json',
                    'runtime/calibrated_statistics.pt'):
        entries.extend((f, str(f.relative_to(a.repo))) for f in sorted(a.repo.glob(pattern)))
    parquet = sorted(a.data.glob('*/data/**/*.parquet'))
    assert len(parquet) == 11, len(parquet)
    entries.extend((f, 'datasets/'+str(f.relative_to(a.data))) for f in parquet)
    assert sum(name.endswith('world_intervals.npz') for _, name in entries) == 1100
    manifest = {'files': [], 'source_repo': str(a.repo), 'source_data': str(a.data),
                'episodes': 1100, 'frames': 523763, 'mask_mode': 'legacy-none'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = a.output.with_suffix('.partial')
    with tarfile.open(temp, 'w') as tar:
        for path, name in entries:
            content = path.read_bytes()
            info = tarfile.TarInfo(name); info.size = len(content); info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))
            manifest['files'].append({'path': name, 'size': len(content),
                                     'sha256': hashlib.sha256(content).hexdigest()})
        content = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo('input_manifest.json'); info.size = len(content); info.mode = 0o644
        tar.addfile(info, io.BytesIO(content))
    temp.replace(a.output)
    print(json.dumps({'archive': str(a.output), 'bytes': a.output.stat().st_size,
                      'files': len(entries)}), flush=True)


if __name__ == '__main__':
    main()
