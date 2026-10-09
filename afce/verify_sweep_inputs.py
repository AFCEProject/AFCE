"""Verify the transferred, unchanged input bundle before deploying local sweep code."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); path = a.root/'input_manifest.json'
    m = json.loads(path.read_text())
    for row in m['files']:
        f = a.root/row['path']
        assert f.resolve().is_relative_to(a.root.resolve()) if hasattr(Path, 'is_relative_to') else '..' not in Path(row['path']).parts
        assert f.stat().st_size == row['size'], row['path']
        h = hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        assert h.hexdigest() == row['sha256'], row['path']
    result = {'passed': True, 'files': len(m['files']), 'episodes': m['episodes'], 'frames': m['frames'],
              'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (a.root/'inputs_verified.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
