"""Preserve the fixed audit, then evaluate every held-out window and more episodes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from afce_all11.evidence import EvidenceData
from afce_all11.eval_codec import evaluate
from afce_all11.eval_geometry import geometric_errors
from afce_all11.eval_tuning import robustness
from afce_all11.prepare import TASKS, atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'data', 'evidence', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--observed-visual', type=Path)
    a = p.parse_args(); torch.set_num_threads(4)
    from afce_all11.observed_evidence import evidence_for_config
    config = torch.load(a.checkpoint, map_location='cpu', weights_only=False)['config']
    data = evidence_for_config(a.data, a.evidence, 'val', config, a.observed_visual)
    fixed = {t: [] for t in TASKS}; dense = {t: [] for t in TASKS}; samples = []
    rng = np.random.default_rng(20260917)
    for i, ep in enumerate(data.episodes):
        start = int(data.ends[i-1]) if i else 0; n = len(ep['actions'])
        local = rng.choice(n, min(32, n), replace=False)
        fixed[ep['task']].extend((local+start).tolist())
        dense[ep['task']].extend(range(start, start+n))
        samples.extend({'task': ep['task'], 'episode': ep['episode'], 'frame': int(t)} for t in local)
    report = {'split': 'episode90-99', 'samples': samples, 'runs': [], 'complete': False,
              'validation_not_independent_test': True, 'validation_windows': len(data),
              'extra_episodes_are_training_diagnostics': True,
              'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    row = evaluate(a.checkpoint, data, fixed, 32)
    row['robustness'] = robustness(a.checkpoint, data, fixed)
    row['geometry'] = geometric_errors(a.checkpoint, data, fixed)
    report['runs'].append(row); atomic_json(a.output, report)
    report['all_validation_frames'] = evaluate(a.checkpoint, data, dense, 32)
    report['all_validation_geometry'] = geometric_errors(a.checkpoint, data, dense)
    atomic_json(a.output, report)
    del data
    # Forty distinct training demonstrations supplement the ten held-out ones.
    # They are NEVER included in validation metrics or treated as generalization.
    train = evidence_for_config(a.data, a.evidence, 'train', config, a.observed_visual)
    chosen = set(np.linspace(0, 89, 40, dtype=int).tolist())
    extra = {t: [] for t in TASKS}; selected = []
    rng = np.random.default_rng(20260918)
    for i, ep in enumerate(train.episodes):
        if ep['episode'] not in chosen: continue
        start = int(train.ends[i-1]) if i else 0
        local = rng.choice(len(ep['actions']), min(32, len(ep['actions'])), replace=False)
        extra[ep['task']].extend((local+start).tolist())
        selected.append({'task': ep['task'], 'episode': ep['episode'], 'frames': local.tolist()})
    report['training_diagnostic_samples'] = selected
    report['training_diagnostic'] = evaluate(a.checkpoint, train, extra, 32)
    report['complete'] = True; atomic_json(a.output, report)
    print(json.dumps({'event': 'expanded_audit_complete', 'validation_windows': report['validation_windows'],
                      'additional_training_episodes': len(selected)}), flush=True)


if __name__ == '__main__':
    main()
