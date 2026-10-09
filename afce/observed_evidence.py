"""Exact causal S/V conditions; never interpolate sparse/future visual evidence."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np

from afce.evidence import EvidenceData

OBSERVATION_SCHEMA = 'afce_dense_observed_visual_v1'


def history_indices(origin, horizon):
    if horizon not in (1, 2) or origin < 0:
        raise ValueError('Invalid causal observation window')
    raw = origin+np.arange(1-horizon, 1, dtype=np.int64)
    return raw.clip(0), raw >= 0


class ObservedEvidenceData(EvidenceData):
    def __init__(self, root, evidence, observed_visual, split='train', mask_mode='legacy-none',
                 observation_horizon=2):
        super().__init__(root, evidence, split, mask_mode)
        history_indices(0, observation_horizon)
        self.observation_horizon = observation_horizon
        self.observed_visual_root = Path(observed_visual)
        manifest = json.loads((self.observed_visual_root/'manifest.json').read_text())
        pca_hash = hashlib.sha256((Path(evidence)/'fitted_pca.npz').read_bytes()).hexdigest()
        if (not manifest.get('complete') or manifest.get('schema') != OBSERVATION_SCHEMA
                or manifest.get('stride') != 1 or manifest.get('feature_shape') != [196, 64]
                or manifest.get('pca_sha256') != pca_hash):
            raise ValueError('Exact per-frame visual conditions with matching PCA are required')
        self.observed_manifest = manifest
        self.observed_cache = OrderedDict()

    def sample(self, index):
        ei, origin = self.locate(index)
        ep = self.episodes[ei]
        batch = super().sample(index)
        indices, valid = history_indices(origin, self.observation_horizon)
        if ei not in self.observed_cache:
            directory = self.observed_visual_root/ep['task']/f"episode_{ep['episode']:06d}"
            receipt = json.loads((directory/'ready.json').read_text())
            frames = np.load(directory/'frames.npy')
            features = np.load(directory/'features.npy', mmap_mode='r')
            n = len(ep['actions'])
            if (receipt.get('schema') != OBSERVATION_SCHEMA or receipt.get('frames') != n
                    or receipt.get('pca_sha256') != self.observed_manifest['pca_sha256']
                    or not np.array_equal(frames, np.arange(n))
                    or features.shape != (n, 196, 64)):
                raise ValueError(f'Invalid exact observed visual cache: {directory}')
            self.observed_cache[ei] = features
            # Bound open memory maps even when training samples all 990 episodes.
            if len(self.observed_cache) > 32:
                self.observed_cache.popitem(last=False)
        self.observed_cache.move_to_end(ei)
        visual = np.asarray(self.observed_cache[ei][indices], dtype=np.float32)
        if not np.isfinite(visual).all():
            raise ValueError('Nonfinite observed visual features')
        batch.update(observed_state=ep['states'][indices].copy(), observed_visual=visual,
                     observed_valid=valid, observed_indices=indices,
                     window_start=np.asarray(origin, dtype=np.int64))
        return batch


def evidence_for_config(root, evidence, split, config, observed_visual=None):
    if config.get('codec_version') == 'local_effect_observed_v2':
        cache = observed_visual or config.get('observed_visual')
        if not cache:
            raise ValueError('Observed visual cache is required for this codec')
        return ObservedEvidenceData(root, evidence, cache, split, config.get('mask_mode', 'legacy-none'),
                                    config['observation_horizon'])
    return EvidenceData(root, evidence, split, config.get('mask_mode', 'legacy-none'))
