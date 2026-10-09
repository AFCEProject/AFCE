"""Partition the existing global cursor without changing sample visits."""
from __future__ import annotations

import hashlib
import multiprocessing

import jax
import numpy as np
import torch

from afce_all11.resumable_data import (
    CursorBatchSampler, ReproducibleSamples, ResumableLoader, collate, worker_init,
)


class ProcessBatchSampler(CursorBatchSampler):
    def __init__(self, *args, local_rows, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_rows = tuple(local_rows)

    def __iter__(self):
        for batch in super().__iter__():
            yield [batch[index] for index in self.local_rows]


class MultihostLoader(ResumableLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device_indices = self.sharding.addressable_devices_indices_map((self.batch_size,))
        self.local_rows = sorted({
            int(row) for indices in device_indices.values()
            for row in np.arange(self.batch_size)[indices[0]]
        })
        local_size = self.batch_size // jax.process_count()
        expected = list(range(jax.process_index()*local_size, (jax.process_index()+1)*local_size))
        if self.local_rows != expected:
            raise ValueError(f'Unexpected process data placement: {self.local_rows} != {expected}')

    def start(self):
        """Start workers without consuming a batch.

        DataLoader iterator construction advances process-local host RNG state.
        Resume callers start workers before restoring the checkpointed host RNG
        so worker lifecycle bookkeeping cannot perturb the resumed trajectory.
        """
        if self.iterator is None:
            generator = torch.Generator().manual_seed(self.seed)
            kwargs = {'multiprocessing_context': multiprocessing.get_context('spawn'), 'prefetch_factor': 2} if self.workers else {}
            self.loader = torch.utils.data.DataLoader(
                ReproducibleSamples(self.dataset, self.seed),
                batch_sampler=ProcessBatchSampler(
                    len(self.dataset), self.batch_size, self.seed, self.committed, local_rows=self.local_rows),
                num_workers=self.workers, persistent_workers=self.workers > 0,
                worker_init_fn=worker_init, collate_fn=collate, generator=generator, **kwargs)
            self.iterator = iter(self.loader)

    def next(self):
        if self.pending is not None:
            raise RuntimeError('Previous batch has not been committed')
        self.start()
        item = next(self.iterator)
        expected = self.committed*self.batch_size+np.asarray(self.local_rows)
        if not np.array_equal(item['visit'], expected):
            raise ValueError('Process dataloader skipped or repeated global sample visits')
        self.pending = {'indices': item['index'].tolist(), 'visits': item['visit'].tolist()}
        batch = jax.tree.map(
            lambda value: jax.make_array_from_process_local_data(
                self.sharding, value, global_shape=(self.batch_size, *value.shape[1:])), item['payload'])
        return batch, self.pending


def create_loader(config, data_sharding):
    from openpi.training import data_loader
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        raise ValueError('Only the finite DexJoCo dataset is supported')
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.transform_dataset(dataset, data_config)
    return MultihostLoader(dataset, data_config, config.batch_size, config.seed, config.num_workers, data_sharding)


def batch_digest(batch):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(batch):
        for shard in leaf.addressable_shards:
            value = np.asarray(shard.data)
            digest.update(str(value.shape).encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.tobytes())
    return np.frombuffer(digest.digest(), dtype=np.uint8)
