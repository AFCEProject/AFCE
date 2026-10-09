"""A committed batch cursor independent of worker prefetch and restarts.

The finite dataset is shuffled without replacement each epoch. Only an optimizer
update commits a batch; prefetched batches are reproducible and can be discarded.
"""
from __future__ import annotations

import contextlib
import hashlib
import multiprocessing
import os
import random
import struct
import numpy as np
import torch


@contextlib.contextmanager
def sample_rng(seed,visit):
    key=int.from_bytes(hashlib.blake2b(struct.pack('<QQ',seed,visit),digest_size=8).digest(),'little')
    py_state,np_state,torch_state=random.getstate(),np.random.get_state(),torch.get_rng_state()
    try:
        random.seed(key);np.random.seed(key&0xffffffff);torch.random.default_generator.manual_seed(key&0x7fffffffffffffff)
        yield
    finally:
        random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(torch_state)


class ReproducibleSamples:
    def __init__(self,dataset,seed): self.dataset,self.seed=dataset,seed
    def __len__(self): return len(self.dataset)
    def __getitem__(self,key):
        index,visit=key
        with sample_rng(self.seed,visit): payload=self.dataset[index]
        return {'payload':payload,'index':np.int64(index),'visit':np.int64(visit)}


class CursorBatchSampler:
    def __init__(self,length,batch_size,seed,start_batch=0,shuffle=True):
        self.length,self.batch_size,self.seed=length,batch_size,seed
        self.start_batch,self.shuffle=start_batch,shuffle
        self.batches_per_epoch=length//batch_size
        if self.batches_per_epoch<1: raise ValueError('Dataset smaller than one batch')

    def permutation(self,epoch):
        if not self.shuffle: return np.arange(self.length,dtype=np.int64)
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,epoch,314159]))).permutation(self.length)

    def __iter__(self):
        number=self.start_batch;previous_epoch=None
        while True:
            epoch,offset=divmod(number,self.batches_per_epoch)
            if epoch!=previous_epoch: order=self.permutation(epoch);previous_epoch=epoch
            lo=offset*self.batch_size
            yield [(int(index),number*self.batch_size+i) for i,index in enumerate(order[lo:lo+self.batch_size])]
            number+=1


def worker_init(_):
    # Data transforms use CPU. Never create extra GPU clients in workers.
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    torch.set_num_threads(1)


def collate(items):
    import jax
    return jax.tree.map(lambda *xs:np.stack([np.asarray(x) for x in xs],axis=0),*items)


class ResumableLoader:
    schema='afce_committed_cursor_v1'
    def __init__(self,dataset,data_config,batch_size,seed,workers,sharding=None):
        self.dataset,self._data_config=dataset,data_config
        self.batch_size,self.seed,self.workers=batch_size,seed,workers
        self.sharding=sharding;self.committed=0;self.pending=None;self.iterator=None;self.loader=None

    def data_config(self): return self._data_config

    def state_dict(self):
        sampler=CursorBatchSampler(len(self.dataset),self.batch_size,self.seed,self.committed)
        epoch,offset=divmod(self.committed,sampler.batches_per_epoch)
        order=sampler.permutation(epoch)
        return {'schema':self.schema,'committed_batches':self.committed,'epoch':epoch,'batch_offset':offset,
                'length':len(self.dataset),'batch_size':self.batch_size,'seed':self.seed,
                'permutation':order,'permutation_sha256':hashlib.sha256(order.tobytes()).hexdigest(),
                'rng_policy':'blake2b(seed,global_sample_visit); PCG64 epoch permutation'}

    def load_state_dict(self,state):
        expected=(self.schema,len(self.dataset),self.batch_size,self.seed)
        got=(state['schema'],state['length'],state['batch_size'],state['seed'])
        if got!=expected: raise ValueError(f'Data resume contract changed: {got} != {expected}')
        if self.iterator is not None: raise RuntimeError('Restore the cursor before starting workers')
        sampler=CursorBatchSampler(len(self.dataset),self.batch_size,self.seed,state['committed_batches'])
        epoch,offset=divmod(state['committed_batches'],sampler.batches_per_epoch)
        if (epoch,offset)!=(state['epoch'],state['batch_offset']): raise ValueError('Invalid epoch cursor')
        expected_order=sampler.permutation(epoch)
        if not np.array_equal(expected_order,state['permutation']): raise ValueError('Permutation algorithm/version changed')
        if hashlib.sha256(expected_order.tobytes()).hexdigest()!=state['permutation_sha256']: raise ValueError('Permutation checksum mismatch')
        self.committed=state['committed_batches']

    def next(self):
        if self.pending is not None: raise RuntimeError('Previous batch has not been committed')
        if self.iterator is None:
            gen=torch.Generator().manual_seed(self.seed)
            kwargs={'multiprocessing_context':multiprocessing.get_context('spawn'),'prefetch_factor':2} if self.workers else {}
            self.loader=torch.utils.data.DataLoader(ReproducibleSamples(self.dataset,self.seed),
                batch_sampler=CursorBatchSampler(len(self.dataset),self.batch_size,self.seed,self.committed),
                num_workers=self.workers,persistent_workers=self.workers>0,worker_init_fn=worker_init,
                collate_fn=collate,generator=gen,**kwargs)
            self.iterator=iter(self.loader)
        item=next(self.iterator)
        expected=np.arange(self.committed*self.batch_size,(self.committed+1)*self.batch_size)
        if not np.array_equal(item['visit'],expected): raise ValueError('Dataloader skipped or repeated a batch')
        self.pending={'indices':item['index'].tolist(),'visits':item['visit'].tolist()}
        batch=item['payload']
        if self.sharding is not None:
            import jax
            batch=jax.tree.map(lambda x:jax.make_array_from_process_local_data(self.sharding,x),batch)
        return batch,self.pending

    def commit(self,optimizer_updates):
        if self.pending is None or optimizer_updates!=self.committed+1: raise ValueError('Optimizer/data cursor mismatch')
        self.committed=optimizer_updates;self.pending=None

    def close(self):
        if self.iterator is not None and hasattr(self.iterator,'_shutdown_workers'): self.iterator._shutdown_workers()
        self.iterator=None;self.loader=None;self.pending=None


def create_loader(config,sharding):
    from openpi.training import data_loader as dl
    data_config=config.data.create(config.assets_dirs,config.model)
    if data_config.rlds_data_dir is not None: raise ValueError('Exact AFCE resume supports the finite DexJoCo dataset only')
    dataset=dl.create_torch_dataset(data_config,config.model.action_horizon,config.model)
    dataset=dl.transform_dataset(dataset,data_config)
    return ResumableLoader(dataset,data_config,config.batch_size,config.seed,config.num_workers,sharding)
