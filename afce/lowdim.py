"""Exact raw-unit action/state windows for both robot embodiments."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from afce.prepare import TASKS

class LowDimData:
    def __init__(self,root,split='train'):
        import pandas as pd
        self.episodes=[]; self.ends=[]; total=0
        for task in TASKS:
            df=pd.concat([pd.read_parquet(f,columns=['episode_index','frame_index','action','observation.state']) for f in sorted((Path(root)/task/'data').rglob('*.parquet'))])
            for ep,g in df.groupby('episode_index',sort=True):
                ep=int(ep)
                if split=='train' and ep>=90 or split=='val' and ep<90: continue
                g=g.sort_values('frame_index')
                a=np.stack(g.action).astype(np.float32); s=np.stack(g['observation.state']).astype(np.float32)
                ad,sd=(44,46) if task.startswith('bimanual_') else (22,23)
                if a.shape!=(len(g),ad) or s.shape!=(len(g),sd): raise ValueError(f'{task}/{ep}: wrong dimensions')
                if not np.array_equal(g.frame_index.to_numpy(),np.arange(len(g))): raise ValueError('Noncontiguous frames')
                self.episodes.append({'task':task,'episode':ep,'actions':a,'states':s})
                total+=len(a); self.ends.append(total)
        self.ends=np.asarray(self.ends); self.total=total
    def __len__(self): return self.total
    def locate(self,index):
        e=int(np.searchsorted(self.ends,index,side='right'))
        if index<0 or index>=self.total: raise IndexError(index)
        return e,index-(int(self.ends[e-1]) if e else 0)
    def sample(self,index):
        ei,t=self.locate(index); ep=self.episodes[ei]; n=len(ep['actions']); d=ep['actions'].shape[-1]; hands=d//22
        ids=np.minimum(t+np.arange(30),n-1)
        # Official LeRobot repeats the last available action at episode tails.
        # Response/visual targets at unavailable future times are explicitly masked.
        return {'action':ep['actions'][ids],'state':ep['states'][t],
            'root':np.zeros((30,9*hands),np.float32),'tips':np.zeros((30,4*hands,3),np.float32),
            'joints':np.zeros((30,16*hands),np.float32),'response':np.zeros((30,37*hands),np.float32),
            'response_valid':np.zeros(30,np.float32),
            'world':np.zeros((6,8,136),np.float32),'world_valid':np.zeros((6,8),np.float32),
            'world_query':np.zeros((6,8,4),np.float32),'world_target':np.zeros((6,8,66),np.float32)}
    def action_stats(self):
        import torch
        result={}
        for d in (22,44):
            arr=np.concatenate([e['actions'] for e in self.episodes if e['actions'].shape[-1]==d])
            result[str(d)]={'action_std':torch.from_numpy(arr.std(0).clip(1e-3)),
                            'response_std':torch.ones(37*(d//22))}
        return result

def grouped_batch(data,indices,device):
    import torch
    groups={}
    for index in indices:
        row=data.sample(int(index)); groups.setdefault(row['action'].shape[-1],[]).append(row)
    return {d:{k:torch.from_numpy(np.stack([row[k] for row in rows])).to(device) for k in rows[0]} for d,rows in groups.items()}
