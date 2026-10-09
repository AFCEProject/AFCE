"""π0.5 flow matching on frozen 30x256 E, retaining official input transforms.

The codec stays frozen and outside the policy input. At inference only predicted
E and the current unnormalized robot state reach its original Torch decoder.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.models import gemma, model as model_lib
from openpi.training import config as config_lib, weight_loaders
from openpi import transforms
from openpi.shared import array_typing as at

class EffectPi(Pi0):
    def __init__(self,config,rngs):
        super().__init__(config,rngs)
        width=gemma.get_config(config.action_expert_variant).width
        self.action_dim=256
        self.action_in_proj=nnx.Linear(256,width,rngs=rngs)
        self.action_out_proj=nnx.Linear(width,256,rngs=rngs)

    def compute_loss(self,rng,observation,actions,*,train=False):
        if observation.effect_target is None:
            raise ValueError('AFCE requires a frozen, frame-aligned effect target')
        target=observation.effect_target
        if target.shape[-2:]!=(30,256): raise ValueError(target.shape)
        return super().compute_loss(rng,observation,jax.lax.stop_gradient(target),train=train)

@dataclasses.dataclass(frozen=True)
class EffectPiConfig(Pi0Config):
    def create(self,rng): return EffectPi(self,nnx.Rngs(rng))
    def inputs_spec(self,*,batch_size=1):
        obs,actions=super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            obs=dataclasses.replace(obs,effect_target=jax.ShapeDtypeStruct((batch_size,30,256),jnp.float32))
        return obs,actions

@dataclasses.dataclass(frozen=True)
class EffectWeights:
    params_path: str
    def load(self,params):
        loaded=model_lib.restore_params(self.params_path,restore_type=np.ndarray)
        # The 44-D raw-action projections are incompatible with 256-D E.
        loaded={k:v for k,v in loaded.items() if k not in ('action_in_proj','action_out_proj')}
        return weight_loaders._merge_params(loaded,params,missing_regex=r'.*(lora|action_in_proj|action_out_proj).*')

@dataclasses.dataclass(frozen=True)
class EffectDataConfig(config_lib.DexJoCoMultiTaskDataConfig):
    afce_cache_root: Path | None = None
    def create(self,assets_dirs,model_config):
        base=super().create(assets_dirs,model_config)
        repack=base.repack_transforms.inputs[0]
        repack=transforms.RepackTransform({**repack.structure,'effect_target':'effect_target'})
        return dataclasses.replace(base,afce_cache_root=self.afce_cache_root,
            repack_transforms=transforms.Group(inputs=[repack]))

def effect_config(cfg,cache,init):
    manifest=json.loads((Path(cache)/'manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('total_frames')!=523763:
        raise ValueError('All 523763 frame-aligned E targets must be exported before π training')
    model=EffectPiConfig(**dataclasses.asdict(cfg.model))
    data=EffectDataConfig(**{f.name:getattr(cfg.data,f.name) for f in dataclasses.fields(cfg.data)},afce_cache_root=Path(cache))
    return dataclasses.replace(cfg,model=model,data=data,weight_loader=EffectWeights(str(init)))

class EffectCacheDataset:
    def __init__(self,dataset,root):
        self.dataset,self.root=dataset,Path(root)
        with np.load(self.root.parent/'normalization.npz') as normalization:
            self.mean=normalization['mean'].astype(np.float32)
            self.std=normalization['std'].astype(np.float32)
        self.cache={}
    def __getstate__(self):
        state=dict(self.__dict__)
        state['cache']={}
        return state
    def __len__(self): return len(self.dataset)
    def __getitem__(self,index):
        sample=dict(self.dataset[index])
        ep,t=int(sample['episode_index']),int(sample['frame_index'])
        if ep not in self.cache:
            self.cache[ep]=np.load(self.root/f'episode_{ep:06d}.npy',mmap_mode='r')
        arr=self.cache[ep]
        if t<0 or t>=len(arr): raise IndexError(f'Missing E at {self.root.name}/{ep}/{t}')
        # Never clamp indices or forward-fill a different future window.
        value=arr[t].astype(np.float32)
        if value.shape!=(30,256) or not np.isfinite(value).all(): raise ValueError('Invalid E cache')
        sample['effect_target']=(value-self.mean)/self.std
        return sample

class DecodedEffectPolicy:
    def __init__(self,latent_policy,codec,mean,std,*,task,device='cuda:0'):
        self.policy,self.codec,self.task,self.device=latent_policy,codec.eval().to(device),task,device
        self.mean,self.std=mean,std
        self.single=not task.startswith('bimanual_')
        for p in self.codec.parameters(): p.requires_grad_(False)
    @property
    def metadata(self): return self.policy.metadata
    def reset(self):
        if hasattr(self.policy,'reset'): self.policy.reset()
    def infer(self,obs,**kwargs):
        import torch,time
        # Caller uses the exact baseline wire input, with raw state at `state`.
        state=np.asarray(obs['state'],np.float32)
        dim=23 if self.single else 46
        state=state[:dim]
        if state.shape!=(dim,): raise ValueError('Raw current state is required by the decoder')
        started=time.monotonic()
        result=self.policy.infer(obs,**kwargs)
        effect=np.asarray(result['actions'],np.float32)
        if effect.shape!=(30,256): raise ValueError('Latent policy output was truncated or action-normalized')
        with torch.inference_mode():
            e=torch.as_tensor(effect*self.std+self.mean,device=self.device)[None]
            s=torch.as_tensor(state,device=self.device)[None]
            action=self.codec.decode(e,s)[0].float().cpu().numpy()
        result['actions']=action
        result['policy_timing']={'infer_ms':1000*(time.monotonic()-started)}
        return result
