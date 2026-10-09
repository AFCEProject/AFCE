"""Observed robot response and sparse, timestamp-aligned visual consequences."""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from afce_all11.lowdim import LowDimData

@lru_cache(maxsize=3)
def hand_model(side):
    import mujoco
    suffix='' if side=='single' else '_'+side
    name='copy' if side=='single' else side
    path=Path(__file__).resolve().parents[1]/f'dexjoco/dexjoco/sim/envs/xmls/panda_allegro_{name}.xml'
    tree=ET.parse(path).getroot()
    # Exact joint transforms and collision-center locations from the actual
    # Panda+Allegro task XML. Mesh appearance and dynamics are irrelevant to FK.
    for child in list(tree):
        if child.tag not in ('compiler','default','worldbody'): tree.remove(child)
    for body in tree.iter('body'):
        for finger in ('ff','mf','rf','th'):
            if body.get('name')==f'{finger}_tip{suffix}':
                body.findall('geom')[-1].set('name',f'afce_tip_{finger}')
    for geom in tree.iter('geom'):
        for key in ('mesh','material','fromto','hfield'): geom.attrib.pop(key,None)
        geom.set('type','sphere'); geom.set('size','.001')
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))
    ids=[model.joint(f'{finger}j{j}{suffix}').qposadr[0] for finger in ('ff','mf','rf','th') for j in range(4)]
    tips=[model.geom(f'afce_tip_{finger}').id for finger in ('ff','mf','rf','th')]
    return model,np.asarray(ids),tips,model.site('attachment_site'+suffix).id

def exact_tips(joints,side):
    import mujoco
    model,ids,tips,grip=hand_model(side); data=mujoco.MjData(model)
    out=[]
    for q in joints:
        data.qpos[ids]=q; mujoco.mj_kinematics(model,data)
        r=data.site_xmat[grip].reshape(3,3)
        out.append((data.geom_xpos[tips]-data.site_xpos[grip])@r)
    return np.asarray(out,dtype=np.float32)

def geometry(states):
    # _compute_observation reads MuJoCo framequat sensors verbatim: wxyz.
    dual=states.shape[-1]==46
    poses=[states[:,:7]]; joints=[states[:,14:30] if dual else states[:,7:23]]
    if dual: poses.append(states[:,7:14]); joints.append(states[:,30:46])
    all_hands=[]
    for side,pose,q in zip(('right','left') if dual else ('single',),poses,joints):
        r=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix().astype(np.float32)
        tip=exact_tips(q,side)
        xyz=np.einsum('tij,tkj->tki',r,tip)+pose[:,:3,None].transpose(0,2,1)
        all_hands.append((r,pose[:,:3],xyz,q))
    return all_hands

class EvidenceData(LowDimData):
    def __init__(self,root,evidence,split='train',mask_mode='verified'):
        super().__init__(root,split)
        if evidence is None: raise ValueError('Evidence cache required')
        self.evidence=Path(evidence)
        manifest=json.loads((self.evidence/'world_complete.json').read_text())
        if not manifest.get('complete') or manifest.get('mask_mode','verified')!=mask_mode:
            raise ValueError('Visual evidence must be complete and match the selected mask protocol')
        if manifest.get('descriptor_dim')!=136:
            raise ValueError('World evidence must include both starting appearance and its change')
        if mask_mode=='verified' and manifest.get('missing_masks'):
            raise ValueError('Verified-mask protocol requires real robot masks')
        self.geo={}; self.world={}
    def sample(self,index):
        ei,t=self.locate(index); ep=self.episodes[ei]; n=len(ep['actions'])
        b=super().sample(index)
        if ei not in self.geo: self.geo[ei]=geometry(ep['states'])
        ids=np.minimum(t+np.arange(30),n-1); nxt=np.minimum(ids+1,n-1)
        roots=[];tips=[];joints=[];response=[]
        for r,p,tip,q in self.geo[ei]:
            root_delta=(p[nxt]-p[ids])@r[t]
            relative=np.einsum('tji,tjk->tik',r[ids],r[nxt])
            rot6=relative[:,:,:2].transpose(0,2,1).reshape(30,6)
            tip_delta=np.einsum('tkj,ji->tki',tip[nxt]-tip[ids],r[t])
            dq=q[nxt]-q[ids]
            root=np.concatenate([root_delta,rot6],-1)
            roots.append(root);tips.append(tip_delta);joints.append(dq)
            response.append(np.concatenate([root,tip_delta.reshape(30,12),dq],-1))
        b.update(root=np.concatenate(roots,-1),tips=np.concatenate(tips,1),joints=np.concatenate(joints,-1),response=np.concatenate(response,-1),response_valid=(t+np.arange(30)+1<n).astype(np.float32))
        if ei not in self.world:
            path=self.evidence/ep['task']/f"episode_{ep['episode']:06d}"/'world_intervals.npz'
            with np.load(path) as z: self.world[ei]={k:z[k] for k in z.files}
        w=self.world[ei]
        take=np.flatnonzero((w['start']>=t)&(w['end']<=min(t+30,n-1)))
        if len(take)>6: raise ValueError('Unexpected number of observed intervals')
        for k,i in enumerate(take):
            raw=w['raw'][i].copy(); raw[:,70]=(int(w['start'][i])-t)/30.;raw[:,71]=(int(w['end'][i])-t)/30.
            b['world'][k]=raw;b['world_valid'][k]=w['valid'][i]
            b['world_target'][k]=raw[:,:66]; b['world_query'][k]=raw[:,[66,67,70,71]]
        return b
    def action_stats(self):
        import torch
        result=super().action_stats(); by_dim={22:[],44:[]}
        for idx in np.random.default_rng(192).integers(0,len(self),2048):
            b=self.sample(int(idx)); by_dim[b['action'].shape[-1]].append(b['response'][b['response_valid']>0])
        for d,arr in by_dim.items():
            std=np.concatenate(arr).std(0).clip(1e-3)
            result[str(d)]['response_std']=torch.from_numpy(std)
        return result
