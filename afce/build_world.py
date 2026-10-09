"""Fit train-only PCA and local visual matches; refuse unavailable robot masks."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from afce.prepare import TASKS,atomic_json

def match_intervals(a,b,masks,weight,xy,nearby,mean=None):
    """Evaluate independent intervals together, preserving the scalar matcher."""
    scores=torch.nn.functional.normalize(a,dim=-1)@torch.nn.functional.normalize(b,dim=-1).transpose(-1,-2)
    scores=scores.masked_fill(~nearby,-torch.inf)
    val,idx=scores.topk(5,-1); prob=(val/.07).softmax(-1)
    batch=torch.arange(len(a),device=a.device)[:,None,None]
    matched=(prob[...,None]*b[batch,idx]).sum(2)
    disp=(prob[...,None]*xy[idx]).sum(2)-xy
    conf=1-(-(prob*prob.clamp_min(1e-8).log()).sum(-1))/np.log(5)
    conf=conf*(1-.8*masks)
    change=1-torch.nn.functional.cosine_similarity(a,matched,dim=-1)
    activity=conf*change
    dynamic=activity.topk(6,-1).indices
    refscore=activity.clone();refscore.scatter_(1,dynamic,torch.inf)
    refscore=refscore.masked_fill(conf<=.05,torch.inf)
    reference=refscore.topk(2,-1,largest=False).indices
    sel=torch.cat([dynamic,reference],-1);df=(matched-a)@weight.T
    rows=torch.arange(len(a),device=a.device)[:,None]
    kind=torch.tensor([1]*6+[0]*2,device=a.device).expand(len(a),-1)[...,None]
    # The specification includes both starting appearance and its change.
    # Keep the fixed target/query layout in columns 0:72, append P(F^-).
    appearance=(a-(mean if mean is not None else 0.))@weight.T
    raw=torch.cat([df[rows,sel],disp[rows,sel],xy[sel],conf[rows,sel,None],kind,
                   torch.zeros(len(a),8,2,device=a.device),appearance[rows,sel]],-1)
    valid=(conf[rows,sel]>1e-6)&torch.isfinite(raw).all(-1)
    return raw,valid.float()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--mask-mode',choices=['verified','legacy-none'],default='verified')
    p.add_argument('--task',action='append',choices=TASKS)
    args=p.parse_args(); root=args.evidence
    manifest=json.loads((root/'manifest.json').read_text())
    if not manifest['complete']: raise ValueError('Sparse feature preparation is not complete')
    episodes=[root/t/f'episode_{ep:06d}' for t in TASKS for ep in range(100)]
    missing=[str(p) for p in episodes if not (p/'robot_mask.npy').exists() or not (p/'mask_verified.json').exists()]
    if missing and args.mask_mode=='verified':
        atomic_json(root/'world_complete.json',{'complete':False,'missing_masks':missing})
        raise ValueError(f'{len(missing)} episodes lack verified robot segmentation')
    mode_file=root/'world_config.json'
    mode={'mask_mode':args.mask_mode,'pca':'fitted_train_only_64','descriptor_dim':136,'interval_stride':5,
          'layout':'df64_disp2_xy2_conf_kind_t0_t1_start_appearance64'}
    if mode_file.exists():
        if json.loads(mode_file.read_text())!=mode:
            raise ValueError('Cannot mix world caches from different protocols')
    else:
        atomic_json(mode_file,mode)
    torch.set_num_threads(4); torch.manual_seed(42)
    pp=root/'fitted_pca.npz'
    if not pp.exists():
        rng=np.random.default_rng(42); samples=[]
        for path in episodes:
            if int(path.name.split('_')[-1])>=90: continue
            f=np.load(path/'features.npy',mmap_mode='r')
            for anchor in rng.integers(0,len(f),2):
                samples.append(f[anchor,rng.choice(196,8,replace=False)].astype(np.float32))
        x=torch.from_numpy(np.concatenate(samples));mean=x.mean(0)
        _,_,v=torch.pca_lowrank(x-mean,q=64,center=False,niter=4)
        temp=pp.with_name('fitted_pca.tmp.npz')
        np.savez(temp,mean=mean.numpy(),weight=v.T.numpy(),seed=42,fit_split='train_episodes_0_89')
        temp.replace(pp)
    with np.load(pp) as z:
        weight=torch.from_numpy(z['weight']).cuda();mean=torch.from_numpy(z['mean']).cuda()
    xy=torch.stack(torch.meshgrid(torch.arange(14),torch.arange(14),indexing='ij'),-1).reshape(-1,2)[:,[1,0]].float().cuda()/14+1/28
    dist=torch.cdist(xy,xy); nearby=dist<=.2
    for path in episodes:
        if args.task and path.parent.name not in args.task: continue
        if (path/'world_intervals.npz').exists(): continue
        f=np.load(path/'features.npy',mmap_mode='r');frames=np.load(path/'frames.npy')
        masks=np.load(path/'robot_mask.npy') if args.mask_mode=='verified' else np.zeros((len(f),196),np.float32)
        if masks.shape!=(len(f),196) or not np.isfinite(masks).all(): raise ValueError(path)
        raws=[];valids=[]
        with torch.inference_mode():
            for lo in range(0,len(f)-1,32):
                hi=min(lo+32,len(f)-1)
                batch=torch.tensor(f[lo:hi+1].astype(np.float32),device='cuda')
                raw,valid=match_intervals(batch[:-1],batch[1:],torch.tensor(masks[lo:hi],device='cuda'),weight,xy,nearby,mean)
                raws.append(raw.cpu().numpy());valids.append(valid.cpu().numpy())
        temp=path/f'world_intervals.{os.getpid()}.tmp.npz'
        np.savez_compressed(temp,start=frames[:-1],end=frames[1:],raw=np.concatenate(raws),valid=np.concatenate(valids))
        temp.replace(path/'world_intervals.npz')
        print(json.dumps({'event':'world_prepared','episode':str(path),'intervals':len(f)-1}),flush=True)
    if not args.task:
        atomic_json(root/'world_complete.json',{'complete':True,'mask_mode':args.mask_mode,'missing_masks':missing,'episodes':len(episodes),'pca':'train_only','descriptor_dim':136,'world_decoder_queries':'xy_t0_t1_only'})

if __name__=='__main__': main()
