"""Train codecs with per-frame sampling, batch 32, and episode-disjoint validation."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from afce_all11.codec import Codec,losses
from afce_all11.lowdim import LowDimData,grouped_batch
from afce_all11.prepare import atomic_json

def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)

@torch.inference_mode()
def validate(model,data,stats,device,indices,action_only):
    model.eval(); sums={}; count=0
    for lo in range(0,len(indices),32):
        for d,b in grouped_batch(data,indices[lo:lo+32],device).items():
            out=model(b); metrics=losses(out,b,stats,action_only=action_only)
            n=len(b['action']); count+=n
            for key,v in metrics.items(): sums[key]=sums.get(key,0.)+float(v)*n
            sums[f'action_mae_{d}']=sums.get(f'action_mae_{d}',0.)+float(metrics['action_mae'])*n
            sums[f'count_{d}']=sums.get(f'count_{d}',0)+n
    result={k:v/count for k,v in sums.items() if not k.startswith(('count_','action_mae_'))}
    for d in (22,44):
        if sums.get(f'count_{d}',0): result[f'action_mae_{d}']=sums[f'action_mae_{d}']/sums[f'count_{d}']
    model.train(); return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fusion',choices=['reference','query','poe','concat'],default='reference')
    p.add_argument('--condition',choices=['A','ASV'],default='ASV')
    p.add_argument('--action-only',action='store_true')
    p.add_argument('--evidence',type=Path)
    p.add_argument('--mask-mode',choices=['verified','legacy-none'],default='verified')
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--eval-windows',type=int,default=512)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.action_only and args.condition!='A': raise ValueError('C0 must use A-only input')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed)
    device='cuda:0'; args.output.mkdir(parents=True,exist_ok=True)
    run_lock=(args.output/'training.lock').open('a')
    fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    config.update(effect_shape=[30,256],world_descriptor_dim=136,independent_action_modules=[22,44])
    cp=args.output/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config: raise ValueError('Configuration differs from existing run')
    if (args.output/'last.pt').exists() and not args.resume: raise ValueError('Existing run: explicitly resume or choose another output')
    atomic_json(cp,config)
    print(json.dumps({'event':'loading_training_data','fusion':args.fusion,'condition':args.condition,'batch_size':args.batch_size,'steps':args.steps}),flush=True)
    if args.action_only:
        train,val=LowDimData(args.data,'train'),LowDimData(args.data,'val')
    else:
        from afce_all11.evidence import EvidenceData
        train,val=EvidenceData(args.data,args.evidence,'train',args.mask_mode),EvidenceData(args.data,args.evidence,'val',args.mask_mode)
    print(json.dumps({'event':'computing_train_only_statistics','train_frames':len(train),'val_frames':len(val)}),flush=True)
    stats=train.action_stats()
    print(json.dumps({'event':'statistics_ready'}),flush=True)
    model=Codec(args.fusion,args.condition).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    rng=np.random.default_rng(args.seed); step=0; best=float('inf')
    if args.resume:
        ckpt=torch.load(args.output/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        step=ckpt['step']; best=ckpt['best']; rng.bit_generator.state=ckpt['numpy_rng']; torch.set_rng_state(ckpt['torch_rng'].cpu())
        torch.cuda.set_rng_state(ckpt['cuda_rng'].cpu()); stats=ckpt['stats']
    vi=np.random.default_rng(103).integers(0,len(val),args.eval_windows)
    started=time.monotonic(); model.train()
    print(json.dumps({'event':'train_started','train_frames':len(train),'val_frames':len(val),'batch_size':args.batch_size,'parameters':sum(x.numel() for x in model.parameters()),'fusion':args.fusion,'action_only':args.action_only}),flush=True)
    for step in range(step+1,args.steps+1):
        optimizer.zero_grad(set_to_none=True); metrics={}
        indices=rng.integers(0,len(train),args.batch_size)
        lam=.05+.15*min(step/1000,1)
        for d,b in grouped_batch(train,indices,device).items():
            out=model(b); values=losses(out,b,stats,lam,lam,action_only=args.action_only)
            weight=len(b['action'])/args.batch_size
            if not torch.isfinite(values['loss']): raise FloatingPointError(f'Nonfinite loss at {step}')
            (values['loss']*weight).backward()
            for key,v in values.items(): metrics[key]=metrics.get(key,0.)+float(v.detach())*weight
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad): raise FloatingPointError(f'Nonfinite gradient at {step}')
        optimizer.step()
        if step==1 or step%50==0:
            row={'event':'train','step':step,**metrics,'grad_norm':float(grad),'elapsed_s':time.monotonic()-started}
            with (args.output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            score=validate(model,val,stats,device,vi,args.action_only)
            improved=score['loss']<best; best=min(best,score['loss'])
            blob={'schema':'afce_all11_30x256_v1','model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'best':best,'val':score,'stats':stats,'config':config,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
            atomic_save(blob,args.output/'last.pt')
            if improved: atomic_save(blob,args.output/'best.pt')
            print(json.dumps({'event':'validation','step':step,**score}),flush=True)
    atomic_json(args.output/'complete.json',{'steps':step,'best_validation_loss':best,'train_frames':len(train),'val_frames':len(val),'status':'codec_training_completed_not_rollout_validated'})

if __name__=='__main__': main()
