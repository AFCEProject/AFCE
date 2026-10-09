"""Export every actual frame's frozen 30x256 target without index substitution."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from afce_all11.codec import Codec
from afce_all11.lowdim import LowDimData
from afce_all11.prepare import atomic_json

def load_codec(path,device='cpu'):
    blob=torch.load(path,map_location='cpu',weights_only=False)
    c=blob['config']
    if c.get('codec_version')=='tuned_v3':
        from afce_all11.tuned_codec import TunedCodec
        m=TunedCodec(c['fusion'],c['condition'],blob['stats'],c['aligned_residual'])
    elif c.get('codec_version')=='calibrated_v2':
        from afce_all11.calibrated_codec import CalibratedCodec
        m=CalibratedCodec(c['fusion'],c['condition'],blob['stats'])
    else:
        m=Codec(c['fusion'],c['condition'])
    m.load_state_dict(blob['model'],strict=True)
    m.eval().to(device)
    for p in m.parameters(): p.requires_grad_(False)
    return m,blob

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence',type=Path);p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args(); torch.set_num_threads(4)
    model,blob=load_codec(args.checkpoint,'cuda:0');args.output.mkdir(parents=True,exist_ok=True)
    if blob['config']['action_only']: data=LowDimData(args.data,'all')
    else:
        from afce_all11.evidence import EvidenceData
        data=EvidenceData(args.data,args.evidence,'all',blob['config'].get('mask_mode','verified'))
    source_sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    record={'complete':False,'total_frames':len(data),'checkpoint_sha256':source_sha,'codec_step':blob['step'],'shape':[30,256],'mask_mode':blob['config'].get('mask_mode'),'episodes':[]}
    atomic_json(args.output/'manifest.json',record)
    offset=0; total=0; sum1=np.zeros(256,np.float64);sum2=np.zeros(256,np.float64)
    for ep in data.episodes:
        n=len(ep['actions']);dest=args.output/ep['task']/f"episode_{ep['episode']:06d}.npy";dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix('.tmp.npy');arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float16,shape=(n,30,256))
        with torch.inference_mode():
            for start in range(0,n,args.batch_size):
                rows=[data.sample(i+offset) for i in range(start,min(start+args.batch_size,n))]
                batch={k:torch.tensor(np.stack([r[k] for r in rows]),device='cuda') for k in rows[0]}
                value=model.encode(batch).float().cpu().numpy()
                if not np.isfinite(value).all(): raise ValueError('Nonfinite E')
                arr[start:start+len(rows)]=value.astype(np.float16)
                if ep['episode']<90:
                    # Normalize the actual quantized cache read by π training.
                    x=arr[start:start+len(rows)].astype(np.float64).reshape(-1,256)
                    sum1+=x.sum(0);sum2+=(x*x).sum(0);total+=len(x)
        arr.flush();del arr;tmp.replace(dest);offset+=n
        record['episodes'].append({'task':ep['task'],'episode':ep['episode'],'frames':n})
        atomic_json(args.output/'manifest.json',record)
        print(f'exported {ep["task"]}/{ep["episode"]}: {n} frame-exact targets',flush=True)
    mean=sum1/total;std=np.sqrt(np.maximum(sum2/total-mean*mean,0)).clip(1e-4)
    np.savez(args.output/'normalization.npz',mean=mean.astype(np.float32),std=std.astype(np.float32),training_tokens=total)
    # Copy the exact decoder source for portable and auditable inference.
    import shutil
    shutil.copy2(args.checkpoint,args.output/'codec.pt')
    record['complete']=offset==523763 and len(record['episodes'])==1100
    atomic_json(args.output/'manifest.json',record)

if __name__=='__main__':main()
