"""30x256 codecs with separate complete single/bimanual action modules."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from effect_codec.model.codec import ActionBranch, ActionDecoder, TriModalMixer, TemporalEffect


class RobotBranch(nn.Module):
    def __init__(self, hands):
        super().__init__()
        self.root = nn.Sequential(nn.Linear(9*hands,64),nn.GELU(),nn.Linear(64,64))
        self.tips = nn.Sequential(nn.Linear(12*hands,64),nn.GELU(),nn.Linear(64,64))
        self.joints = nn.Sequential(nn.Linear(16*hands,64),nn.GELU(),nn.Linear(64,64))
        self.merge = nn.Linear(192,64)

    def forward(self, root, tips, joints):
        return self.merge(torch.cat([self.root(root),self.tips(tips.flatten(-2)),self.joints(joints)],-1))


class SparseWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(136,128),nn.GELU(),nn.Linear(128,64))
        self.query = nn.Parameter(torch.randn(30,64)*.02)
        self.time = nn.Embedding(30,64)
        self.attn = nn.MultiheadAttention(64,4,batch_first=True)
        self.null = nn.Parameter(torch.zeros(1,1,64))

    def forward(self,w,valid):
        b = w.shape[0]
        tokens = self.proj(w.flatten(1,2))
        mask = ~valid.flatten(1,2).bool()
        # A null token is accessible only for a genuinely empty observation.
        empty = mask.all(-1,keepdim=True)
        tokens = torch.cat([tokens,self.null.expand(b,1,-1)],1)
        mask = torch.cat([mask,~empty],1)
        q = (self.query+self.time.weight).unsqueeze(0).expand(b,-1,-1)
        return self.attn(q,tokens,tokens,key_padding_mask=mask,need_weights=False)[0]


class QueryFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(64,256)
        self.types = nn.Parameter(torch.randn(3,256)*.02)
        self.query = nn.Parameter(torch.randn(30,256)*.02)
        layer = nn.TransformerDecoderLayer(256,8,1024,dropout=0.,batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(layer,2)

    def forward(self,a,s,v):
        b,h,_ = a.shape
        memory = self.project(torch.stack([a,s,v],2))+self.types
        q = self.query[None].expand(b,-1,-1).reshape(b*h,1,256)
        return self.decoder(q,memory.reshape(b*h,3,256)).reshape(b,h,256)


class PoEFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(64,256),nn.GELU(),nn.Linear(256,512)) for _ in range(3)])
        self.last_contributions = None

    def forward(self,a,s,v):
        mu,prec = [],[]
        for net,x in zip(self.experts,(a,s,v),strict=True):
            mean,raw = net(x).float().chunk(2,-1)
            mu.append(mean)
            # Bounded log precision avoids overflow and arbitrary scale drift.
            prec.append(torch.exp(4*torch.tanh(raw/4)))
        mu,prec = torch.stack(mu,0),torch.stack(prec,0)
        weights = prec/prec.sum(0,keepdim=True)
        self.last_contributions = weights.detach().mean((1,2,3))
        return (weights*mu).sum(0).to(a.dtype)


class WorldDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4,256)
        self.attn = nn.MultiheadAttention(256,8,batch_first=True)
        self.out = nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,66))

    def forward(self,e,queries):
        b,k,m,_ = queries.shape
        q = self.q(queries.reshape(b,k*m,4))
        y = self.attn(q,e,e,need_weights=False)[0]
        return self.out(y).reshape(b,k,m,66)


class Codec(nn.Module):
    def __init__(self, fusion='reference', conditioning='ASV'):
        super().__init__()
        if fusion not in ('reference','query','poe','concat'):
            raise ValueError(fusion)
        if conditioning not in ('ASV','A'):
            raise ValueError(conditioning)
        self.fusion_kind,self.conditioning = fusion,conditioning
        self.action_encoders = nn.ModuleDict({str(d):ActionBranch(action_dim=d) for d in (22,44)})
        self.robot_encoders = nn.ModuleDict({str(d):RobotBranch(d//22) for d in (22,44)})
        self.action_decoders = nn.ModuleDict({str(d):ActionDecoder(action_dim=d,state_dim=23*(d//22)) for d in (22,44)})
        self.response_decoders = nn.ModuleDict({str(d):nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,37*(d//22))) for d in (22,44)})
        self.world = SparseWorld()
        self.world_decoder = WorldDecoder()
        self.temporal = TemporalEffect()
        if fusion == 'reference':
            self.mixer = TriModalMixer()
            self.project = nn.Linear(192,256)
        elif fusion == 'concat':
            self.project = nn.Sequential(nn.Linear(192,256),nn.GELU(),nn.Linear(256,256))
        elif fusion == 'query':
            self.fuser = QueryFusion()
        else:
            self.fuser = PoEFusion()

    def encode(self,batch,drop=None):
        d = str(batch['action'].shape[-1])
        a = self.action_encoders[d](batch['action'])
        if self.conditioning == 'A':
            s,v = torch.zeros_like(a),torch.zeros_like(a)
        else:
            s = self.robot_encoders[d](batch['root'],batch['tips'],batch['joints'])
            v = self.world(batch['world'],batch['world_valid'])
        a = torch.zeros_like(a) if drop == 'action' else a
        s = torch.zeros_like(s) if drop == 'state' else s
        v = torch.zeros_like(v) if drop == 'world' else v
        if self.fusion_kind == 'reference':
            x = self.project(torch.cat(self.mixer(a,s,v),-1))
        elif self.fusion_kind == 'concat':
            x = self.project(torch.cat([a,s,v],-1))
        else:
            x = self.fuser(a,s,v)
        return self.temporal(x)

    def decode(self,e,state):
        d = str(22 if state.shape[-1] == 23 else 44)
        return self.action_decoders[d](e,state)

    def forward(self,batch,drop=None):
        e = self.encode(batch,drop)
        d = str(batch['action'].shape[-1])
        return {'E':e,'action':self.decode(e,batch['state']),
                'response':self.response_decoders[d](e),
                'world':self.world_decoder(e,batch['world_query'])}


def masked_huber(pred,target,mask):
    error = F.smooth_l1_loss(pred.float(),target.float(),reduction='none')
    mask = torch.broadcast_to(mask,error.shape).to(error.dtype)
    return (error*mask).sum()/mask.sum().clamp_min(1)


def losses(out,batch,stats,lambda_s=.2,lambda_v=.2,action_only=False):
    d = batch['action'].shape[-1]
    scale = stats[str(d)]['action_std'].to(out['action'].device)
    a = []
    for off in range(0,d,22):
        for lo,hi in ((0,3),(3,6),(6,22)):
            sl = slice(off+lo,off+hi)
            a.append(F.smooth_l1_loss(out['action'][...,sl].float()/scale[sl],batch['action'][...,sl]/scale[sl]))
    la = torch.stack(a).mean()
    sy = batch['response']
    ss = stats[str(d)]['response_std'].to(sy.device)
    groups=[]
    for off in range(0,sy.shape[-1],37):
        for lo,hi in ((0,3),(3,9),(9,21),(21,37)):
            sl=slice(off+lo,off+hi)
            groups.append(masked_huber(out['response'][...,sl]/ss[sl],sy[...,sl]/ss[sl],batch['response_valid'][...,None]))
    ls=torch.stack(groups).mean()
    wp,wt=out['world'],batch['world_target']
    if 'world' in stats:
        world_scale=stats['world']['std'][:66].to(wp.device)
        wp,wt=wp/world_scale,wt/world_scale
    vm=batch['world_valid'][...,None]
    lv=masked_huber(wp[...,:64],wt[...,:64],vm)+masked_huber(wp[...,64:],wt[...,64:],vm)
    total=la if action_only else la+lambda_s*ls+lambda_v*lv
    return {'loss':total,'LA':la,'LS':ls,'LV':lv,'action_mae':(out['action']-batch['action']).abs().mean()}
