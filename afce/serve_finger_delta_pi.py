"""Serve a raw-action or frozen-codec π0.5 policy with identical input processing."""
import argparse
import dataclasses
import os
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--task',required=True)
    p.add_argument('--effect-cache',type=Path);p.add_argument('--port',type=int,default=18000)
    p.add_argument('--joint-decoder-init',type=Path)
    p.add_argument('--single-finger',action='store_true')
    p.add_argument('--dual-decoder',action='store_true',
                   help='Build the dual-decoder checkpoint and serve adaptive D1')
    p.add_argument('--rng-offset',type=int,default=0,
                   help='Advance the policy sampling key by this many completed inference requests')
    args=p.parse_args()
    if args.single_finger and (args.dual_decoder or not args.joint_decoder_init):
        raise ValueError("single-finger requires its own jointly trained decoder")
    for key in ('checkpoint','data','base','output','effect_cache','joint_decoder_init'):
        if getattr(args,key) is not None: setattr(args,key,getattr(args,key).resolve())
    repo=Path(__file__).resolve().parents[1]
    openpi_root=Path(os.environ.get('AFCE_OPENPI_ROOT',repo/'openpi')).resolve()
    if not (openpi_root/'config.yaml').is_file() or not (openpi_root/'scripts').is_dir():
        raise FileNotFoundError(f'Invalid AFCE_OPENPI_ROOT: {openpi_root}')
    sys.path.insert(0,str(openpi_root/'scripts'));os.chdir(openpi_root)
    import dexjoco_multitask as baseline
    from serve_dexjoco_checkpoint import _load_jax_model,_load_policy
    from openpi.policies.policy import Policy
    from openpi.training import checkpoints
    from openpi import transforms
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    import numpy as np
    cfg,_,_=baseline.build_config(baseline.Args(operation='inspect',data_root=args.data,init_params_path=args.base,
        exp_name=args.name,checkpoint_base_dir=args.output/'checkpoints',assets_base_dir=args.output/'assets'))
    cfg=dataclasses.replace(cfg,name='afce_official',data=dataclasses.replace(cfg.data,balance='proportional'))
    if args.effect_cache:
        from afce.pi_bridge import effect_config,DecodedEffectPolicy
        from afce.export_effect import load_codec
        if args.joint_decoder_init:
            from afce.pi_bridge_joint import joint_effect_config,JointDecodedEffectPolicy
            cfg=joint_effect_config(cfg,args.effect_cache,args.base,args.joint_decoder_init,
                                    decoder_warmup_steps=60001 if args.dual_decoder else 0)
            if args.single_finger:
                from afce.finger_delta_alignment import finger_delta_config
                cfg=finger_delta_config(cfg)
            if args.dual_decoder:
                from afce.readout_alignment import readout_config
                from afce.dual_decoder_alignment import dual_decoder_config,DualDecodedEffectPolicy
                cfg=readout_config(cfg,'gtaux',alignment_start=0,alignment_ramp_steps=10000)
                cfg=dual_decoder_config(cfg)
        else:
            cfg=effect_config(cfg,args.effect_cache,args.base)
        model=_load_jax_model(cfg,args.checkpoint);data=cfg.data.create(cfg.assets_dirs,cfg.model)
        norm=checkpoints.load_norm_stats(args.checkpoint/'assets',data.asset_id)
        latent=Policy(model,transforms=[transforms.InjectDefaultPrompt(None),*data.data_transforms.inputs,
            transforms.Normalize(norm,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs],
            output_transforms=[],metadata={'method':'afce30x256','task':args.task})
        if args.rng_offset < 0:
            raise ValueError('rng-offset must be nonnegative')
        if args.rng_offset:
            import jax
            for _ in range(args.rng_offset):
                latent._rng, _ = jax.random.split(latent._rng)
            print(f'POLICY_RNG_OFFSET requests={args.rng_offset}', flush=True)
        # E uses its own per-channel standardization; never apply action stats
        # or DualArmOutputs (which would truncate E to 44 channels).
        if args.joint_decoder_init:
            policy=(DualDecodedEffectPolicy(latent,model,task=args.task) if args.dual_decoder
                    else JointDecodedEffectPolicy(latent,model,task=args.task))
        else:
            codec,_=load_codec(args.effect_cache/'codec.pt')
            with np.load(args.effect_cache/'normalization.npz') as z:
                policy=DecodedEffectPolicy(latent,codec,z['mean'],z['std'],task=args.task)
    else: policy=_load_policy(cfg,args.checkpoint)
    WebsocketPolicyServer(policy=policy,host='0.0.0.0',port=args.port,metadata=policy.metadata).serve_forever()

if __name__=='__main__':main()
