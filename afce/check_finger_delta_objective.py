"""Check unchanged auxiliary values/E gradients and new decoder gradients before GPU training."""
import argparse
from pathlib import Path
import json
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from afce.jax_action_decoder import ActionDecoder,load_arrays
from afce.readout_alignment import ReadoutEffectPi,time_weight
from afce.dual_decoder_alignment import DualDecoderEffectPi
from afce.finger_delta_alignment import SingleFingerEffectPi
from afce.prepare import atomic_json
from types import SimpleNamespace

class Head(nnx.Module):
    auxiliary=SingleFingerEffectPi.auxiliary
    finger_delta_loss=SingleFingerEffectPi.finger_delta_loss
    def __init__(self,arrays):
        self.single_decoder=ActionDecoder(arrays,22)
        self.bimanual_decoder=ActionDecoder(arrays,44)
        self.effect_mean=nnx.Variable(jnp.asarray(arrays['effect_mean']))
        self.effect_std=nnx.Variable(jnp.asarray(arrays['effect_std']))
        self.finger_delta_scale_floor=.1

def main():
    p=argparse.ArgumentParser();p.add_argument('--decoder-init',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    arrays=load_arrays(args.decoder_init);m=Head(arrays);rng=np.random.default_rng(42)
    clean=jnp.asarray(rng.normal(size=(2,30,256)).astype('float32'))
    state=np.zeros((2,46),dtype='float32');acts=np.zeros((2,30,44),dtype='float32')
    for i,dim in enumerate((22,44)):
        state[i,:23*(dim//22)]=arrays[f'state_mean_{dim}']
        acts[i,:,:dim]=rng.normal(size=(30,dim)).astype('float32')*arrays[f'action_std_{dim}']+arrays[f'action_mean_{dim}']
    valid=np.ones((2,30),dtype=bool);valid[0,25:]=False
    ob=SimpleNamespace(effect_state=jnp.asarray(state),effect_actions=jnp.asarray(acts),effect_valid=jnp.asarray(valid),effect_action_dim=jnp.asarray([22,44]))
    weights=time_weight(jnp.asarray([.3,.5]));report={'passed':False,'components':{}}
    for name,old,new in (
        ('gtaux',lambda mod,z:ReadoutEffectPi.auxiliary(mod,z,z,ob,weights,'ground_truth')[0],lambda mod,z:mod.auxiliary(z,z,ob,weights,'ground_truth')[0]),
        ('finger_delta',lambda mod,z:DualDecoderEffectPi.finger_delta_loss(mod,z,ob,weights)[0],lambda mod,z:mod.finger_delta_loss(z,ob,weights)[0])):
        a,ga=jax.value_and_grad(lambda z:old(m,z))(clean);b,gb=jax.value_and_grad(lambda z:new(m,z))(clean)
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b));np.testing.assert_array_equal(np.asarray(ga),np.asarray(gb))
        oldgrads=nnx.grad(old)(m,clean);newgrads=nnx.grad(new)(m,clean)
        oldnorm=float(optax.global_norm(oldgrads));norms={arm:float(optax.global_norm(newgrads[arm+'_decoder'])) for arm in ('single','bimanual')}
        assert oldnorm==0 and all(v>0 and np.isfinite(v) for v in norms.values())
        assert float(jnp.linalg.norm(gb))>0
        report['components'][name]={'value':float(b),'forward_and_E_gradient_bitwise_equal':True,'old_decoder_grad_norm':oldnorm,'new_decoder_grad_norm':norms,'E_gradient_norm':float(jnp.linalg.norm(gb))}
    report.update(passed=True,normalization_and_finger_formula_unchanged=True,decoder_joint_gradient_enabled=True)
    atomic_json(args.output,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
