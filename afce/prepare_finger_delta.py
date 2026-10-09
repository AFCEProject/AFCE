"""Read-only finger-Δ asset validation and immutable ablation protocol."""
import hashlib
import json
import os
from pathlib import Path

RUNTIME=Path(os.environ.get('AFCE_RUNTIME', str(Path.cwd()/'runtime'))).resolve()
CODE=Path(os.environ.get('AFCE_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
QUERY=Path(os.environ.get('AFCE_QUERY_ROOT', str(RUNTIME/'query_effect'))).resolve()
ROOT=Path(os.environ.get('AFCE_EXPERIMENT_ROOT', str(RUNTIME/'query_finger_delta'))).resolve()

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 manifest=json.loads((QUERY/'effect_cache/manifest.json').read_text())
 receipt=json.loads((QUERY/'joint_decoder_init.json').read_text())
 parity=json.loads((QUERY/'decoder_parity.json').read_text())
 calibration=json.loads((RUNTIME/'query_effect_scratch3_20260920/calibration.json').read_text())
 assert manifest['complete'] is True
 assert receipt['codec_sha256']==manifest['checkpoint_sha256']
 assert parity['passed'] is True
 assert calibration['chosen_weights']['ground_truth']==0.25628781345139295
 assert (QUERY/'pi/assets').is_dir()
 protocol=dict(schema='finger_delta_joint_fixed',effect_checkpoint_sha256=manifest['checkpoint_sha256'],
  effect_cache=str(QUERY/'effect_cache'),decoder_init_sha256=sha(QUERY/'joint_decoder_init.npz'),
  effect_normalization_sha256=sha(QUERY/'effect_cache/normalization.npz'),
  effect_manifest_sha256=sha(QUERY/'effect_cache/manifest.json'),
  initialization='official pi0.5 base plus original finger-Δ decoder',seed=42,batch_size=32,target_updates=60000,
  nodes=2,gpus=8,decoder_warmup_steps=0,decoder_update_multiplier=.25,
  gtaux_weight=0.25628781345139295,finger_delta_weight=.05,finger_delta_scale_floor=.1,
  auxiliary_ramp_updates=10000,auxiliary_time_gate=[.2,.7],
  objective='FM(E) + ramp * (0.25628781345139295 GT-AUX + 0.05 finger-delta)',
  decoder_gradient='GT-AUX and finger-delta jointly update pi and the same action decoder',
  removed=['independent frozen D0','independent adaptive D1','D1 predicted-E adaptation','D1 true-E anchor','D1 full-rollout adaptation'],
  comparison_scope='decoder training scheme; not decoder count alone',
  evaluation=dict(seed=0,tasks=11,episodes_per_task=50,total=550,nodes=2,gpus=8,num_denoise_steps=10,
                  continuation='committed episode and environment/server RNG offset',scheduling='eight active per-task file locks; three queued tasks picked by first free GPU'))
 p=ROOT/'run_protocol.json';p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert json.loads(p.read_text())==protocol,'Existing run protocol changed'
 else:
  q=p.with_suffix('.tmp');q.write_text(json.dumps(protocol,indent=2)+'\n');q.replace(p)
 print(json.dumps(dict(passed=True,protocol=str(p),effect_checkpoint_sha256=manifest['checkpoint_sha256'])))

if __name__=='__main__':main()
