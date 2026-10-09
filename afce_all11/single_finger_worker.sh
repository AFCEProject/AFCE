#!/bin/bash
set -Eeuo pipefail
trap '' USR1
NODES=${1:?two-node list}; DEADLINE=${2:?absolute save deadline}
[[ ${SLURM_JOB_PARTITION:?} == debug ]]
# Required env: AFCE_ROOT, AFCE_RUNTIME, AFCE_PYTHON (optional overrides below).
CODE=${AFCE_ROOT:?set AFCE_ROOT to this repository}
RUNTIME=${AFCE_RUNTIME:?set AFCE_RUNTIME to your runtime directory}
QUERY=${AFCE_QUERY_ROOT:-$RUNTIME/query_c01}
EXPERIMENT=${AFCE_EXPERIMENT_ROOT:-$RUNTIME/query_c01_single_finger}
EVAL=${AFCE_EVAL_ROOT:-$RUNTIME/evaluations/c01_single_finger_seed0}
PYTHON=${AFCE_PYTHON:-python}
NAME=c01_single_finger_seed42_60000
ROOT="$EXPERIMENT/pi/checkpoints/afce_all11_official/$NAME"
GATE="$EXPERIMENT/gates/resume_v2"
export PYTHONPATH="$CODE:$CODE/openpi/src:$CODE/openpi/packages/openpi-client/src:$CODE/dexjoco"
export AFCE_ROOT="$CODE" AFCE_RUNTIME="$RUNTIME" AFCE_QUERY_ROOT="$QUERY" AFCE_EXPERIMENT_ROOT="$EXPERIMENT" AFCE_EVAL_ROOT="$EVAL"
export AFCE_OPENPI_ROOT="$CODE/openpi" PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1
ulimit -n 16384
mapfile -t HOSTS < <(scontrol show hostnames "$NODES")
[[ ${#HOSTS[@]} == 2 ]]
mkdir -p "$EXPERIMENT/gates" "$EXPERIMENT/pi" "$GATE" "$EVAL"
cd "$CODE/openpi"
"$PYTHON" -m afce_all11.prepare_single_finger
available() { (( $(date +%s)+$1 < DEADLINE )); }
latest_update() {
 "$PYTHON" -c 'import json,sys;from pathlib import Path;p=Path(sys.argv[1])/"latest_complete.json"; print(json.loads(p.read_text())["optimizer_updates"] if p.is_file() else 0)' "$1"
}
passed() {
 "$PYTHON" -c 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.exit(0 if p.is_file() and json.loads(p.read_text()).get(sys.argv[2]) is True else 1)' "$1" "$2"
}
phase() {
 "$PYTHON" - "$EXPERIMENT/pipeline_status.json" "$1" "$NODES" "$DEADLINE" <<'PY'
import json,sys,os,time
from pathlib import Path
p=Path(sys.argv[1]);q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(phase=sys.argv[2],complete=False,nodes=sys.argv[3],deadline_unix=int(sys.argv[4]),job_id=os.environ['SLURM_JOB_ID'],utc=time.time()),indent=2));q.replace(p)
PY
}
run_distributed() {
 local output=$1 name=$2 stop=$3 trace=$4
 local port=$((22000 + (SLURM_JOB_ID + stop + ${#name} * 17) % 12000))
 local args=(--query-root "$QUERY" --decoder-init "$QUERY/joint_decoder_init.npz" --assets-dir "$QUERY/pi/assets"
             --output "$output" --name "$name" --variant single_finger --weight 0.25628781345139295
             --decoder-warmup-steps 0 --stop-at-update "$stop" --deadline-unix "$DEADLINE" --initialize-from-base)
 [[ $trace == 1 ]] && args+=(--trace)
 env AFCE_COORDINATOR="${HOSTS[0]}:$port" env -u LD_LIBRARY_PATH \
 srun --export="ALL,LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" --nodes=2 --ntasks=2 --ntasks-per-node=1 \
   --nodelist="$NODES" --cpus-per-task=64 --gpus-per-task=4 --exact --exclusive --kill-on-bad-exit=1 --unbuffered \
   "$PYTHON" -u -m afce_all11.multihost_single_finger_pi "${args[@]}"
}
run_audit() {
 env -u LD_LIBRARY_PATH srun --export="ALL,LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
   --nodes=1 --ntasks=1 --nodelist="${HOSTS[0]}" --cpus-per-task=64 --gpus-per-task=1 \
   --exact --exclusive --kill-on-bad-exit=1 --unbuffered env JAX_PLATFORMS=cpu "$PYTHON" -u "$@"
}
while available 150; do
 if ! passed "$EXPERIMENT/gates/resume_gate_v2.json" passed; then
  phase resume_gate
  CONT="$GATE/checkpoints/afce_all11_official/continuous"
  INTR="$GATE/checkpoints/afce_all11_official/interrupted"
  if [[ $(latest_update "$CONT") -lt 4 ]]; then
   available 480 || exit 75
   run_distributed "$GATE" continuous 4 1
   continue
  fi
  if [[ $(latest_update "$INTR") -lt 2 ]]; then
   available 480 || exit 75
   run_distributed "$GATE" interrupted 2 1
   continue
  fi
  if [[ $(latest_update "$INTR") -lt 4 ]]; then
   available 480 || exit 75
   run_distributed "$GATE" interrupted 4 1
   continue
  fi
  available 300 || exit 75
  run_audit -m afce_all11.check_single_finger_checkpoint --checkpoint "$CONT/3" \
    --decoder-init "$QUERY/joint_decoder_init.npz" --output "$EXPERIMENT/gates/decoder_updated_v2.json"
  run_audit -m afce_all11.check_single_finger_gate --root "$GATE" \
    --decoder-audit "$EXPERIMENT/gates/decoder_updated_v2.json" --output "$EXPERIMENT/gates/resume_gate_v2.json"
  continue
 fi
 if ! passed "$ROOT/complete.json" complete; then
  phase training
  available 420 || exit 75
  run_distributed "$EXPERIMENT/pi" "$NAME" 60000 0
  passed "$ROOT/complete.json" complete || exit 75
  continue
 fi
 if ! passed "$EXPERIMENT/final_decoder_audit.json" passed; then
  available 150 || exit 75
  phase final_checkpoint_audit
  run_audit -m afce_all11.check_single_finger_checkpoint --checkpoint "$ROOT/59999" \
    --decoder-init "$QUERY/joint_decoder_init.npz" --output "$EXPERIMENT/final_decoder_audit.json"
  continue
 fi
 if ! passed "$EVAL/complete.json" complete; then
  phase eval_seed0
  "$PYTHON" -m afce_all11.single_finger_eval_state --mode prepare --training-root "$ROOT" \
    --effect-cache "$QUERY/effect_cache" --decoder-init "$QUERY/joint_decoder_init.npz" --eval-root "$EVAL"
  available 180 || exit 75
  env -u LD_LIBRARY_PATH srun --export="ALL,LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
    --nodes=2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --gpus-per-task=1 \
    --exact --exclusive --nodelist="$NODES" --kill-on-bad-exit=1 --unbuffered \
    "$PYTHON" -u -m afce_all11.single_finger_eval_worker --deadline-unix "$DEADLINE"
  "$PYTHON" -m afce_all11.single_finger_eval_state --mode summary --eval-root "$EVAL"
  passed "$EVAL/complete.json" complete || exit 75
  continue
 fi
 phase complete
 "$PYTHON" - "$EXPERIMENT/pipeline_status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads(p.read_text());d['complete']=True;q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
PY
 echo C01_SINGLE_FINGER_PIPELINE_COMPLETE
 exit 0
done
exit 75
