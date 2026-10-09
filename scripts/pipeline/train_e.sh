#!/usr/bin/env bash
# Effect (E) training launcher for the public AFCE recipe (30k updates).
set -euo pipefail
if [[ $# -lt 2 || ${1:-} == --help ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/pipeline/train_e.sh INPUT_BUNDLE OUTPUT [--resume]"
  exit 0
fi
bundle=$1
output=$2
shift 2
exec "${AFCE_PYTHON:-python}" -u -m afce.train_methods \
  --data "$bundle/datasets" \
  --evidence "$bundle/runtime/data" \
  --statistics "$bundle/runtime/calibrated_statistics.pt" \
  --output "$output" --fusion query --steps 30000 --batch-size 32 --seed 42 \
  --lr 1e-4 --min-lr 3e-5 --schedule two_stage --switch-step 20000 \
  --warmup 0 --weight-decay .01 --beta2 .99 --action-weight 1 \
  --position-weight 2 --rotation-weight .05 --lambda-s .2 --lambda-v .2 \
  --eval-every 1000 "$@"
