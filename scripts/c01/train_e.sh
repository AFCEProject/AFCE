#!/usr/bin/env bash
# C01 Query E, matching the selected 30k checkpoint's recorded recipe.
set -euo pipefail
if [[ $# -lt 2 || ${1:-} == --help ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/c01/train_e.sh INPUT_BUNDLE OUTPUT [--resume]"
  exit 0
fi
c01_bundle=$1
c01_output=$2
shift 2
exec "${C01_PYTHON:-python}" -u -m afce_all11.train_methods \
  --data "$c01_bundle/datasets" \
  --evidence "$c01_bundle/runtime/data" \
  --statistics "$c01_bundle/runtime/calibrated_statistics.pt" \
  --output "$c01_output" --fusion query --steps 30000 --batch-size 32 --seed 42 \
  --lr 1e-4 --min-lr 3e-5 --schedule two_stage --switch-step 20000 \
  --warmup 0 --weight-decay .01 --beta2 .99 --action-weight 1 \
  --position-weight 2 --rotation-weight .05 --lambda-s .2 --lambda-v .2 \
  --eval-every 1000 "$@"
