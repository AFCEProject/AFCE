#!/usr/bin/env bash
# source after: conda activate dexjoco
ENV_PREFIX="${CONDA_PREFIX:-}"
if [[ -z "$ENV_PREFIX" || ! -d "$ENV_PREFIX/lib/osmesa-deps" ]]; then
  echo "use_osmesa.bash: activate dexjoco first (missing osmesa-deps)" >&2
  return 1 2>/dev/null || exit 1
fi
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/osmesa-deps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "MUJOCO_GL=osmesa"
