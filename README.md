# AFCE — Action–Future Coupled Effect Learning

Open-source training and evaluation code for **Beyond Actions: Learning Future Operation Targets for Vision-Language-Action Models**.

This package contains the main training and evaluation path used in the paper:

**Effect (E) training → joint action decoder export → π0.5 single-decoder + GT-AUX + finger-Δ policy → DexJoCo eval**

Website: [AFCEProject/AFCE-Web](https://github.com/AFCEProject/AFCE-Web)  
Code: [AFCEProject/AFCE](https://github.com/AFCEProject/AFCE)

## What's included

| Path | Role |
| --- | --- |
| `afce/` | prepare / E train / export / π+decoder / serve / eval |
| `effect_codec/` | Shared Effect codec used by the main pipeline |
| `effect_vla/` | DINOv3 evidence helpers and optional Task-Effect pilot |
| `openpi/` | π0.5 training/serving fork with AFCE hooks |
| `dexjoco/` | DexJoCo MuJoCo environments + OpenPI eval client |
| `configs/` | Official `rand_obj` / `rand_full` / multi-task eval YAMLs |
| `experiments/finger_delta/` | Canonical experiment protocol + reference hashes |
| `scripts/pipeline/` | E statistics, train launcher, source verify |

## What's not included

- Datasets, DINOv3 weights, π0.5 checkpoints, Effect caches, rollout videos
- Hardware teleoperation stack
- Internal cluster migration / WIP experiment trees
- Absolute lab machine paths (use environment variables below)

## Layout

```text
AFCE/
├── afce/
├── effect_codec/
├── effect_vla/
├── openpi/
├── dexjoco/
├── configs/
├── experiments/finger_delta/
├── scripts/
├── docs/
├── environment-dexjoco.yaml
└── LICENSE
```

## Install

```bash
# DexJoCo / MuJoCo side
conda env create -f environment-dexjoco.yaml
conda activate dexjoco

# OpenPI / JAX side (follow openpi/README.md)
cd openpi && bash install.bash && conda activate openpi
cd ..
```

Set the import path from the repository root:

```bash
export AFCE_ROOT="$PWD"
export AFCE_OPENPI_ROOT="$PWD/openpi"
export PYTHONPATH="$PWD:$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src:$PWD/dexjoco${PYTHONPATH:+:$PYTHONPATH}"
```

## External assets

Provide these locally (not shipped in git):

| Env var | Meaning |
| --- | --- |
| `AFCE_DATA` | 11-task DexJoCo LeRobot dataset root |
| `AFCE_DINO` | Local DINOv3 weights directory |
| `AFCE_RUNTIME` | Writable runtime root for caches / checkpoints |
| `AFCE_PI05_BASE_PARAMS` | π0.5 base Orbax params (`action_dim=44`) |

Also update `openpi/config.yaml` with your checkpoint and dataset paths.

Public data / model references used by DexJoCo:

- Dataset: [DexJoCo-Datasets-LeRobot](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot)
- Baseline policies: [DexJoCo-Pi05](https://huggingface.co/DexJoCo/DexJoCo-Pi05)

## Main pipeline (summary)

Full commands and original run notes: [`experiments/finger_delta/README.md`](experiments/finger_delta/README.md).

```bash
export AFCE_DATA=/absolute/path/to/dexjoco_lerobot_datasets
export AFCE_DINO=/absolute/path/to/dinov3

python -m afce.prepare \
  --data "$AFCE_DATA" --dino "$AFCE_DINO" --output "$PWD/runtime/data"
python -m afce.build_world \
  --evidence "$PWD/runtime/data" --mask-mode legacy-none
python scripts/pipeline/prepare_statistics.py \
  --data "$AFCE_DATA" --evidence "$PWD/runtime/data" \
  --output "$PWD/runtime/calibrated_statistics.pt"

python -m afce.package_sweep_inputs \
  --repo "$PWD" --data "$AFCE_DATA" --output "$PWD/runtime/pipeline-inputs.tar"
mkdir -p runtime/pipeline-inputs && tar -xf runtime/pipeline-inputs.tar -C runtime/pipeline-inputs

CUDA_VISIBLE_DEVICES=0 bash scripts/pipeline/train_e.sh \
  "$PWD/runtime/pipeline-inputs" "$PWD/runtime/effect_query_seed42"

python -m afce.export_effect_resume \
  --checkpoint "$PWD/runtime/effect_query_seed42/last.pt" \
  --data "$AFCE_DATA" --evidence "$PWD/runtime/pipeline-inputs/runtime/data" \
  --output "$PWD/runtime/effect_cache"
python -m afce.export_joint_decoder \
  --effect-cache "$PWD/runtime/effect_cache" \
  --output "$PWD/runtime/joint_decoder_init.npz"
```

Policy train / serve / eval entrypoints:

```bash
python -m afce.multihost_finger_delta_pi --help
python -m afce.serve_finger_delta_pi --help
python -m afce.finger_delta_eval_worker --help
```

Optional multi-node Slurm helper (`afce/finger_delta_worker.sh`) expects `AFCE_ROOT`, `AFCE_RUNTIME`, and optionally `AFCE_PYTHON`.

## Reported seed-0 result

From the frozen reference run (11 tasks × 50 episodes, seed 0): **284 / 550 = 51.64%** overall. See `experiments/finger_delta/README.md` for the per-task table and hash checks:

```bash
python scripts/pipeline/verify_source.py
```

Note: after open-source cleanup some auxiliary modules were removed; regenerate or adjust the source manifest if you need bit-exact verification against the private experiment snapshot.

## License

MIT — see [`LICENSE`](LICENSE). DexJoCo / OpenPI components retain their upstream attributions where applicable.
