# Task-Effect Grounded π0.5（DexJoCo 第一阶段）

在已经跑通的 DexJoCo π0.5 上验证：

```text
Task Effect → Robot Grounding → Action
```

第一阶段唯一目标：`Success(π0.5+Effect+Grounding) > Success(π0.5-Continue)`。

DINOv3 和 future frame **只用于离线构造训练 target**。Inference 删除它们。

## 目录

```text
effect_vla/
├── data/          # LeRobot 窗口、离线 cache、wrist/fingertip grounding
├── effect/        # Frozen DINOv3、correspondence、robot mask、E*
├── model/         # JAX Effect Query / Grounding / zero-init adapter / π0.5 包装
├── loss/          # L_E, L_G
├── configs/       # v1 默认超参
├── eval/          # 沿用现有 dexjoco-openpi-eval
├── scripts/       # cache / 可视化
└── tests/
```

OpenPI 侧只做了最小挂钩（默认全部关闭，原 11-task 训练不受影响）：

- `Pi0Config.effect_grounding`
- Observation 可选 `effect_target` / `grounding_target`
- 新 config 名：`<task>_continue`、`<task>_effect_v1_bootstrap`、`<task>_effect_v1`

## 默认超参（v1）

| 项 | 值 |
|---|---|
| Anchors K | 2（0.5H, H），H=30 |
| Effect dim | 512 |
| Geometry | wrist Δp+6D + 4 Allegro fingertips / hand |
| Loss | `L = L_FM + 0.2 L_E + 0.1 L_G` |
| Adapter | zero-init，初始行为 ≈ 原 π0.5 |
| Effect 相机 | 第三人称 front/ego；wrist RGB 仍进 policy |
| Bootstrap | 3K（冻 VLM LoRA） |
| Joint FT / Continue | 13K extra steps |

Allegro 实际是 4 指。方案里的「5 fingertips」按 DexJoCo 指尖顺序落地为食指/中指/无名指/拇指。

## 实现顺序

### 1. 冻结 baseline

记录现有 11-task success。把已训好的 π0.5 params 路径写进 `openpi/config.yaml`：

```yaml
effect_init_checkpoint_root: "../checkpoints/pi05_ckpts"
# 或显式：
# effect_init_checkpoints:
#   water_plant: "../checkpoints/pi05_ckpts/water_plant/<exp>/<step>/params"
```

### 2–3. 离线 cache + 可视化

在 **openpi conda 环境**（有 LeRobot / transformers）里：

```bash
cd "$AFCE_ROOT"   # repository root
export PYTHONPATH="$PWD:$PWD/openpi/src:$PYTHONPATH"

# 先 3-task pilot
python -m effect_vla.scripts.cache_effect_targets \
  --task water_plant --task pick_bucket --task hammer_nail \
  --device cuda --skip-existing

python -m effect_vla.scripts.visualize_effect --task water_plant --n 50
```

确认高权重区域落在物体 / 接触区，而不是整只机械臂。

LeRobot 导出不含完整 MuJoCo qpos 时，robot mask 默认为 0（不抑制）。若之后用仿真 segmentation 写出 `robot_mask.npz`，cache 会自动读取。

### 4–6. 训练

```bash
cd openpi
conda activate openpi
export PYTHONPATH="..:$PYTHONPATH"

# Stage 1 bootstrap（让 Ê / G / adapter 活起来）
python scripts/train.py water_plant_effect_v1_bootstrap

# Stage 2 joint FT（同一 config 名再跑 13K，或直接用 water_plant_effect_v1 从同一 ckpt 起）
python scripts/train.py water_plant_effect_v1

# 必须同时跑的 control
python scripts/train.py water_plant_continue
```

Pilot 三任务：`water_plant`、`pick_bucket`、`hammer_nail`。Go 标准：平均 +≥3 points，或 2/3 明显提升且第三个不崩。

### 7. Rollout

评测协议与现有 baseline **完全相同**（同一 yaml、seed、episode 数）：

```bash
# openpi 环境
python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config water_plant_effect_v1 \
  --policy.dir ../checkpoints/pi05_ckpts/water_plant_effect_v1/<exp>/<step>

# dexjoco 环境
conda activate dexjoco
python -m effect_vla.eval.dexjoco_rollout --task water_plant --port 8000 --run
```

### 8. Ablation（仅 Full 涨点之后）

Pilot 任务额外注册了：

- `<task>_effect_only`：去掉 G
- `<task>_no_effect_cond`：仍训 L_E，但 Action Expert 看不到 Ê

Generic Future（Ablation 2）用 `--generic-future` 再 cache 一份到独立目录，然后改 `effect_cache_root`。

## 训练时记录的四类指标

1. Action：`loss_fm`
2. Effect：`effect_cosine`
3. Grounding：`wrist_pos_err` / `tip_err`
4. Rollout：success rate

## 测试

```bash
cd "$AFCE_ROOT"
PYTHONPATH="$PWD" python -m unittest effect_vla.tests.test_effect
```
