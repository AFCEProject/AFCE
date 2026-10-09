# C01 E → 单 decoder + finger-Δ

本分支包含 C01 Query E 的数据准备、训练、导出，以及 π0.5 单 decoder + GT-AUX + finger-Δ 的训练、断点恢复、推理和评测代码。策略训练中的 π 与动作 decoder 一起更新，推理使用同一套 decoder；单臂 22 维和双臂 44 维分别有对应的动作头。

## 代码入口

| 环节 | 入口 |
| --- | --- |
| DINO 特征与帧对齐 | `afce_all11/prepare.py`、`prepare_shard.py` |
| 世界变化特征与训练集 PCA | `afce_all11/build_world.py` |
| C01 E 的统计量 | `scripts/c01/prepare_statistics.py`、`afce_all11/calibrated_codec.py` |
| C01 E 训练 | `afce_all11/train_methods.py`、`scripts/c01/train_e.sh` |
| E 模型与几何损失 | `afce_all11/{codec,calibrated_codec,tuned_codec,geometry_objectives}.py`、`effect_afce_v21/model/` |
| E 离线评估 | `afce_all11/eval_expanded.py` |
| E 缓存、decoder 导出 | `afce_all11/export_effect_resume.py`、`export_joint_decoder.py`、`check_joint_decoder.py` |
| π 与 decoder 的目标函数 | `afce_all11/single_finger_alignment.py` |
| 两节点八卡训练 | `afce_all11/multihost_single_finger_pi.py` |
| 恢复检查 | `afce_all11/check_single_finger_gate.py`、`check_single_finger_checkpoint.py` |
| 推理服务 | `afce_all11/serve_single_finger_pi.py` |
| 八 worker 评测队列与汇总 | `afce_all11/single_finger_eval_worker.py`、`single_finger_eval_state.py` |
| Clariden 训练至评测流水线 | `afce_all11/single_finger_worker.sh` |

`openpi/` 和 `dexjoco/dexjoco_openpi_client/` 中的配套修改也包含在此分支中。

## 环境与外部资产

在仓库根目录运行下列命令前，激活相应 Python 环境并设置：

```bash
export PYTHONPATH="$PWD:$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src:$PWD/dexjoco${PYTHONPATH:+:$PYTHONPATH}"
export AFCE_OPENPI_ROOT="$PWD/openpi"
```

E 的原运行环境是 Python/PyTorch CUDA 环境：PyTorch 2.7.1+cu126、NumPy 2.2.6、MuJoCo 3.8.1、SciPy 1.15.3、pandas 2.2.3。特征准备还需要 AV 与 `effect_vla/effect/dino_extractor.py` 使用的 DINOv3/Transformers 依赖。策略运行使用 Clariden `pytorch/v2.8.0:v1` uenv；JAX 0.5.3、Flax 0.10.2、Optax 0.2.8、Orbax 0.11.13，完整实际版本见 `reference/training_contract.json`。这些是原运行环境记录，不代表在所有平台上重新安装并验证过的一套通用锁文件。

需要另外提供 11 任务 LeRobot 数据集（包括视频）、DINOv3 权重、π0.5 基础权重及训练资产。代码仓库不含数据集、模型 checkpoint、E 缓存或评测视频。原 C01 checkpoint、decoder 初始化和缓存的哈希保存在 `reference/`。数据来源见仓库主 README 中的数据集链接。

## C01 E 数据准备与训练

以下流程在仓库根目录运行。`C01_DATA` 指向包含 11 个任务目录的 LeRobot 数据根目录，`C01_DINO` 指向本地 DINOv3 权重目录。

```bash
export C01_DATA=/absolute/path/to/dexjoco_lerobot_datasets
export C01_DINO=/absolute/path/to/dinov3

python -m afce_all11.prepare \
  --data "$C01_DATA" --dino "$C01_DINO" --output "$PWD/runtime/data"
python -m afce_all11.build_world \
  --evidence "$PWD/runtime/data" --mask-mode legacy-none
python scripts/c01/prepare_statistics.py \
  --data "$C01_DATA" --evidence "$PWD/runtime/data" \
  --output "$PWD/runtime/calibrated_statistics.pt"

python -m afce_all11.package_sweep_inputs \
  --repo "$PWD" --data "$C01_DATA" --output "$PWD/runtime/c01-inputs.tar"
mkdir -p runtime/c01-inputs
tar -xf runtime/c01-inputs.tar -C runtime/c01-inputs
python -m afce_all11.verify_sweep_inputs --root "$PWD/runtime/c01-inputs"

CUDA_VISIBLE_DEVICES=0 bash scripts/c01/train_e.sh \
  "$PWD/runtime/c01-inputs" "$PWD/runtime/C01_query_seed42"
# 恢复同一训练时，在上述命令末尾加 --resume。
```

训练入口使用 C01 的 30,000 更新配置：Query ASV、E=30×256、seed 42、batch 32；前 20k 学习率 1e-4，之后 3e-5；AdamW β2=0.99，位置项权重 2，旋转矩阵损失权重 0.05。原 checkpoint 完整配置在 `reference/e_checkpoint_config.json`。`legacy-none` 是原数据协议，表示未使用已验证的机器人分割掩码。

E 使用每任务 episode 0–89 训练、90–99 开发验证；策略训练使用全部 100 个 episode。重新生成统计量或缓存会生成新的文件哈希，不应拿原 checkpoint 的断点状态冒充新训练状态。

## 导出与策略流水线

```bash
python -m afce_all11.export_effect_resume \
  --checkpoint "$PWD/runtime/C01_query_seed42/last.pt" \
  --data "$C01_DATA" --evidence "$PWD/runtime/c01-inputs/runtime/data" \
  --output "$PWD/runtime/c01_effect_cache"
python -m afce_all11.export_joint_decoder \
  --effect-cache "$PWD/runtime/c01_effect_cache" \
  --output "$PWD/runtime/joint_decoder_init.npz"
JAX_PLATFORMS=cpu python -m afce_all11.check_joint_decoder \
  --effect-cache "$PWD/runtime/c01_effect_cache" \
  --decoder-init "$PWD/runtime/joint_decoder_init.npz" \
  --output "$PWD/runtime/decoder_parity.json"
```

策略目标是 `FM(E) + ramp × (0.25628781345139295 × GT-AUX + 0.05 × finger-Δ)`。原运行完成 60,000 optimizer updates，最终目录索引为 `59999`。完整模型、优化器、数据游标与随机数状态随 checkpoint 保存。

`single_finger_worker.sh` 是原 Clariden 流水线，接收两个参数：两节点的 Slurm nodelist、绝对 Unix 保存截止时间；在已有 debug allocation 内执行。它依次完成资产核验、真实 GPU 连续/中断恢复检查、训练、最终 decoder 检查和 seed 0 评测，保存后需后继作业继续时返回 75。

集群入口已改为环境变量：`AFCE_ROOT`、`AFCE_RUNTIME`、`AFCE_QUERY_ROOT`、`AFCE_EXPERIMENT_ROOT`、`AFCE_PI05_BASE_PARAMS`、`AFCE_PYTHON`。跨机器使用前请配置代码、数据、基础权重、π assets、E 缓存和输出位置；原资产核验还读取 GT-AUX calibration 记录。`single_finger_worker.sh` 不是克隆后立即可运行的通用 Slurm 提交器。底层训练和服务入口参数见各自 `--help`。

## 原运行结果与源码校验

以下结果来自随分支保存的已完成记录：训练 seed 42，评测 seed 0，11 任务各 50 局，10 步去噪，两节点八 GPU。总成功率 **284/550 = 51.64%**。

| 任务 | 成功 / 局数 | 成功率 |
| --- | ---: | ---: |
| `bimanual_assembly` | 6/50 | 12% |
| `bimanual_hanoi` | 15/50 | 30% |
| `bimanual_microwave_cook` | 42/50 | 84% |
| `bimanual_photograph` | 32/50 | 64% |
| `bimanual_unlock_ipad` | 4/50 | 8% |
| `click_mouse` | 29/50 | 58% |
| `fold_glasses` | 29/50 | 58% |
| `hammer_nail` | 45/50 | 90% |
| `pick_bucket` | 36/50 | 72% |
| `pinch_tongs` | 4/50 | 8% |
| `water_plant` | 42/50 | 84% |

单臂合计 **185/300 = 61.67%**，双臂合计 **99/250 = 39.60%**。这是单个评测 seed 的结果。

`reference/` 保存 E 配置、策略训练 contract、恢复检查、最终 decoder 检查、完成标记、逐任务 episode 成败与代码哈希。`training_complete.json` 的状态仅表示训练阶段结束；评测完成以 `evaluation_complete.json` 为准。

```bash
python scripts/c01/verify_source.py
```

该命令仅使用 Python 标准库，核对 99 个文件及 E 训练组合哈希。模型和训练核心代码与原实验记录逐字节一致。此次发布进行了源码哈希、Python 语法、Shell 语法和仓内模块依赖检查，没有重新执行 30k E 训练、60k 策略训练或 550 局评测。
