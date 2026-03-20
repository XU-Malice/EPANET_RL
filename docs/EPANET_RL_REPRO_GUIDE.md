# EPANET_RL 论文复现中文教程（从 0 到可训练）

> 文档定位：面向“几乎从 0 开始”的使用者，帮助你看懂仓库、跑通测试、完成训练与评估。
>
> 重要原则：本文只基于当前仓库实际代码与脚本，不编造未实现内容。

---

## 1. 项目总体目标

### 1.1 这个项目在复现什么

当前 `EPANET_RL` 的主线目标是：

- 用强化学习做 EPANET Net3 供水系统的泵调度。
- 将“1 天调度”建模为 24 步序贯决策任务（每步 1 小时）。
- 在满足水力约束前提下，尽可能降低泵能耗成本。

### 1.2 为什么主线环境是 `Net3WntrEnv`

主线环境在 [env_wntr.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/env_wntr.py)（你当前仓库路径下对应 `src/epanet_rl/env_wntr.py`）：

- 每个 step 调用 WNTR/EPANET 做真实水力仿真。
- 能体现论文复现重点：24 步序贯、tank 状态传递、水力违例提前终止等。
- 相比 [env.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/env.py) 的简化版，更适合论文主线训练。

### 1.3 当前 PPO / E-PPO 主线

训练入口是 [train_ppo_net3.py](/home/dengxu/projects/EPANET_RL/scripts/train_ppo_net3.py)：

- `--algo ppo` 或 `--algo eppo`
- `ppo` 可视为 `sigma=0`
- `eppo` 常用 `--sigma 0.2`
- 默认 `r_benchmark=406.54`、`p_hydraulic=-200`、`device=cpu`

---

## 2. 论文信息 vs 当前实现 vs 工程近似（总表）

下面这张表是本项目最关键的“认知对齐”。

| 主题 | 论文明确给出 | 当前仓库已有实现 | 工程可运行近似/说明 |
|---|---|---|---|
| 环境时间语义 | 24 步、每步 1 小时 | `Net3WntrEnv.EPISODE_STEPS=24`，`STEP_SECONDS=3600` | 无 |
| 状态定义 | demand + tank levels | `env_wntr._get_obs()` 按 `[demand, tank]` 拼接 | 无 |
| 动作空间 | 两泵各 8 档，共 64 | `action_space.py` 固定 8 档并双向映射 | 无 |
| 违例处理 | 大惩罚 + 提前终止 | `reward.compute_total_reward()` + `env_wntr.step()` | 无 |
| 奖励主体 | `r_benchmark/24 - E_pump_t` | `reward.compute_regular_reward()` | 无 |
| tank 末步惩罚 | 末步检查，支持 proportional/constant | `reward.compute_tank_penalty()` | 无 |
| Net3 预处理 | 控制规则、Pipe330、电价、效率 | `inp_modifier.py` 有文本级修改流程 | 是否与论文原始工程文件完全逐行一致，需实验核验 |
| PPO 关键超参 | actor/critic 网络与 lr、gamma、clip、epochs | `train_ppo_net3.py` 已参数化并默认对齐 | E-PPO 的“实现细节”通过 SB3 `ent_coef` 映射，不是论文源码逐行复刻 |
| Z-Score | 需统计 mean/std | `compute_zscore_stats.py` 可产出需求统计 | 统计样本规模由你运行参数决定 |

---

## 3. 仓库结构说明

## 3.1 从 0 开始如何读懂本仓库

如果你是第一次接触这个仓库，推荐按下面顺序读，而不是一上来就直接看训练脚本。

### 第 1 步：先建立“论文任务”的三件套概念

先只记住三件事：

1. **状态是什么**：当前仓库实现中，状态由 `demand + tank levels` 组成。
2. **动作是什么**：两台泵、每台 8 档速度，共 64 个离散动作。
3. **奖励是什么**：主体是 `r_benchmark / 24 - E_pump_t`，若发生水力违例则直接给大负惩罚并提前结束。

这三件事分别对应：

- `src/epanet_rl/env_wntr.py`：状态从哪里来、step 做了什么；
- `src/epanet_rl/action_space.py`：64 个动作如何编码；
- `src/epanet_rl/reward.py`：奖励到底怎么结算。

### 第 2 步：再看“需求随机化”和“INP 预处理”

很多初学者一开始会困惑：为什么环境不是简单读取 `Net3.inp` 然后直接跑？

原因是当前论文复现主线里，还有两层前处理：

- **需求随机化**：`src/epanet_rl/demand_randomization.py`
  - 论文明确给出的：时间乘子、空间乘子都来自截断正态；
  - 当前仓库实现：把它们组合成 `(T, N)` 的逐小时逐节点需水矩阵。
- **INP 预处理**：`src/epanet_rl/inp_modifier.py`
  - 论文明确给出的：需要调整控制规则、Pipe330、电价、效率等；
  - 当前仓库实现：通过文本改写生成一个“修改版 INP”。

读懂这两层之后，你再回去看 `env_wntr.py`，会更容易明白环境为什么在 `reset()` / `step()` 中这样组织数据。

### 第 3 步：最后再看训练/评估脚本

当你已经知道“环境、动作、奖励”之后，再看脚本会轻松很多：

- `scripts/train_ppo_net3.py`：负责把环境接到 SB3 PPO 上，并管理训练输出目录；
- `scripts/evaluate_policy_net3.py`：负责离线评估一个保存好的模型；
- `scripts/compute_r_benchmark.py`：帮助你估计 `r_benchmark` 的合理量级；
- `scripts/compute_zscore_stats.py`：帮助你为 Z-Score 缩放准备统计量。

### 第 4 步：把“论文原意”和“仓库工程实现”分开看

读代码时，建议你始终带着下面三个标签：

- **论文明确给出的**：例如 24 步、64 动作、违例提前终止。
- **当前仓库实现**：例如 `info` 里额外输出大量诊断字段，便于排障与统计。
- **工程近似**：例如 NaN fallback、clip 兜底、通过 SB3 参数映射 E-PPO 风格设置。

这样做的好处是：

- 你在写复现实验报告时，不会把工程便利措施误写成论文原话；
- 你在比较结果差异时，更容易判断差异来自论文设定、仓库实现，还是你的实验参数。

### 第 5 步：初学者最推荐的最小阅读路径

如果时间很少，建议至少按下面顺序快速过一遍：

1. `src/epanet_rl/action_space.py`
2. `src/epanet_rl/reward.py`
3. `src/epanet_rl/demand_randomization.py`
4. `src/epanet_rl/inp_modifier.py`
5. `src/epanet_rl/env_wntr.py`
6. `scripts/train_ppo_net3.py`
7. `scripts/evaluate_policy_net3.py`

前 5 个文件回答“任务本身是什么”，后 2 个文件回答“如何把任务拿去训练和评估”。

当前仓库常见顶层目录：

- `src/epanet_rl/`：核心逻辑（环境、奖励、动作、随机化、缩放、INP 处理）。
- `scripts/`：可执行入口（benchmark、zscore、训练、评估、诊断）。
- `tests/`：pytest 测试。
- `networks/`：网络输入文件（`Net3.inp`、`Anytown.inp`）。
- `outputs/`：训练产物（模型、配置、监控日志）。
- `logs/`：命令日志、审计日志。
- `artifacts/`：统计脚本输出（例如 json 结果）。
- `docs/`：文档。

职责划分建议：

- `src` 是“逻辑真相”，不要在脚本里复制核心逻辑。
- `scripts` 只做流程编排、统计和参数入口。
- `tests` 负责论文语义回归和工程契约回归。

---

## 4. 核心源码文件逐个说明

## 4.1 [env_wntr.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/env_wntr.py)

### 文件目标

实现论文复现主线环境 `Net3WntrEnv`。

### 你需要优先看懂的函数

- `__init__`：参数校验、读取并修改 INP、提取网络元数据、初始化 observation/action space。
- `reset`：
  - 生成 demand 随机轨迹（24 小时）
  - 随机初始 tank level
  - 返回初始 observation 和诊断 info
- `step`：
  - 把 `action_id` 解码为两泵速度
  - 调用 `_simulate_single_step()` 跑 1 小时仿真
  - 计算 `e_pump_t`、违例、reward、终止条件
  - 输出丰富 info（含诊断字段）
- `_simulate_single_step`：真正的 WNTR/EPANET 单步仿真与结果抽取。

### 输入输出摘要

- 输入：离散动作 `0..63`
- 输出：Gymnasium 5 元组 `(obs, reward, terminated, truncated, info)`

### 常见坑

- `p_hydraulic` 必须是负值，否则构造时报错。
- step 后若 `terminated=True`，必须先 `reset()` 才能继续。
- `info` 字段很关键，benchmark/诊断都依赖它（特别是 `e_pump_t`）。

## 4.2 [reward.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/reward.py)

### 文件目标

用纯函数实现奖励逻辑，便于测试和复现审计。

### 核心函数

- `compute_regular_reward`: `r_benchmark / 24 - e_pump_t`
- `compute_tank_penalty`: 末步 tank 惩罚（proportional/constant）
- `compute_total_reward`: 统一处理违例优先级、末步惩罚叠加与终止标记

### 关键语义

- 如果 `hydraulic_violation=True`，直接返回 `p_hydraulic` 并 `terminated=True`。
- tank 惩罚只在最后一步检查。

## 4.3 [action_space.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/action_space.py)

### 文件目标

定义论文动作空间并提供双向映射。

### 关键对象

- `ACTION_LEVELS = (0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)`
- `ACTION_COUNT = 64`
- `action_id_to_speeds` / `speeds_to_action_id`

## 4.4 [demand_randomization.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/demand_randomization.py)

### 文件目标

按论文两步法生成 demand 随机化：

1. 时间乘子（time multipliers）
2. 空间乘子（space multipliers）
3. 最终需求矩阵组合

### 关键函数

- `sample_truncated_normal`
- `generate_time_multipliers`
- `generate_space_multipliers`
- `compose_randomized_demands`
- `generate_randomized_demands`

## 4.5 [inp_modifier.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/inp_modifier.py)

### 文件目标

把原始 Net3 INP 转成复现实验更可控的版本。

### 主要做的事

- 删除目标控制规则（`CONTROLS`/`RULES`）。
- 确保 Pipe330 在 `STATUS` 常闭。
- 设置 TOU 电价相关 `ENERGY` 与 `PATTERNS` 内容。
- 设置全局效率参数。

## 4.6 [scaling.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/scaling.py)

### 文件目标

提供需求和水箱特征缩放：

- Max-Min
- Z-Score

### 说明

- 这是纯函数工具模块，环境调用它，不在这里维护 episode 状态。

## 4.7 [env.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/env.py)

### 文件定位

- 简化环境，更多用于流程联调和早期 smoke test。
- 论文主线训练不使用这个环境。

---

## 5. scripts 脚本逐个说明

## 5.1 [compute_r_benchmark.py](/home/dengxu/projects/EPANET_RL/scripts/compute_r_benchmark.py)

### 作用

随机策略 rollout，统计能耗 benchmark 多口径结果。

### 什么时候用

- 你想核对当前环境下随机策略基准能耗时。

### 常用命令

```bash
python scripts/compute_r_benchmark.py \
  --episodes 200 \
  --seed 42 \
  --net3-inp networks/Net3.inp \
  --scaling-mode max_min \
  --delta-time 0.3 \
  --delta-space 0.3 \
  --p-hydraulic -200 \
  --progress
```

### 输出怎么看

重点关注：

- `candidate_benchmark_all`
- `candidate_benchmark_successful`
- `candidate_benchmark_full_horizon`
- fallback/non_fallback 分组

## 5.2 [compute_zscore_stats.py](/home/dengxu/projects/EPANET_RL/scripts/compute_zscore_stats.py)

### 作用

统计状态特征均值方差（供 Z-Score 缩放使用）。

### 常用命令

```bash
python scripts/compute_zscore_stats.py \
  --episodes 200 \
  --seed 42 \
  --net3-inp networks/Net3.inp \
  --delta-time 0.3 \
  --delta-space 0.3 \
  --progress \
  --output-json artifacts/zscore_stats_net3.json
```

### 输出怎么看

- `demand_mean` / `demand_std`
- `tank_mean` / `tank_std`

## 5.3 [train_ppo_net3.py](/home/dengxu/projects/EPANET_RL/scripts/train_ppo_net3.py)

### 作用

主线训练入口（PPO / E-PPO）。

### 常用命令

PPO 冒烟：

```bash
python scripts/train_ppo_net3.py \
  --algo ppo \
  --total-timesteps 2000 \
  --seed 42 \
  --net3-inp networks/Net3.inp \
  --delta-time 0.1 \
  --delta-space 0.1 \
  --scaling-mode max_min \
  --r-benchmark 406.54 \
  --p-hydraulic -200 \
  --device cpu
```

E-PPO 冒烟：

```bash
python scripts/train_ppo_net3.py \
  --algo eppo \
  --sigma 0.2 \
  --total-timesteps 2000 \
  --seed 42 \
  --net3-inp networks/Net3.inp \
  --delta-time 0.1 \
  --delta-space 0.1 \
  --scaling-mode max_min \
  --r-benchmark 406.54 \
  --p-hydraulic -200 \
  --device cpu
```

### 输出怎么看

每个 `outputs/ppo_net3/run_*` 中至少有：

- `train_config.json`
- `train_summary.json`
- `optimizer_lrs.json`
- `vec_monitor.csv`
- `sb3_logs/progress.csv`

## 5.4 [evaluate_policy_net3.py](/home/dengxu/projects/EPANET_RL/scripts/evaluate_policy_net3.py)

### 作用

离线评估训练好的策略，输出 reward/energy/tank 审计指标。

### 常用命令

```bash
python scripts/evaluate_policy_net3.py \
  --model-path outputs/ppo_net3/run_xxx/ppo_net3_model.zip \
  --algo ppo \
  --episodes 100 \
  --seed 42 \
  --net3-inp networks/Net3.inp \
  --delta-time 0.1 \
  --delta-space 0.1 \
  --scaling-mode max_min \
  --r-benchmark 406.54 \
  --p-hydraulic -200 \
  --output-json artifacts/eval_ppo.json
```

### 输出怎么看

重点指标：

- `mean_total_reward`
- `mean_total_energy_cost`
- `hydraulic_violation_rate`
- `successful_episode_rate`
- `successful_mean_total_energy_cost`
- `mean_total_tank_penalty`
- `mean_volume_change_ratio`

## 5.5 [diagnose_env_wntr_rollout.py](/home/dengxu/projects/EPANET_RL/scripts/diagnose_env_wntr_rollout.py)

### 作用

逐步打印环境关键字段，用于定位早停、违例、压力异常等问题。

### 常用命令

```bash
python scripts/diagnose_env_wntr_rollout.py --seed 42 --max-steps 24
```

---

## 6. tests 测试文件说明

核心测试文件及作用：

- [test_action_space.py](/home/dengxu/projects/EPANET_RL/tests/test_action_space.py)
  - 验证 64 动作定义与映射可逆。
- [test_demand_randomization.py](/home/dengxu/projects/EPANET_RL/tests/test_demand_randomization.py)
  - 验证截断正态范围与 demand 组合公式。
- [test_reward.py](/home/dengxu/projects/EPANET_RL/tests/test_reward.py)
  - 验证 reward 主结构与末步 tank penalty。
- [test_scaling.py](/home/dengxu/projects/EPANET_RL/tests/test_scaling.py)
  - 验证缩放边界与零方差稳定性。
- [test_inp_modifier_net3.py](/home/dengxu/projects/EPANET_RL/tests/test_inp_modifier_net3.py)
  - 验证 Net3 修改规则是否按预期生效。
- [test_env_wntr_contract.py](/home/dengxu/projects/EPANET_RL/tests/test_env_wntr_contract.py)
  - 验证主环境关键契约（违例终止、fallback、参数校验）。
- [test_env_wntr_paper_requirements.py](/home/dengxu/projects/EPANET_RL/tests/test_env_wntr_paper_requirements.py)
  - 验证论文语义相关需求（24步/状态结构/动作等）。

如何区分：

- 论文需求导向：`*paper_requirements.py`
- 工程稳定性导向：`*contract.py` + 各模块纯函数单测

---

## 7. 环境准备与依赖检查

## 7.1 进入虚拟环境

```bash
cd /home/dengxu/projects/EPANET_RL
source ~/epanet_rl/bin/activate
```

## 7.2 检查 python 路径

```bash
which python
python -V
python -c "import sys; print(sys.executable)"
```

## 7.3 检查核心依赖

```bash
python -c "import numpy, gymnasium, wntr; print(numpy.__version__, gymnasium.__version__, wntr.__version__)"
python -c "import torch, stable_baselines3 as sb3; print(torch.__version__, sb3.__version__)"
```

## 7.4 检查 GPU 可见性（可选）

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 7.5 为什么训练默认建议 CPU

当前是 `MlpPolicy` + 单环境主流程，CPU 训练通常更稳定、更易复现和排障，因此脚本默认 `--device cpu`。

---

## 8. 分步测试教程（命令 + 结果解释）

## 8.1 全量测试

```bash
python -m pytest -q
```

正常现象：

- 大部分测试通过。
- 若有 `xfailed`，通常是仓库中明确标记的“暂未完全实现项”。

异常通常意味着：

- 依赖缺失（wntr/torch/sb3 等）。
- 环境或奖励逻辑契约回归。

## 8.2 单步环境合同检查

```bash
python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path('src').resolve()))
from epanet_rl.env_wntr import Net3WntrEnv

env = Net3WntrEnv(net3_inp_path='networks/Net3.inp', scaling_mode='max_min', r_benchmark=406.54, p_hydraulic=-200)
obs, info = env.reset(seed=42)
obs2, reward, terminated, truncated, info2 = env.step(63)
print('obs_shape', obs.shape, 'next_obs_shape', obs2.shape)
print('reward', reward, 'terminated', terminated, 'truncated', truncated)
print('e_pump_t', info2.get('e_pump_t'))
print('hydraulic_violation', info2.get('hydraulic_violation'))
print('termination_reason', info2.get('termination_reason'))
env.close()
PY
```

## 8.3 诊断 rollout

```bash
python scripts/diagnose_env_wntr_rollout.py --seed 42 --max-steps 24
```

## 8.4 r_benchmark 统计

```bash
python scripts/compute_r_benchmark.py --episodes 200 --seed 42 --net3-inp networks/Net3.inp --scaling-mode max_min --delta-time 0.3 --delta-space 0.3 --p-hydraulic -200 --progress
```

## 8.5 zscore 统计

```bash
python scripts/compute_zscore_stats.py --episodes 200 --seed 42 --net3-inp networks/Net3.inp --delta-time 0.3 --delta-space 0.3 --progress
```

## 8.6 PPO 冒烟训练

```bash
python scripts/train_ppo_net3.py --algo ppo --total-timesteps 2000 --seed 42 --net3-inp networks/Net3.inp --delta-time 0.1 --delta-space 0.1 --scaling-mode max_min --r-benchmark 406.54 --p-hydraulic -200 --device cpu
```

## 8.7 E-PPO 冒烟训练

```bash
python scripts/train_ppo_net3.py --algo eppo --sigma 0.2 --total-timesteps 2000 --seed 42 --net3-inp networks/Net3.inp --delta-time 0.1 --delta-space 0.1 --scaling-mode max_min --r-benchmark 406.54 --p-hydraulic -200 --device cpu
```

## 8.8 评估训练结果

```bash
python scripts/evaluate_policy_net3.py --help
```

---

## 9. 整体测试与复现流程（推荐顺序）

建议严格按以下顺序：

1. 激活环境 + 依赖检查。
2. 运行 `pytest -q` 建立回归基线。
3. 做单步检查与诊断 rollout。
4. 跑 `compute_r_benchmark.py` 了解当前基线。
5. 跑 `compute_zscore_stats.py` 生成缩放统计。
6. 先做 PPO 冒烟，再做 E-PPO 冒烟。
7. 用 `evaluate_policy_net3.py` 对比两种算法。
8. 再进入长程正式训练和论文结果对照。

日志和结果查看路径：

- 训练输出：`outputs/ppo_net3/run_*`
- 统计 JSON：你在命令里指定的 `--output-json`
- 运行日志：`logs/*.log`

---

## 10. 当前与论文仍可能有差异的点

### 10.1 论文明确给出的，当前已对齐

- 24 步、每步 1 小时语义。
- 两泵 8 档，共 64 动作。
- 违例大惩罚与提前终止。
- PPO 关键超参数入口（network/lr/gamma/clip/epochs）。

### 10.2 当前仓库已有实现，但仍需实验验证效果

- `inp_modifier.py` 对 Net3 的规则、电价、效率改造流程。
- `compute_r_benchmark.py` 多口径 benchmark 统计。
- `evaluate_policy_net3.py` 奖励分解与 tank 末端状态审计。

### 10.3 工程可运行近似（非论文源码逐行复刻）

- E-PPO 在训练脚本中通过 SB3 `ent_coef` 映射实现。
- critic `[256,128,1]` 在 SB3 表达为 `vf=[256,128] + 隐式 value head`。
- `r_benchmark` 默认可直接传论文值 `406.54`，同时保留本地统计脚本结果作参考。

### 10.4 为什么当前建议冻结 `env_wntr.py`

你的主线目标已从“修环境”转到“做对照实验与复现流程”。

- 训练阶段频繁改环境会破坏可复现性。
- 先固定环境，再做 PPO/E-PPO 对照更稳。

---

## 11. 后续建议（按优先级）

1. 固定一组训练配置，分别跑 PPO 与 E-PPO（`sigma=0.2`）。
2. 用统一评估脚本在相同 seeds 下做对照。
3. 汇总 reward / energy / tank 指标并与论文目标比对。
4. 仅在“证据充分”时再回到环境细节微调。

---

## 12. 注释补充计划（不改逻辑）

如果继续补中文注释，建议优先级：

1. [env_wntr.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/env_wntr.py)：`step` 与 `_simulate_single_step` 的字段和失败分支。
2. [train_ppo_net3.py](/home/dengxu/projects/EPANET_RL/scripts/train_ppo_net3.py)：分离学习率与 SB3 覆盖机制。
3. [compute_r_benchmark.py](/home/dengxu/projects/EPANET_RL/scripts/compute_r_benchmark.py)：多口径统计定义。
4. [evaluate_policy_net3.py](/home/dengxu/projects/EPANET_RL/scripts/evaluate_policy_net3.py)：成功子集与 tank 审计指标。
5. [demand_randomization.py](/home/dengxu/projects/EPANET_RL/src/epanet_rl/demand_randomization.py)：两步随机化公式与 mask 语义。



## 批量复现实验

### 为什么新增 suite 脚本

当你已经确认单次 `train_ppo_net3.py` / `evaluate_policy_net3.py` 可以工作后，下一步通常不是“手工一条条命令反复敲”，而是系统化地做 PPO / E-PPO 对照实验。

因此当前仓库增加了：

- `scripts/run_reproduction_suite.py`

它的定位是：

- 只做**流程编排**，不改 `env_wntr.py` 主逻辑；
- 自动按实验矩阵批量训练；
- 每个训练完成后自动调用评估脚本；
- 最后输出 suite 级别的 `summary_json / summary_csv`。

### 论文明确给出的 / 当前仓库实现 / 工程优化

这一节非常重要，建议你在写实验记录时直接照着这个口径整理。

#### 论文明确给出的

- 主线环境固定为 `Net3WntrEnv`
- 24 步决策、每步 1 小时
- PPO 可视为 `sigma=0`
- E-PPO 对照重点是 `sigma=0.2`

#### 当前仓库已有实现

- `scripts/train_ppo_net3.py` 支持 `--algo ppo` 与 `--algo eppo`
- 当前仓库中，`algo=ppo` 时即使传入非零 `sigma`，训练脚本也会忽略它，并把有效 entropy 系数视为 0
- `scripts/evaluate_policy_net3.py` 会输出 `energy cost / reward / tank penalty / volume_change_ratio` 等评估指标

#### 工程优化

- 为避免无效重复实验，`scripts/run_reproduction_suite.py` 在构建实验矩阵时：
  - `ppo` 只生成 `sigma=0`
  - `eppo` 才展开用户传入的 `sigma` 列表
- 这不会改变论文口径，反而能避免生成一堆“名义上 sigma 不同、实际上训练等价”的 `ppo` 重复实验

### 如何运行批量 suite

推荐服务器命令示例：

```bash
python scripts/run_reproduction_suite.py \
  --algos ppo eppo \
  --sigmas 0 0.2 0.3 \
  --delta 0.3 \
  --timesteps 100000 \
  --seeds 1 2 3 \
  --eval-episodes 20 \
  --device cpu \
  --output-dir outputs/reproduction_suite \
  --force
```

如果你要切到 200000 步，只需要把：

```bash
--timesteps 100000
```

改成：

```bash
--timesteps 200000
```

### suite 会生成什么结果

每次运行都会在你指定的 `--output-dir` 下再创建一个：

- `suite_时间戳/`

其中通常包含：

- `suite_manifest.json`
  - 记录这次 suite 打算跑哪些实验格子
- `suite_summary.json`
  - 结构化总汇总，适合后续程序读取
- `suite_summary.csv`
  - 表格化总汇总，适合直接用 pandas / Excel / LibreOffice 看
- 每个实验一个独立目录
  - 里面会有 `train.log`、`eval.log`、`experiment_meta.json`、`eval_summary.json` 等文件

### 如何看 `summary_json / summary_csv`

#### `suite_summary.json`

更适合：

- 写自动分析脚本
- 做二次聚合
- 保留“论文给定 / 仓库实现 / 工程近似”的元数据说明

重点关注：

- `suite_config`
- `suite_stats`
- `experiments[]`

其中 `experiments[]` 里会保留每个实验的：

- `algo`
- `sigma`
- `seed`
- `status`
- `mean_total_reward`
- `mean_total_energy_cost`
- `mean_total_tank_penalty`
- `mean_volume_change_ratio`
- `hydraulic_violation_rate`
- `full_horizon_rate`
- `successful_episode_rate`

#### `suite_summary.csv`

更适合：

- 快速横向比较不同 seed / 不同 sigma
- 直接画表
- 先人工筛查哪些实验值得深入分析

一个很实用的经验是：

1. 先看 `mean_total_energy_cost`
2. 再看 `mean_total_tank_penalty`
3. 再看 `mean_volume_change_ratio`
4. 最后结合 `successful_episode_rate` 判断是不是“靠透支 tank volume 换低能耗”

### 如何断点续跑

如果服务器训练到一半断了，不建议把整个输出目录删掉重来。

你可以重新执行同类命令，并加上：

```bash
--skip-existing
```

它的语义是：

- 如果某个实验目录下已经存在 `eval_summary.json`
- suite 就直接读取已有结果，不再重复训练和评估

这对多 seed、大时间步长实验非常有用。

推荐断点续跑示例：

```bash
python scripts/run_reproduction_suite.py \
  --algos ppo eppo \
  --sigmas 0 0.2 0.3 \
  --delta 0.3 \
  --timesteps 200000 \
  --seeds 1 2 3 \
  --eval-episodes 20 \
  --device cpu \
  --output-dir outputs/reproduction_suite \
  --force \
  --skip-existing
```

### 使用建议

如果你当前最关心的是“penalty form 和 PPO / E-PPO 对照”，建议优先这样组织实验：

1. 先固定 `delta=0.3`
2. 先跑 `ppo(sigma=0)` 与 `eppo(sigma=0.2)`
3. 再补 `eppo(sigma=0.3)` 看熵增强是否进一步放大“省电但透支 tank”现象
4. 优先同时看：
   - `energy cost`
   - `tank penalty`
   - `volume_change_ratio`
   - `successful_episode_rate`

这样更容易把“真优化”与“投机策略”分开。
