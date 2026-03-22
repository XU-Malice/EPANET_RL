# EPANET_RL 与论文严格复现方案的差距分析与下一步计划

## 1. 文档目的

本文对比当前仓库实现与目标论文复现方案，回答三件事：

1. **哪些内容已经完成或基本完成**；
2. **哪些地方仍与“严格论文版最小复现”存在差距**；
3. **下一步应该按什么优先级修改和完善**。

本文只基于当前仓库的静态代码、脚本、测试和文档进行分析，不依赖运行结果。所有判断均尽量附上当前仓库中的文件路径与行号，便于审计。

---

## 2. 审计范围与依据

本次对比重点检查了以下文件：

- 环境与状态/动作/奖励
  - `src/epanet_rl/env_wntr.py`
  - `src/epanet_rl/env.py`
  - `src/epanet_rl/action_space.py`
  - `src/epanet_rl/reward.py`
  - `src/epanet_rl/demand_randomization.py`
  - `src/epanet_rl/inp_modifier.py`
- 训练与评估/实验脚本
  - `scripts/train_ppo_net3.py`
  - `scripts/evaluate_policy_net3.py`
  - `scripts/run_reproduction_suite.py`
- 回归测试与项目文档
  - `tests/test_action_space.py`
  - `tests/test_env_paper_requirements.py`
  - `tests/test_env_wntr_paper_requirements.py`
  - `README.md`
  - `docs/EPANET_RL_REPRO_GUIDE.md`
  - `docs/CODEBASE_UNDERSTANDING_CN.md`

---

## 3. 总体结论

### 3.1 当前仓库已经具备的“论文复现骨架”

当前工程**并不是空壳**，已经具备严格论文复现所需的几个核心骨架：

1. **有真实水力环境主线**：`Net3WntrEnv` 已经使用 `wntr.sim.EpanetSimulator` 执行单步仿真，而不是只停留在玩具环境层面，见 `src/epanet_rl/env_wntr.py:511`。
2. **有论文风格的状态定义**：WNTR 主线环境观测维度只由 `junction demands + tank levels` 组成，测试也明确校验了 `92 + 3 = 95` 维，见 `src/epanet_rl/env_wntr.py:190-191, 203-245` 与 `tests/test_env_wntr_paper_requirements.py:39-48`。
3. **有论文风格的奖励骨架**：已实现 `r_benchmark / 24 - e_pump_t`、水力违例大惩罚、末步 tank penalty 的三级逻辑，见 `src/epanet_rl/reward.py:94-198`。
4. **有需求随机化实现**：已实现时间乘子、空间乘子、截断正态和按 `(T, N)` 组合需水矩阵，见 `src/epanet_rl/demand_randomization.py:45-123, 126-200, 203-248`。
5. **有 Net3 工况修改器**：已实现 controls/rules 删除、Pipe 330 常闭、TOU 电价和全局效率改写，见 `src/epanet_rl/inp_modifier.py:107-308`。
6. **有 PPO/E-PPO 训练入口**：训练脚本已经显式对齐论文里的 actor/critic 网络、学习率、gamma、clip、epochs 等关键超参数，见 `scripts/train_ppo_net3.py:140-183, 269-278, 319-324, 653-666`。
7. **有统一评估入口**：可对训练完成的 PPO/E-PPO 模型做统一聚合评估，见 `scripts/evaluate_policy_net3.py:209-286`。

因此，这个仓库当前最准确的定位不是“从零开始”，而是：

> **已经完成了论文复现的主干实现，但还没有收缩到“严格论文版最小复现”，也还没有把论文 3.1 / 3.2 / 3.3 三阶段实验完整、可审计地落地。**

### 3.2 当前最关键的问题

当前最大的偏差不是“没有环境”，而是**工程实现已经向可运行、可诊断方向扩展，但与用户要求的“严格论文版最小复现”还不完全一致**。最典型的偏差包括：

1. **动作空间仍是 8 档、64 动作，并包含 `0.0` 停泵档**，而目标方案要求严格切回论文中的 `0.70~1.00` 七档、49 个组合动作。
   - 证据：`src/epanet_rl/action_space.py:23-25`
   - 测试也固定把“论文动作定义”写成 64 动作：`tests/test_action_space.py:21-24`
2. **环境默认 `r_benchmark` 仍是 `2000.0` 占位值**，而目标方案要求主复现实验固定为论文给出的 `406.54`。
   - 证据：`src/epanet_rl/env_wntr.py:104`
   - 测试明确写明“当前还是 placeholder”：`tests/test_env_wntr_paper_requirements.py:149-155`
3. **tank penalty 仍以“constant / proportional”通用配置方式存在**，还没有实现论文 3.1 所需的 6 种 penalty 口径及其固定实验顺序。
   - 证据：`src/epanet_rl/reward.py:20-44, 115-142`
4. **E-PPO 目前是用 SB3 PPO 的 `ent_coef=sigma` 做参数映射**，而不是显式独立的 `PPOAgent / EPPOAgent` 结构。
   - 证据：`scripts/train_ppo_net3.py:269-278, 653-666`
5. **实验层尚未按论文顺序拆成 `3.1 penalty -> 3.2 entropy -> 3.3 model comparison` 三个正式脚本**；当前只有一个泛化的 `run_reproduction_suite.py`。
   - 证据：`scripts/run_reproduction_suite.py:95-182`
6. **尚未实现 GA / PSO / DE 基线**，因此 3.3 model comparison 还不具备论文级可比性。
   - 证据：仓库中没有 `ga_runner.py / pso_runner.py / de_runner.py` 等对应实现，`rg` 仅命中现有文档和说明，未见基线代码。
7. **工程增强仍混入主线环境**，例如 NaN fallback 后回退到 `(0.0, 0.0)` 执行动作，这与“严格论文版最小复现、暂不混入停泵档和工程增强”的目标不一致。
   - 证据：`src/epanet_rl/env_wntr.py:289-294, 326-331, 660`

---

## 4. 对照结果总表

| 模块 | 目标方案要求 | 当前状态 | 结论 |
|---|---|---|---|
| 案例网络 | Net3 论文工况，去控制、330 常闭、双泵受控、TOU 电价、效率 0.75 | 已有 `inp_modifier.py` 完成大部分文本级改造 | **基本完成，但缺少 330/335 歧义的显式审计说明** |
| 环境时间语义 | 24h / 1h step / 一天一集 | `Net3WntrEnv` 已实现 24 步、3600 秒 | **完成** |
| 状态空间 | 仅 `[demands, tank_levels]` | WNTR 主线已符合 | **完成** |
| 需水随机化 | 两阶段截断正态；大用户不随机化 | 两阶段随机化已实现；可通过 mask 排除部分节点 | **基本完成，但需明确映射“大用户节点名单”** |
| tank 初值 | 有效液位区间均匀采样 | 已实现 uniform 采样 | **完成** |
| 动作空间 | 七档 `0.70~1.00`，共 49 个离散组合，不含停泵档 | 当前是八档，含 `0.0`，共 64 | **未完成，且是主线偏差** |
| 奖励骨架 | 能耗 + 水力惩罚 + 日末水箱惩罚 | 结构已具备 | **基本完成** |
| hydraulic penalty | 失败即 `-200` 并终止 | 已支持 `p_hydraulic`，训练脚本默认 `-200` | **完成** |
| benchmark reward | 主实验固定 `406.54` | 训练脚本默认 406.54，但环境默认仍 2000 | **部分完成** |
| tank penalty 研究 | 需要 6 种 penalty 的正式实验 | 仅有通用 config，没有 6 种论文口径 | **未完成** |
| PPO 参数 | actor/critic 网络、学习率、gamma、clip、epochs 尽量对齐表 1 | 训练脚本已显式对齐主要参数 | **基本完成** |
| E-PPO | 仅在 policy objective 中加入 entropy bonus，critic 不变 | 当前通过 SB3 `ent_coef` 做近似实现 | **部分完成** |
| 状态归一化 | 为提升初始熵，对状态做归一化 | 已有 `max_min` / `z_score` | **完成，但论文统计标定仍待完成** |
| 实验 3.1 | penalty study | 未单独落地 | **未完成** |
| 实验 3.2 | entropy study | 未单独落地 | **未完成** |
| 实验 3.3 | model comparison with PPO/E-PPO/GA/PSO/DE | 未落地基线算法 | **未完成** |
| 工程结构 | 论文映射式目录结构 | 当前仍是单包聚合结构 | **可运行，但与目标结构不一致** |

---

## 5. 分模块详细对比

## 5.1 案例网络与 Net3 工况改造

### 已完成内容

当前 `inp_modifier.py` 已覆盖论文工况改造中的大部分关键动作：

- 删除相关 controls/rules：`src/epanet_rl/inp_modifier.py:107-125`
- 强制 Pipe 330 为 Closed：`src/epanet_rl/inp_modifier.py:127-155`
- 重写 `ENERGY`，设置全局效率、谷电价和电价 pattern：`src/epanet_rl/inp_modifier.py:173-220`
- 插入 24 小时 TOU 电价 pattern：`src/epanet_rl/inp_modifier.py:222-269`
- 主修改流程固定按“控制 -> 状态 -> 能源 -> pattern”执行：`src/epanet_rl/inp_modifier.py:287-308`

测试也已经把这些要求写入回归断言：

- 删除与 10 / 330 / 335 相关控制：`tests/test_inp_modifier_net3.py:48-59`
- 330 常闭：`tests/test_inp_modifier_net3.py:56-59`
- 全局效率 75、价格与 pattern 写入：`tests/test_inp_modifier_net3.py:62-85`

### 与目标方案的差距

1. **论文中 330/335 记号冲突尚未在仓库中形成显式审计说明。**
   当前代码确实把 10 / 330 / 335 都视为需要清理或改造的对象，但仓库中还没有一份明确说明，解释“为什么最终按 pumps 10/335 受控、pipe 330 常闭来解析论文歧义”。
2. **全局效率写法仍是全局参数，不是按泵级别显式标注 10 / 335。**
   这在 Net3 双泵场景下通常可以工作，但如果要追求论文复现审计性，建议在后续文档和配置中补充“当前实现为何使用全局效率”的说明，或改为更显式的泵级设置。

### 结论

这一块已经非常接近论文目标，**优先级高但不是最大缺口**；更需要补的是“论文歧义解释和审计说明”。

---

## 5.2 状态空间与 episode 语义

### 已完成内容

`Net3WntrEnv` 已经满足论文环境主线的几个关键点：

- episode 固定 24 步：`src/epanet_rl/env_wntr.py:89`
- 每步 1 小时、3600 秒：`src/epanet_rl/env_wntr.py:90-91`
- reset 时一次性生成 24 小时 demand 轨迹：`src/epanet_rl/env_wntr.py:217-234`
- reset 时 tank 初值在 `[min, max]` 上均匀采样：`src/epanet_rl/env_wntr.py:238-239`
- observation space 由 demands 与 tank levels 组成：`src/epanet_rl/env_wntr.py:190-191, 700-740`

对应测试：

- `tests/test_env_wntr_paper_requirements.py:29-48`
- `tests/test_env_wntr_paper_requirements.py:64-81`

### 与目标方案的差距

这一块与用户给出的“严格论文版最小复现”是**最接近的一块**。主要差距不在状态定义本身，而在两个附属点：

1. **大用户节点不随机化的论文要求，当前仅有 `randomizable_mask` 能力，但没有仓库级固定名单或配置。**
   - 证据：`src/epanet_rl/demand_randomization.py:155-200`
   - 环境端也只是构造 mask：`src/epanet_rl/env_wntr.py:186, 747-769`
2. **状态归一化统计虽然已具备接口，但大规模统计标定仍处于“待完成”状态。**
   - 证据：`tests/test_env_wntr_paper_requirements.py:160-172` 标记了 xfail

### 结论

**状态空间主线已完成。** 后续只需补“非随机化节点名单”和“状态归一化统计口径”两个审计问题。

---

## 5.3 需水随机化

### 已完成内容

当前实现已经很好地贴近论文的“两阶段随机化”定义：

- 截断正态采样器：`src/epanet_rl/demand_randomization.py:45-123`
- 时间乘子生成：`src/epanet_rl/demand_randomization.py:126-152`
- 空间乘子生成：`src/epanet_rl/demand_randomization.py:155-200`
- 最终需求矩阵组合公式：`src/epanet_rl/demand_randomization.py:203-248`
- 一站式 `generate_randomized_demands`：`src/epanet_rl/demand_randomization.py:251-299`

测试也已经覆盖了：

- 乘子范围
- `delta=0`
- mask 固定节点
- 组合公式
- 返回形状

见 `tests/test_demand_randomization.py:19-94`。

### 与目标方案的差距

1. **采样实现带有 clip fallback，是偏工程稳健性的近似，不是严格理论截断采样。**
   - 证据：`src/epanet_rl/demand_randomization.py:65-73, 116-121`
2. **尚未在项目配置层面明确区分 `Δ=0.3 / 0.6 / 0.9` 的正式实验集。**
   当前随机化实现本身已可支持，但实验层没有按论文正式拆分。
3. **“大用户节点不随机化”在论文中是固定规则，在当前仓库中仍是可选接口。**

### 结论

**算法层面基本完成，实验口径层面未完成。**

---

## 5.4 动作空间

### 当前状态

当前动作空间是：

- `ACTION_LEVELS = (0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)`
- `ACTION_COUNT = 64`

见 `src/epanet_rl/action_space.py:23-25`。

测试也把这一定义当成“论文动作定义”固定了下来：

- `tests/test_action_space.py:21-24`

WNTR 主线环境直接将其作为离散动作空间：

- `src/epanet_rl/env_wntr.py:190`
- `tests/test_env_wntr_paper_requirements.py:52-60`

### 与目标方案的差距

这是当前仓库与目标方案之间**最明确、最结构性的冲突**：

1. **目标方案要求 7 档离散泵速：**
   `0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00`
2. **当前实现多出了 `0.0` 停泵档**，导致动作数从 49 变为 64。
3. **环境 fallback 还会主动退回 `(0.0, 0.0)` 动作**，说明“停泵”不仅存在于动作空间中，还存在于异常处理主线中。
   - 证据：`src/epanet_rl/env_wntr.py:289-294, 326-331`
4. **测试、文档、环境合同都已经围绕 64 动作构建**，这意味着动作空间一旦修正，将牵动较多文件同步修改。

### 结论

如果项目要切到“严格论文版最小复现”，**动作空间是必须优先修正的第一项代码改造**。

---

## 5.5 奖励函数与 terminal penalty

### 已完成内容

当前奖励实现已经具备论文 Algorithm 1 的主体框架：

- 常规奖励：`compute_regular_reward`，见 `src/epanet_rl/reward.py:94-112`
- 末步 tank penalty：`compute_tank_penalty`，见 `src/epanet_rl/reward.py:115-142`
- 水力违例优先、末步叠加 penalty 的总装配：`compute_total_reward`，见 `src/epanet_rl/reward.py:145-198`

而且已经被环境接入：

- `src/epanet_rl/env_wntr.py:412-424`

### 与目标方案的差距

1. **论文 3.1 需要 6 种固定 penalty 形式，而当前仅有 `constant / proportional` 两类抽象模式。**
2. **当前 proportional 模式是“系数 × shortfall_ratio × r_benchmark”，虽然可以表达论文的 penalty3~6，但没有固定命名、没有实验枚举，也没有把 penalty1/2 和 penalty3~6 收敛到统一论文接口。**
3. **`r_benchmark` 在环境默认值仍是 2000.0，导致奖励模块虽然公式正确，但默认实验口径不正确。**
4. **论文要求日末水箱惩罚只在 `V24 < V0` 时追加，当前实现是基于 `final_tank_volume < initial_tank_volume`，这在语义上基本一致，但需要在复现实验说明中固定“体积口径”。**

### 结论

**奖励框架已完成，论文 penalty study 尚未完成。** 这正是 3.1 实验需要补的核心。

---

## 5.6 WNTR 主线环境与工程增强

### 已完成内容

`Net3WntrEnv` 已经不只是接口骨架，而是完整的真实仿真环境主线：

- 使用真实 `EpanetSimulator`：`src/epanet_rl/env_wntr.py:511`
- 抽取 tank level、pressure、pump flows、pump head gains：`src/epanet_rl/env_wntr.py:515-554`
- 计算能耗成本：`src/epanet_rl/env_wntr.py:664-698`
- 执行 reward 结算并回传丰富 info：`src/epanet_rl/env_wntr.py:412-477`

测试也明确把它视为论文主线环境，而不是玩具环境：

- `tests/test_env_wntr_paper_requirements.py:86-145`

### 与目标方案的差距

1. **主线环境仍混入工程 fallback。**
   当前一旦仿真异常或 NaN，可回退到 `(0.0, 0.0)` 执行动作；这和“严格论文版最小复现，暂不混入停泵档和工程增强”的目标冲突。
2. **当前 `pressure_violation_threshold` 默认值是 `0.0`，需要进一步确认是否与论文水力约束口径完全一致。**
   - 证据：`src/epanet_rl/env_wntr.py:106, 529-533`
3. **环境内部仍保留较多审计字段和容错路径，这很好，但需要通过配置把“strict paper mode”与“engineering robust mode”明确分开。**

### 结论

环境主线本身是可用的，**真正需要改的是把“论文严格模式”从“工程稳健模式”中剥离出来**。

---

## 5.7 PPO / E-PPO 算法层

### 已完成内容

训练脚本已经对齐了表 1 的大部分关键超参数：

- `r_benchmark=406.54`：`scripts/train_ppo_net3.py:140-141`
- `p_hydraulic=-200`：`scripts/train_ppo_net3.py:144-147`
- actor lr / critic lr / gamma / clip / epochs：`scripts/train_ppo_net3.py:160-183`
- actor `[256, 128, 64]`，critic `[256, 128] + implicit head`：`scripts/train_ppo_net3.py:319-324`
- 分离 actor/critic 学习率：`scripts/train_ppo_net3.py:330-392`

### 与目标方案的差距

1. **当前 E-PPO 不是独立实现，而是通过 SB3 PPO 的 `ent_coef=sigma` 做近似映射。**
   - 证据：`scripts/train_ppo_net3.py:269-278, 653-666`
2. **算法结构尚未形成论文导向的独立模块，如 `ppo.py` 与 `eppo.py`。**
3. **当前项目仍将“训练入口脚本”承担了过多算法表达责任，不利于后续做可审计实验矩阵。**

### 结论

从“能跑 PPO / E-PPO”角度看已基本完成；从“严格论文复现与算法审计”角度看仍需结构化整理。

---

## 5.8 实验层：3.1 / 3.2 / 3.3

### 当前状态

当前仓库已有：

- 训练脚本：`scripts/train_ppo_net3.py`
- 评估脚本：`scripts/evaluate_policy_net3.py`
- 泛化实验编排脚本：`scripts/run_reproduction_suite.py`

其中 `run_reproduction_suite.py` 已经能按 `algo / sigma / delta / seed` 组合批量训练评估，见 `scripts/run_reproduction_suite.py:95-182, 213-235`。

### 与目标方案的差距

这一层是**当前缺口最大的部分**：

1. **没有正式的 `run_penalty_study.py`**，因此论文 3.1 尚未落地。
2. **没有正式的 `run_entropy_study.py`**，因此论文 3.2 尚未落地。
3. **没有正式的 `run_model_compare.py`**，因此论文 3.3 尚未落地。
4. **没有 GA / PSO / DE 基线实现**，因此 model comparison 目前无法达到论文目标。
5. **当前批量脚本默认 `sigmas=[0.0, 0.2, 0.3]`，但论文 3.2 还需要系统性 sweep `0.0, 0.1, 0.2, 0.3, 0.5`，且分 `delta=0.3/0.6/0.9` 与 3 个 seed。**
   - 证据：`scripts/run_reproduction_suite.py:107-117`
6. **当前批量脚本默认 `eval_episodes=20`，与论文中 penalty study 的 100 个随机测试集、model comparison 的 15 个测试案例并不一致。**
   - 证据：`scripts/run_reproduction_suite.py:133-136`

### 结论

**实验层目前仍处于“工程自动化”阶段，尚未进入“论文实验流水线”阶段。**

---

## 5.9 文档与仓库认知一致性

### 已完成内容

仓库文档已经开始主动区分“论文明确给出 / 当前仓库实现 / 工程近似”：

- `docs/EPANET_RL_REPRO_GUIDE.md`
- `docs/CODEBASE_UNDERSTANDING_CN.md`

这对后续审计非常有帮助。

### 与目标方案的差距

1. **README 仍明显落后于代码现状。**
   `README.md` 仍写着“Project skeleton only / No PPO training implementation yet”，见 `README.md:1-30`，这与当前仓库已有的 WNTR 环境、PPO 训练和评估脚本明显不一致。
2. **尚无一份专门面向“论文严格复现 vs 当前工程实现差距”的计划文档。**
   本文正是在补这个空缺。
3. **尚未在文档中显式写明 330/335 论文记号冲突的解析原则。**

### 结论

文档层已有基础，但**仍缺一份统一的审计式计划文档和若干关键说明更新**。

---

## 6. 哪些内容可以判定为“已完成”

下面这些能力，当前仓库可以视为**已经完成或基本完成**：

### 6.1 已完成

- Net3 WNTR/EPANET 单步环境主线
- 24 步、1 小时步长、一日一集的 episode 语义
- 观测主线为 `demands + tank_levels`
- reset 时 24 小时 demand 轨迹生成
- tank 初值随机化
- 两阶段 demand randomization
- Net3 工况文本预处理主流程
- 奖励主公式与 hydraulic violation 终止逻辑
- PPO 主训练入口
- PPO/E-PPO 的统一评估入口
- 关键单元测试和环境合同测试

### 6.2 基本完成但需收口

- PPO 超参数论文对齐
- E-PPO 熵项增强
- 状态归一化接口
- Net3 工况修改器

这些内容不是从零开始重写，而是需要**从“可运行工程版”向“严格论文版”收口**。

---

## 7. 哪些内容明确“尚未完成”

以下内容仍可明确判定为未完成：

1. **严格论文版 49 动作空间**
2. **环境默认 benchmark 固定为 406.54**
3. **论文 6 种 tank penalty 及其正式实验**
4. **严格区分 PPO 与 E-PPO 的算法模块化实现**
5. **3.1 penalty study 正式脚本**
6. **3.2 entropy study 正式脚本**
7. **3.3 model comparison 正式脚本**
8. **GA / PSO / DE 基线实现**
9. **大用户节点固定名单与配置**
10. **严格论文模式与工程稳健模式的分离**
11. **README 与实际代码能力对齐**
12. **330/335 歧义解析说明写入 README/配置/实验说明**

---

## 8. 下一步修改优先级建议

如果目标是“论文复现优先”，建议严格按下面顺序推进，而不是先做更多工程增强。

## Phase A：先收缩到严格论文环境

这是最优先阶段，目标是把当前工程从“可运行近似”收缩为“论文严格模式”。

### A1. 修正动作空间为 49 个离散组合

优先级：**最高**

原因：当前 64 动作主线会直接改变 agent 的搜索空间、策略熵、训练难度和最终结论，是最能污染论文结论的差异。

应修改的内容：

- `src/epanet_rl/action_space.py`
- `src/epanet_rl/env.py`
- `src/epanet_rl/env_wntr.py`
- `tests/test_action_space.py`
- `tests/test_env_contract.py`
- `tests/test_env_wntr_contract.py`
- `tests/test_env_paper_requirements.py`
- `tests/test_env_wntr_paper_requirements.py`
- 所有文档中关于“64 动作 / 8 档 / 含 0.0”的描述

应达到的结果：

- 动作档位只保留 `0.70~1.00` 的七档；
- `action_space.n == 49`；
- 移除主线 fallback 对 `(0.0, 0.0)` 的依赖，必要时改为“仿真失败即 episode fail”或另设非主线 debug 模式。

### A2. 把严格论文模式下的默认 `r_benchmark` 固定为 406.54

优先级：**最高**

原因：reward 标尺会直接影响训练信号。

应修改的内容：

- `src/epanet_rl/env_wntr.py`
- `src/epanet_rl/env.py`
- 相关测试中的 placeholder 断言
- 文档中所有“当前仍是 2000.0 占位值”的表述

应达到的结果：

- 主线环境默认值与训练脚本默认值一致；
- 2000.0 只允许存在于历史说明或 debug 兼容层，不再作为主线默认值。

### A3. 明确 strict paper mode 与 engineering mode

优先级：**高**

原因：当前主线环境混入 fallback、停泵回退等工程增强，不利于审计。

建议做法：

- 在环境配置中新增 `strict_paper_mode=True/False`；
- strict 模式下关闭 `(0.0, 0.0)` fallback；
- 所有论文实验一律使用 strict 模式；
- 诊断脚本才允许启用工程增强。

---

## Phase B：补齐论文 3.1 的 penalty study

### B1. 将 reward 模块改造成论文 6 种 penalty 的正式接口

优先级：**最高（仅次于 A）**

建议：

- 在 reward 层引入明确的 `penalty1 ... penalty6` 枚举或命名函数；
- 保留当前通用公式能力，但不要再把论文主实验建模为“constant / proportional 任意切换”；
- 把 penalty3 固化为后续默认主线。

### B2. 新增正式实验入口 `run_penalty_study.py`

优先级：**高**

实验要求应对齐用户给出的方案：

- demand uncertainty 固定 `delta=0.3`
- 训练 6 个 penalty 版本
- 100 个随机测试集评估
- 输出平均能耗、tank volume 约束、成功率与结果表

---

## Phase C：补齐论文 3.2 的 entropy study

### C1. 显式区分 PPO 与 E-PPO 的算法表达

优先级：**高**

建议：

- 将当前训练脚本中的算法逻辑下沉为独立模块；
- 明确 `PPOAgent` 与 `EPPOAgent` 的唯一差异就是 policy entropy bonus；
- critic loss 保持一致。

### C2. 新增 `run_entropy_study.py`

优先级：**高**

实验矩阵建议固定为：

- `delta in [0.3, 0.6, 0.9]`
- `sigma in [0.0, 0.1, 0.2, 0.3, 0.5]`
- `seed in [1, 2, 3]`

当前 `run_reproduction_suite.py` 可作为实现基础，但不应继续作为论文正式实验脚本本身。

---

## Phase D：补齐论文 3.3 的 model comparison

### D1. 新增基线算法实现

优先级：**高**

缺失项：

- GA
- PSO
- DE

如果这三类基线不补齐，论文最关键的 model comparison 无法完成。

### D2. 新增 `run_model_compare.py`

优先级：**高**

要求：

- PPO / E-PPO 预训练后只做推理
- GA / PSO / DE 每个测试案例在线重新求解
- 汇总 15 个测试案例
- 输出 energy cost、推理时间、成功率、tank 末态等指标

---

## Phase E：补齐审计与文档

### E1. 更新 README

优先级：**中高**

原因：当前 README 仍宣称仓库只是 skeleton，这与实际工程能力不符，容易误导后续协作。

### E2. 在文档中明确记录 330/335 歧义解析

优先级：**中高**

建议固定写法：

- 受控泵解析为 10 和 335；
- 关闭管道为 330；
- 全局效率 0.75 当前按 10 / 335 工况解释；
- 这是对论文内部记号不一致的审计性解析。

### E3. 明确非随机化大用户节点名单

优先级：**中**

当前代码已有接口，但缺正式配置和说明。

---

## 9. 建议的“下一步实施清单”

如果只看近期两周内最值得推进的内容，建议按下面顺序实施：

1. **先修动作空间：64 -> 49**
2. **再修主线环境默认 `r_benchmark: 2000.0 -> 406.54`**
3. **从主线环境剥离 `(0.0, 0.0)` fallback，建立 strict paper mode**
4. **把 reward 层补成论文 6 种 penalty 形式**
5. **落地 `run_penalty_study.py`，先完成 3.1**
6. **随后再把 entropy study 做成正式脚本**
7. **最后再补 GA / PSO / DE 和 model comparison**

这是因为：

- **动作空间、reward 基准、strict mode** 决定“你的环境到底是不是论文环境”；
- **penalty study** 决定“你后面到底该固定哪种 tank penalty”；
- **entropy study** 决定“E-PPO 该用哪个 sigma”；
- **model comparison** 只能放在这些基础稳定之后。

---

## 10. 最终判断

综合当前代码和目标方案，仓库现状可以概括为：

> **环境主干、奖励骨架、Net3 工况修改、PPO/E-PPO 训练评估入口已经具备，说明项目已经越过“从 0 到 1”的阶段；但它仍停留在“工程可运行复现版”，尚未完全收缩为“严格论文版最小复现”，也尚未按论文 3.1 / 3.2 / 3.3 的顺序形成正式实验流水线。**

因此，下一步不建议继续增加停泵档、扩展状态、连续动作或更多工程特性；而应该集中做三件事：

1. **把当前实现收缩为严格论文版环境**；
2. **把论文实验顺序拆成正式脚本并逐阶段完成**；
3. **把所有论文歧义与工程近似显式记录成可审计文档。**

只有这样，后续得到的结果才更有资格被称为“论文复现结果”，而不是“参考论文思想做的一套工程系统”。
