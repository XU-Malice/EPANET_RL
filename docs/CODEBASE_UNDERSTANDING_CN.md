# EPANET_RL 仓库理解报告（中文教学版）

> 目标读者：几乎从 0 开始、希望先“看懂仓库再动手”的学习者。
>
> 本文只基于**当前仓库实际代码、脚本与测试**进行说明，不把“论文应该怎样”与“仓库已经怎样”混为一谈。

---

## 1. 先说结论：这个仓库现在到底在做什么？

这个仓库的核心目标，是围绕 **EPANET / WNTR 的 Net3 供水网络**，构建一个可用于强化学习（RL）泵调度实验的代码基础。

如果你只记住一句话，可以记成：

- **环境主线**：`src/epanet_rl/env_wntr.py` 中的 `Net3WntrEnv`
- **训练主入口**：`scripts/train_ppo_net3.py`
- **评估主入口**：`scripts/evaluate_policy_net3.py`
- **benchmark 统计入口**：`scripts/compute_r_benchmark.py`
- **z-score 统计入口**：`scripts/compute_zscore_stats.py`

这几个入口共同形成了一条比较完整的实验链：

1. 先把 `Net3.inp` 处理成适合 RL 控制的版本；
2. 在环境里按 24 个离散时间步（每步 1 小时）做顺序决策；
3. 用奖励函数引导策略学习“低能耗、少违例”的泵调度；
4. 用 benchmark 与 z-score 脚本分别提供奖励标尺和特征统计；
5. 用评估脚本对训练好的策略做统一统计；
6. 用 `tests/` 保证动作、奖励、需水随机化、INP 修改、环境合同等关键语义不被改坏。

---

## 2. 顶层目录结构怎么读？

下面按“学习优先级”介绍顶层目录。

### 2.1 `src/epanet_rl/`

这是**核心逻辑目录**，真正决定“环境如何运行、奖励怎么算、动作是什么、需求如何随机化”的地方都在这里。

建议阅读顺序：

1. `action_space.py`
2. `reward.py`
3. `demand_randomization.py`
4. `inp_modifier.py`
5. `env_wntr.py`
6. `scaling.py`
7. `env.py`（作为对照：它是简化环境，不是主线环境）

### 2.2 `scripts/`

这是**实验入口目录**。特点是：

- 负责“如何跑”；
- 不负责定义最底层环境逻辑；
- 适合当作“实验说明书”来读。

对于新手，建议先把这里当作“使用手册”：

- `train_ppo_net3.py`：训练
- `evaluate_policy_net3.py`：评估已训练模型
- `compute_r_benchmark.py`：算奖励中的参考能耗基准
- `compute_zscore_stats.py`：算 z-score 所需均值方差
- `diagnose_env_wntr_rollout.py`：调试 WNTR 主线环境
- `diagnose_env_rollout.py`：调试简化环境

### 2.3 `tests/`

这是**语义守护目录**。它不是“附属品”，而是帮助你理解仓库非常高价值的材料。

因为测试会明确写出：

- 作者认为哪些接口是必须稳定的；
- 哪些行为被视为论文要求；
- 哪些功能目前还是工程近似或占位实现；
- 哪些地方刻意保留 `xfail`，表示“暂未完全复现”。

### 2.4 `networks/`

存放 EPANET 网络文件：

- `Net3.inp`
- `Anytown.inp`

当前主线脚本几乎都围绕 `Net3.inp`。

### 2.5 `docs/`

当前已有一份 `EPANET_RL_REPRO_GUIDE.md`，它更像“复现实验说明”。

而本文 `CODEBASE_UNDERSTANDING_CN.md` 的定位，是更偏**代码结构理解与教学准备**。

---

## 3. 推荐的阅读路径（给从 0 开始的人）

如果你直接打开 `env_wntr.py`，很可能会觉得信息量过大。更合理的学习顺序是：

### 第 1 层：先弄清楚动作、奖励、随机化

- `action_space.py`：动作到底是什么
- `reward.py`：每一步 reward 怎么算
- `demand_randomization.py`：需求如何生成

### 第 2 层：再看环境骨架

- `env_wntr.py`：reset / step / observation / 单步仿真

### 第 3 层：再看实验入口

- `train_ppo_net3.py`
- `evaluate_policy_net3.py`
- `compute_r_benchmark.py`
- `compute_zscore_stats.py`

### 第 4 层：最后用测试反向验证理解

- `tests/test_action_space.py`
- `tests/test_reward.py`
- `tests/test_demand_randomization.py`
- `tests/test_inp_modifier_net3.py`
- `tests/test_env_wntr_contract.py`
- `tests/test_env_wntr_paper_requirements.py`

---

## 4. 重点文件逐个讲解

---

## 4.1 `src/epanet_rl/action_space.py`

### 这个文件是干什么的？

它定义了**离散动作空间**。

这里的动作不是“连续控制某台泵转速”，而是：

- 两台泵；
- 每台泵 8 个离散速度档位；
- 因此总动作数 = `8 × 8 = 64`。

### 代码里实际怎么定义？

速度档位固定为：

- `0.0`
- `0.7`
- `0.75`
- `0.8`
- `0.85`
- `0.9`
- `0.95`
- `1.0`

动作 id 的映射规则是：

- `pump1` 作为高位索引；
- `pump2` 作为低位索引；
- `action_id = pump1_index * 8 + pump2_index`

所以：

- `0` 对应 `(0.0, 0.0)`
- `63` 对应 `(1.0, 1.0)`

### 你学习时要抓住什么？

这个文件很小，但非常关键，因为它决定了：

- 环境 `action_space = Discrete(64)` 是怎么来的；
- 训练出来的策略输出的整数动作，如何被翻译成两台泵的速度；
- 日志里出现的动作号，如何人工还原。

### 论文 / 当前实现 / 工程近似

#### a. 论文明确给出的

- 两台泵、每台 8 档、共 64 个离散动作。

#### b. 当前仓库已有实现

- 用 `ACTION_LEVELS` 固定 8 档；
- 提供 `action_id_to_speeds()` 与 `speeds_to_action_id()` 的双向映射；
- 提供 `all_action_combinations()` 方便遍历全动作集。

#### c. 为工程可运行做的近似

- `speeds_to_action_id()` 在反查时容忍极小浮点误差，避免 `0.9` 与 `0.9000000001` 这类问题导致误判。

---

## 4.2 `src/epanet_rl/reward.py`

### 这个文件是干什么的？

它实现了**奖励函数**，而且是以**纯函数**的方式实现。

这意味着：

- 奖励逻辑不直接依赖环境内部状态；
- 更容易单元测试；
- 更容易审计“到底是哪个条件导致 reward 变成这样”。

### 奖励逻辑分哪几层？

#### 1）常规奖励

常规奖励公式是：

`r_benchmark / 24 - e_pump_t`

直观解释：

- `r_benchmark / 24` 像一个每步可获得的“基准奖励”；
- `e_pump_t` 是当前步泵能耗成本；
- 所以能耗越高，奖励越低。

#### 2）水力违例优先级最高

如果发生 `hydraulic_violation=True`，则：

- 直接给 `p_hydraulic`（一个大负值）；
- 且 `terminated=True`。

这意味着一旦出违例，不再考虑普通奖励或末步 tank penalty 的叠加抵消。

#### 3）末步水箱惩罚

只有在最后一步（默认第 23 步，0-based）才检查：

- 若最终总 tank volume 小于初始总 tank volume；
- 则施加 `tank penalty`。

支持两种模式：

- `constant`
- `proportional`

其中 proportional 模式按照“短缺比例 × `r_benchmark` × 系数”来算。

### 为什么这个文件重要？

因为训练最终优化的就是 reward，而 reward 是否合理，决定了策略会学成什么样。

比如：

- 如果只惩罚能耗，不惩罚末端 tank 亏空，策略可能会“省电但透支储水”；
- 如果 hydraulic penalty 不够重，策略可能会频繁冒险；
- 如果末步 penalty 太大，策略可能过度保守。

### 论文 / 当前实现 / 工程近似

#### a. 论文明确给出的

- 常规奖励核心形式：`r_benchmark / 24 - E_pump_t`；
- 水力违例要大惩罚并提前终止；
- tank penalty 在末步检查。

#### b. 当前仓库已有实现

- `compute_regular_reward()`：常规奖励；
- `compute_tank_penalty()`：末步 tank penalty；
- `compute_total_reward()`：统一封装优先级与终止语义。

#### c. 为工程可运行做的近似

- `TankPenaltyConfig` 暴露了 proportional / constant 两种模式，方便实验切换；
- 当 `initial_tank_volume <= 0` 时直接返回 0，防止除零或不合理放大。

---

## 4.3 `src/epanet_rl/demand_randomization.py`

### 这个文件是干什么的？

它负责**需求随机化**，也就是每个 episode 的用水需求不是固定不变，而是带扰动的。

这是 RL 训练里很重要的一环，因为：

- 如果需求永远固定，策略容易过拟合单一日型；
- 有随机化后，策略更可能学到鲁棒性更好的调度行为。

### 这个文件的核心思想是什么？

它把需求随机化拆成两层：

#### 1）时间乘子 `time_multipliers`

表示同一天 24 个时间步上，整体负荷怎么变。

#### 2）空间乘子 `space_multipliers`

表示不同节点之间，哪个节点偏高、哪个节点偏低。

#### 3）最终组合公式

最终某个时间步 `t`、某个节点 `i` 的 demand 由下面四部分相乘得到：

- 节点基础需水 `base_demands[i]`
- 原始模式 `default_pattern[t]`
- 时间随机乘子 `time_multipliers[t]`
- 空间随机乘子 `space_multipliers[i]`

也就是：

`demand[t, i] = base_demands[i] * default_pattern[t] * time_multipliers[t] * space_multipliers[i]`

### 采样怎么做？

文件中使用了**截断正态分布**采样，范围在：

- 时间乘子：`[1 - delta_time, 1 + delta_time]`
- 空间乘子：`[1 - delta_space, 1 + delta_space]`

### 这个文件里哪些函数最值得记？

- `sample_truncated_normal()`：底层采样器
- `generate_time_multipliers()`：生成时间乘子
- `generate_space_multipliers()`：生成空间乘子
- `compose_randomized_demands()`：按公式组装需求矩阵
- `generate_randomized_demands()`：一站式接口

### 你学习时要注意什么？

这个文件虽然不长，但直接决定：

- `reset()` 时每个 episode 的 demand 轨迹长什么样；
- max-min 缩放边界估计时如何理解随机范围；
- z-score 统计需要在什么分布下收集样本。

### 论文 / 当前实现 / 工程近似

#### a. 论文明确给出的

- 时间与空间随机乘子来自截断正态；
- 最终需水由 base demand、默认 pattern、时间乘子、空间乘子共同构成。

#### b. 当前仓库已有实现

- 所有随机性都显式依赖 `numpy.random.Generator`，可复现性较好；
- 支持 `randomizable_mask`，允许某些节点不参与随机化。

#### c. 为工程可运行做的近似

- 截断正态采用 rejection sampling；
- 如果多轮采样后仍有位置没填满，会用 `clip` 做兜底，优先保证函数稳定返回，而不是追求理论分布采样的最严格形式。

---

## 4.4 `src/epanet_rl/inp_modifier.py`

### 这个文件是干什么的？

它负责把原始的 `Net3.inp` 改造成更适合 RL 实验控制的版本。

你可以把它理解成：

- 原始 EPANET 网络文件里，可能包含一些原有控制规则；
- 但做 RL 时，我们希望“动作由智能体控制”，而不是继续被原来的控制逻辑干扰；
- 所以需要预处理 INP 文本。

### 它主要做了哪些修改？

#### 1）删除特定控制规则

删除与以下对象相关的控制条目：

- Link / Pump / Pipe 10
- 330
- 335

也包括一些相关注释描述，例如 lake source / bypass pipe 等。

#### 2）确保 Pipe 330 关闭

在 `[STATUS]` section 中确保：

- `330 Closed`
- 且只保留一条，不重复。

#### 3）重写 `[ENERGY]`

重设全局能源相关参数：

- 全局效率 `75`（表示 0.75）
- 全局谷价 `OFFPEAK_PRICE_USD_PER_KWH`
- 全局价格模式 `TOU_PATTERN_ID`
- 保留 / 延续 demand charge

#### 4）插入或更新 TOU 电价模式

在 `[PATTERNS]` 里插入一个新的 TOU pattern：

- 07:00–23:00 为峰价；
- 其他时段为谷价；
- 倍率通过 `peak/offpeak` 比值计算。

### 为什么这个文件非常重要？

因为 WNTR 主环境在初始化时并不是直接拿原始 `Net3.inp` 去跑，而是：

1. 读原始 INP 文本；
2. 用 `modify_inp_text()` 改写；
3. 写入临时文件；
4. 再用这个“修改版 INP”做仿真。

也就是说，这个文件其实决定了**环境仿真的基础工况**。

### 论文 / 当前实现 / 工程近似

#### a. 论文明确给出的

- Net3 要做控制规则、电价、效率、Pipe 330 等特定工况修改。

#### b. 当前仓库已有实现

- 用纯文本方式改写 INP；
- 提供 `modify_inp_text()` 与 `modify_inp_file()`；
- 还带了 CLI，可以独立运行。

#### c. 为工程可运行做的近似

- 当前做法是文本级 section 改写，而不是在 EPANET API 层逐条建模；
- 它能稳定审计与测试，但是否与论文原始工程文件“逐行完全一致”，仍应通过实验结果进一步核验。

---

## 4.5 `src/epanet_rl/env_wntr.py`

> 这是整个仓库最重要的文件。

### 这个文件是干什么的？

它实现了主线强化学习环境 `Net3WntrEnv`。

和简化环境 `env.py` 不同，这里每个 step 都会：

- 构造一个 WNTR `WaterNetworkModel`；
- 应用当前小时的 demand；
- 应用当前动作对应的两台泵速度；
- 跑一次 EPANET 仿真；
- 从结果里提取 tank、压力、泵流量、扬程、能耗等；
- 再通过 `reward.py` 算出 reward。

### 你要把这个环境理解成什么？

可以把它理解成下面这个流程：

#### reset 阶段

- 采样 24 小时 demand 轨迹；
- 随机化初始 tank levels；
- 生成初始 observation；
- 返回一些审计信息（乘子、泵名、水箱名）。

#### step 阶段

输入是离散动作 `0..63`，输出是 Gymnasium 标准五元组：

- `obs`
- `reward`
- `terminated`
- `truncated`
- `info`

中间具体做的事包括：

1. 把 `action_id` 解码成两台泵速度；
2. 对 tank level 做数值稳定性裁剪；
3. 调 `_simulate_single_step()` 跑真实单步仿真；
4. 若原动作导致异常 / NaN，按配置可回退到 `(0,0)` 动作；
5. 从仿真结果中提取：
   - `tank_levels`
   - `pump_flows`
   - `pump_head_gains`
   - `min_pressure`
   - `pump_energy_cost`
6. 构造 `StepRewardInput`，调用 `compute_total_reward()`；
7. 根据是否违例或到达 24 步决定 `terminated/truncated`；
8. 在 `info` 中塞入大量诊断信息。

### `_simulate_single_step()` 是环境的水力核心

这个函数做的事情，可以粗略理解为：

1. 从修改后的 INP 文件重新创建一个 WNTR 网络对象；
2. 设置单步仿真时长为 1 小时；
3. 写入本步 tank 初始液位；
4. 写入本步 demand；
5. 写入两台泵速度；
6. 调 `wntr.sim.EpanetSimulator(wn).run_sim()`；
7. 从结果里提取：
   - 末时刻 tank level
   - 末时刻与全步最小压力
   - 泵流量和扬程
   - 单步电费估算

### 这个环境里的 observation 是什么？

状态向量是：

- 当前小时的所有 junction demand
- 当前 3 个 tank levels

拼接而成。

**注意：这里不把压力、泵流量等仿真结果放进状态。**

这点很关键，因为它体现了当前代码对“状态定义”的选择：

- 状态 = demand + tank levels
- 而不是把更多水力变量都塞进去

### 这个环境里的动作是如何生效的？

动作先经 `action_id_to_speeds()` 变成两台泵速度，然后在 `_apply_pump_speeds()` 里写入：

- `pump.base_speed`
- `pump.speed_pattern_name = None`
- `pump.initial_status = OPEN/CLOSED`

也就是：

- 速度为 0 → 泵关闭
- 速度大于 0 → 泵打开

### 这个环境里的能耗是如何估算的？

当前实现按 `Q/H/η` 做单步电费估算：

- 用流量、扬程、密度、重力、效率计算功率；
- 再折算为 1 小时能耗；
- 再按当前小时是峰价还是谷价换算成成本。

### 当前环境里最值得注意的工程保护逻辑

这是本仓库最容易被忽略、但对“为什么它能跑”非常关键的部分。

#### 1）tank level 边界裁剪

在仿真前，会把 tank levels 向区间内部轻微推开一个 `epsilon`，避免“刚好卡在上下边界”引发数值不稳定。

#### 2）NaN fallback

如果原动作仿真结果出现 NaN，且 `enable_nan_fallback=True`：

- 会尝试把动作替换成 `(0.0, 0.0)` 再跑一次；
- 若 fallback 也失败，则记为仿真失败并当作 hydraulic violation。

#### 3）仿真失败统一当作水力违例

这是很典型的工程选择：

- 不让训练流程因为一次仿真异常而直接崩掉；
- 而是把它视为一个不可接受动作，给惩罚并终止 episode。

### `info` 为什么很重要？

这个环境的 `info` 不只是附带信息，而是很多脚本的**数据来源**。

比如：

- benchmark 脚本需要 `info["e_pump_t"]`
- 评估脚本要用 `base_reward` / `tank_penalty` / `initial_tank_volume` / `final_tank_volume`
- 诊断脚本要看 `pump_flows` / `pump_head_gains` / `min_pressure`
- fallback 分组统计要看 `fallback_used`

所以在这个仓库里，`info` 可以看成“实验审计接口”。

### 论文 / 当前实现 / 工程近似

#### a. 论文明确给出的

- episode 长度 24；
- 每步 1 小时；
- 状态为 demand + tank levels；
- 水力违例给大惩罚并提前终止；
- 使用真实水力仿真作为环境基础。

#### b. 当前仓库已有实现

- 真的调用了 WNTR / EPANET 做单步仿真；
- `reset()` 会重采样 demand，并随机初始 tank level；
- `step()` 通过 `reward.py` 统一处理 reward 与终止；
- `info` 字段很丰富，支持训练、评估、benchmark、诊断共享。

#### c. 为工程可运行做的近似

- tank level 的 `epsilon` 裁剪；
- 原动作异常时的 fallback 到 `(0,0)`；
- 当前能耗口径采用 **single_point** 估算，而不是更复杂的时间积分；
- 当前违例终止规则依然以“末时刻压力规则”为主，而不是把 full-step 规则直接接入终止决策；
- 仿真异常被视为 hydraulic violation，这是一种训练友好的鲁棒化处理。

---

## 4.6 `src/epanet_rl/scaling.py`

### 这个文件是干什么的？

它实现状态特征缩放，主要有两类：

- `max_min`
- `z_score`

而且分成两组接口：

- demand 的缩放
- tank level 的缩放

### 为什么这个文件独立出来？

因为缩放最好保持为纯函数：

- 便于测试；
- 便于环境重用；
- 便于单独替换策略。

### 这里有哪些工程保护？

- 如果 max-min 的上下界几乎相等，会给一个稳定值，而不是除零；
- 如果 z-score 的标准差接近 0，会返回指定的 `zero_std_value`，避免爆炸。

### 论文 / 当前实现 / 工程近似

#### a. 论文明确给出的

- 需要支持 ESFC-Max-Min 与 ESFC-Z-Score 这类状态缩放思想。

#### b. 当前仓库已有实现

- 已有稳定的 max-min / z-score 纯函数实现；
- `env.py` 与 `env_wntr.py` 都能调用。

#### c. 为工程可运行做的近似

- 对零方差、相等边界做了稳定性保护；
- `env_wntr.py` 默认 z-score 统计量目前并不是通过大规模离线校准直接注入，而是先用范围估计近似，后续再用统计脚本补充更真实的统计量。

---

## 4.7 `src/epanet_rl/env.py`

### 这个文件是什么定位？

它是一个**简化环境**，不是当前论文复现主线。

### 为什么仓库里还保留它？

因为它有几个工程价值：

- 在没装好 WNTR / EPANET 时，帮助先打通 RL pipeline；
- 更适合做 smoke test；
- 能较快验证动作、reward、reset/step 接口是否一致。

### 它和 `env_wntr.py` 的根本区别

- `env.py`：内部状态转移是简化的、近似的，并不调用真实单步水力仿真；
- `env_wntr.py`：每步调用真实 WNTR / EPANET 仿真，是主线环境。

### 教学建议

如果你是第一次接触这个项目：

- 可以先看 `env.py` 理解接口；
- 但真正理解论文复现，一定要回到 `env_wntr.py`。

---

## 5. 训练、评估、benchmark、zscore 的入口分别是什么？

这一节非常重要，因为很多人读仓库时会混淆“哪个脚本是干什么的”。

---

## 5.1 训练入口：`scripts/train_ppo_net3.py`

### 作用

这是**主训练入口**，用于在 `Net3WntrEnv` 上训练 PPO 或 E-PPO。

### 训练脚本做了哪些关键事情？

#### 1）解析实验参数

包括：

- 算法类型：`ppo` / `eppo`
- `sigma`
- `delta_time` / `delta_space`
- `scaling_mode`
- `r_benchmark`
- `p_hydraulic`
- PPO 超参数（lr、gamma、clip、epochs、batch size 等）

#### 2）把 PPO 与 E-PPO 的差别映射到 `ent_coef`

代码里的核心逻辑是：

- `ppo` → `ent_coef = 0`
- `eppo` → `ent_coef = sigma`

所以当前仓库并没有单独实现一个全新的 E-PPO 优化器，而是把它落到 SB3 PPO 的 entropy bonus 系数上。

#### 3）固定使用主线环境 `Net3WntrEnv`

训练环境不是简化环境，而是：

- `Net3WntrEnv`
- 并强制检查 `EPISODE_STEPS == 24`

#### 4）自定义 Policy 和 PPO 类

这是训练脚本里比较“工程化”的地方。

它定义了：

- `PaperLikeActorCriticPolicy`
- `PaperLikePPO`

目的不是重写 PPO 算法本体，而是为了保证：

- actor / critic 可以使用**分离学习率**；
- 且这个分离学习率在 SB3 的训练过程中不会被统一覆盖掉。

#### 5）网络结构按论文对齐

脚本里显式设置：

- actor: `[256, 128, 64]`
- critic: `[256, 128, 1]`

在 SB3 里实际写成：

- `pi = [256, 128, 64]`
- `vf = [256, 128]`

因为 value head 的最终单输出层由 SB3 自己补上。

#### 6）输出实验审计文件

训练结束后会写：

- 模型文件
- `train_config.json`
- `train_summary.json`
- `optimizer_lrs.json`
- SB3 日志与 monitor 文件

### 对学习者最关键的理解

这个训练脚本不是“随便调个 PPO”，而是在尽量做三件事：

1. 尽量贴近论文的超参数和网络结构；
2. 尽量复用成熟的 SB3 训练框架；
3. 尽量增加实验审计能力，方便回头核对到底用了什么设置。

### 论文 / 当前实现 / 工程近似

#### a. 论文明确给出的

- PPO 相关网络结构与主要超参数；
- PPO 与 E-PPO 的区别应体现在探索 / 熵项上。

#### b. 当前仓库已有实现

- 主线训练脚本完整可运行；
- 使用 SB3 PPO；
- 用自定义策略类和 PPO 子类保证分离学习率；
- 训练过程可记录丰富日志与 JSON 摘要。

#### c. 为工程可运行做的近似

- E-PPO 不是重写论文原版算法，而是通过 `ent_coef = sigma` 映射到 SB3 PPO；
- 使用 `DummyVecEnv` + `VecMonitor` 的单环境封装，这是常见工程实现方式，不等价于论文源码逐行复刻；
- 进度反馈、JSON 审计、非交互日志输出等，都是工程可用性增强。

---

## 5.2 评估入口：`scripts/evaluate_policy_net3.py`

### 作用

这是**统一策略评估脚本**，用于评估已经训练好的 PPO / E-PPO 模型。

### 它评估什么？

它不是只看一个总 reward，而是同时统计：

- `total_reward`
- `total_energy_cost`
- `total_base_reward`
- `total_tank_penalty`
- `episode_length`
- `hydraulic_violation`
- `full_horizon`
- `initial_tank_volume`
- `final_tank_volume`
- `volume_change_ratio`

### 为什么这样设计？

因为只看总 reward 不够。

比如一个策略 reward 还行，但可能是通过：

- 减少储水量换来的；
- 或者经常差一点就违例；
- 或者末步 tank penalty 特别大。

这个脚本把这些因素拆开后，才更容易理解策略到底学到了什么。

### 这个脚本的工作流程

1. 加载模型；
2. 构建 `Net3WntrEnv`；
3. 每个 episode 用 `deterministic=True` 做评估；
4. 从每步 `info` 里累计能耗、reward 分解、tank volume 信息；
5. 最后计算聚合统计并打印 / 存 JSON。

### 论文 / 当前实现 / 工程近似

#### a. 论文明确给出的

- 需要比较 PPO 与 E-PPO 等不同策略在相同环境下的表现。

#### b. 当前仓库已有实现

- 已有统一评估入口；
- 能同时给出 reward、能耗、违例率、全程率、末端储量变化等统计。

#### c. 为工程可运行做的近似

- 当前评估逻辑强调“可解释统计”，比只复刻论文表格更偏工程分析；
- 评估本身不改变环境规则，只负责观测和汇总。

---

## 5.3 benchmark 入口：`scripts/compute_r_benchmark.py`

### 作用

它用于通过**随机策略 rollout** 统计 `r_benchmark` 的候选值。

### 为什么需要这个脚本？

因为奖励里的常规项是：

`r_benchmark / 24 - e_pump_t`

如果 `r_benchmark` 取值不合理：

- 奖励尺度会失真；
- 不同步骤之间的激励强弱会改变；
- 训练稳定性和论文对齐性都会受影响。

所以 `r_benchmark` 最好不是拍脑袋，而是通过 rollout 统计得到。

### 这个脚本统计哪些口径？

它不会只给一个数字，而是给多种候选：

- `candidate_benchmark_all`
- `candidate_benchmark_successful`
- `candidate_benchmark_full_horizon`
- `candidate_benchmark_successful_non_fallback`
- `candidate_benchmark_full_horizon_non_fallback`

### 为什么会有这么多口径？

因为你可能会问：

- 要不要把违例 episode 算进去？
- 要不要排除 fallback 发生过的 episode？
- 要不要只统计完整 24 步的 episode？

仓库作者当前没有把这个问题强行定死，而是把多种口径都输出，方便你自己与论文或实验目标对齐。

### 论文 / 当前实现 / 工程近似

#### a. 论文明确给出的

- `r_benchmark` 应来自 rollout 统计，而不是随意手写。

#### b. 当前仓库已有实现

- 已有随机策略 rollout 统计脚本；
- 逐 episode 累计 `info["e_pump_t"]` 作为总能耗成本；
- 提供丰富分组统计。

#### c. 为工程可运行做的近似

- 哪一种 candidate 最接近论文最终口径，当前仍需研究者根据实验设定自行选择；
- fallback 分组是工程调试需求，并非论文必须显式给出的概念。

---

## 5.4 z-score 入口：`scripts/compute_zscore_stats.py`

### 作用

它用于统计 Z-Score 缩放所需的：

- `demand_mean`
- `demand_std`
- `tank_mean`
- `tank_std`

### 为什么不能直接手算一个均值方差？

因为环境状态来自：

- demand 随机化；
- tank 状态演化；
- 不同动作与不同 episode 长度；
- 真实仿真过程。

所以比较可靠的办法是：

- 跑很多 episode；
- 收集真实观测样本；
- 统计均值和标准差。

### 这个脚本的一个关键点

它在统计时会强制：

- `scaling_mode="none"`

也就是说，必须对**未缩放的原始状态**做统计，避免“先缩放再统计，统计结果又拿去缩放”的口径错误。

### 它如何处理大样本？

脚本使用 `_RunningFeatureStats` 做**在线均值 / 方差统计**，而不是把所有状态样本一次性堆进内存。

这意味着：

- 更省内存；
- 更适合长 rollout 或大量 episode。

### 论文 / 当前实现 / 工程近似

#### a. 论文明确给出的

- Z-Score 需要先估计状态特征的 mean/std。

#### b. 当前仓库已有实现

- 已有完整脚本统计 demand 和 tank 的均值方差；
- 使用在线算法流式累积统计量。

#### c. 为工程可运行做的近似

- 统计精度依赖你跑多少个 episode；
- 如果只跑少量样本，它只能算“小样本近似统计”，不是论文大规模统计的严格替代。

---

## 6. 测试目录 `tests/` 应该怎么读？

很多人读仓库时忽视测试，这是很可惜的。这个仓库的测试实际上非常适合拿来教学。

下面按重要性介绍。

---

## 6.1 `tests/test_action_space.py`

### 它验证什么？

- 动作档位是否与预期一致；
- 总动作数是否为 64；
- 所有动作能否 round-trip（id → speeds → id）；
- 非法动作 / 非法速度是否会报错。

### 它对理解仓库有什么帮助？

帮助你确认：

- 动作定义是固定的，不是“运行时动态生成”；
- 映射规则是一个强合同，后续不能随便改。

---

## 6.2 `tests/test_reward.py`

### 它验证什么？

- 常规奖励公式；
- hydraulic violation 是否覆盖其他奖励；
- constant / proportional tank penalty 是否正确；
- 末步 reward 是否按“base_reward + tank_penalty”组合。

### 它对理解仓库有什么帮助？

帮助你非常明确地知道：

- reward 优先级是什么；
- tank penalty 只在末步出现；
- `p_hydraulic` 是硬覆盖，而不是加到 base reward 上。

---

## 6.3 `tests/test_demand_randomization.py`

### 它验证什么？

- 截断正态样本是否落在合法范围；
- `delta=0` 时是否全为 1；
- 时间 / 空间乘子形状是否正确；
- 非随机化 mask 是否生效；
- 最终 demand 组合公式是否正确。

### 它对理解仓库有什么帮助？

帮助你确认：

- 需求随机化不是随意写的，而是有明确数学结构；
- `compose_randomized_demands()` 是这个模块最核心的函数之一。

---

## 6.4 `tests/test_inp_modifier_net3.py`

### 它验证什么？

- 目标控制规则是否真的被移除；
- Pipe 330 是否只保留一个 `Closed` 状态；
- `[ENERGY]` 是否正确写入效率、价格、pattern；
- TOU pattern 是否有 24 个值且峰谷时段正确；
- 默认 demand pattern 引用是否仍保留。

### 它对理解仓库有什么帮助？

它说明：

- `inp_modifier.py` 不是“随便改改文本”；
- 这些修改被当作实验前提条件的一部分，而且是被测试保护的。

---

## 6.5 `tests/test_env_wntr_contract.py`

### 它验证什么？

这是非常关键的一组“环境合同测试”。主要检查：

- tank 处于上界附近时，不会立刻产生 NaN 输出；
- `p_hydraulic` 非负会被拒绝；
- 发生水力违例时，是否立即终止并返回惩罚；
- 传给求解器的 tank level 是否低于上界一个 `epsilon`；
- 正常情况下执行动作是否就是原动作；
- fallback 是否只在原动作失败时才启用。

### 它对理解仓库有什么帮助？

这个文件几乎把 `env_wntr.py` 的工程稳定性策略都点明了。

也就是说，如果你想理解：

- 为什么要做 tank clipping；
- 为什么要有 fallback；
- 仿真异常时环境怎么处理；

这个测试文件特别值得细读。

---

## 6.6 `tests/test_env_wntr_paper_requirements.py`

### 它验证什么？

这是最接近“论文对齐检查清单”的测试之一。主要检查：

- 24 步、1 小时步长；
- 状态是不是 demand + tank level；
- 动作空间是不是 64 个离散动作；
- reset 是否随机化 demand 和初始 tank；
- step 是否真的调用 EPANET 模拟器；
- `info` 是否包含必须的水力输出；
- 固定动作 rollout 至少能跑多个有限步而不立刻 NaN 崩溃。

### 它还显式标出了什么“还没完全做完”？

它包含 `xfail` 项，比如：

- 基于大规模 episode 的 z-score 校准仍待完善。

### 它对理解仓库有什么帮助？

这个测试文件最能帮助你区分：

- 什么是仓库作者自认为“已经做到”的论文核心要求；
- 什么是“接口先有了，但还没完全校准”的部分。

---

## 6.7 `tests/test_env_contract.py` 与 `tests/test_env_paper_requirements.py`

### 它们是给谁看的？

主要是给简化环境 `env.py` 用的。

### 为什么仍值得看？

因为它们能帮你理解：

- 简化环境与主线环境的接口是一致的；
- 哪些论文要求在简化环境里只是“占位满足”；
- 哪些功能在 `env.py` 上仍刻意标记为 `xfail`。

特别是 `test_env_paper_requirements.py` 明确写出了：

- 当前 `env.py` 是 minimal runnable skeleton；
- 不是完整的 WNTR/EPANET 单步仿真环境；
- 大规模 z-score 校准也还不是它的职责。

这对避免误把 `env.py` 当主线环境非常重要。

---

## 6.8 `tests/test_scaling.py` 与 `tests/test_import.py`

### 作用

- `test_scaling.py`：验证缩放公式与边界稳定性；
- `test_import.py`：验证包可导入、基本路径常量存在。

### 教学价值

这两组测试不“耀眼”，但它们构成了仓库最底层的稳定基础。

---

## 7. 论文要求、当前实现、工程近似：应该怎样明确区分？

这是本报告最重要的一节。

很多复现项目都会出现一个问题：

- 把“论文说的”
- “代码现在已经实现的”
- “为了先跑起来做的工程近似”

混在一起。

这个仓库其实已经通过代码注释、测试、脚本说明，把三者分得比较清楚。下面我将其整理成更系统的版本。

---

## 7.1 A 类：论文明确给出的内容

根据当前仓库中的代码注释、测试与复现说明，论文主线明确强调的内容至少包括：

### 1）时序决策结构

- 一个 episode 为 24 步；
- 每步 1 小时。

### 2）状态定义

- 状态由 demand 与 tank levels 组成。

### 3）动作定义

- 两台泵；
- 每台泵 8 档速度；
- 共 64 个离散动作。

### 4）奖励主结构

- 常规奖励基于 `r_benchmark / 24 - E_pump_t`；
- 水力违例给大负惩罚并提前终止；
- 末步检查 tank volume 并施加 tank penalty。

### 5）需求随机化思路

- 时间乘子与空间乘子；
- 截断正态；
- demand 由多因子组合。

### 6）Net3 工况预处理

- 原控制规则、Pipe330、效率、电价、pattern 等需要调整。

### 7）训练超参数主线

- PPO / E-PPO 的网络结构与关键超参有明确目标值。

---

## 7.2 B 类：当前仓库已经有的实现

下面这些内容，当前仓库中已经有明确代码实现，不是空壳：

### 1）主线真实环境

- `Net3WntrEnv` 会真的调用 WNTR / EPANET 单步仿真。

### 2）动作空间模块

- 64 个离散动作的双向映射是完整可用的。

### 3）奖励函数模块

- 常规奖励、tank penalty、违例优先级都已纯函数化。

### 4）需求随机化模块

- 时间乘子、空间乘子、需求组合公式都完整实现。

### 5）INP 修改器

- 能以文本方式稳定改写 Net3 的关键配置。

### 6）训练脚本

- 已有可运行的 PPO / E-PPO 训练入口；
- 已对网络结构、分离学习率、日志审计做工程实现。

### 7）评估脚本

- 已能加载模型并跑多 episode 聚合统计。

### 8）benchmark / zscore 脚本

- 已能通过随机策略 rollout 输出统计结果。

### 9）测试体系

- 已对动作、奖励、需求随机化、INP 改写、主线环境合同进行了较系统的保护。

---

## 7.3 C 类：为了工程可运行做的近似或增强

这是你后续如果要做“论文严格复现”时最需要盯住的部分。

### 1）NaN fallback

当原动作仿真输出异常时，会尝试回退到 `(0,0)` 动作再跑。

这显然是一个工程稳定性策略，不太像论文会单独强调的算法核心。

### 2）tank level `epsilon` 裁剪

仿真前把液位轻微拉离边界，以减少求解器数值问题。

这是非常典型的工程近似。

### 3）single-point 能耗口径

当前 `_simulate_single_step()` 里能耗成本采用单点估算，而不是更复杂的时间积分方案。

这在工程上更稳，但它未必是论文最严格的能耗积分口径。

### 4）E-PPO 到 `ent_coef` 的映射

当前实现把 E-PPO 的差异通过 SB3 PPO 的 entropy coefficient 表达。

这是一种合理的工程映射，但不能自动等同于“完全复刻论文原始 E-PPO 代码”。

### 5）z-score 的两阶段现实

当前环境里有 z-score 接口，也有专门统计脚本；
但“大规模、严格校准后的统计量”仍需要额外运行统计脚本来补足。

### 6）文本改写 INP 而非复刻原始实验工况文件

目前做法适合测试和审计，但仍应通过最终实验结果验证是否与论文原工况严格一致。

### 7）简化环境 `env.py`

它是为了工程联调和 smoke test 而存在，不应被误认为论文主线实现。

---

## 8. 如果你现在要“教别人这个仓库”，最好的讲法是什么？

如果你要向一个初学者讲解这个项目，我建议按下面的教学顺序讲。

### 第一步：先讲任务本身

“这个仓库是在做供水系统泵调度的强化学习。一天分成 24 个小时，每小时决定两台泵的速度档位，希望既省电，又不要造成水力违例，还要保证最后储水别亏太多。”

### 第二步：讲状态、动作、奖励

- 状态：当前 demand + 当前 tank levels
- 动作：两台泵的 64 个离散组合
- 奖励：基准奖励减能耗，违例重罚，末步检查储水

### 第三步：讲环境如何“落地”

“环境不是凭空模拟，而是每一步都调用 WNTR / EPANET 做真实水力仿真。”

### 第四步：讲为什么需要 INP 修改

“因为 RL 想自己控制泵，所以要先把原始 INP 里干扰控制的规则清掉，并统一能源价格和效率设定。”

### 第五步：讲训练链条

- benchmark 脚本给 `r_benchmark`
- zscore 脚本给均值方差
- train 脚本训练 PPO / E-PPO
- evaluate 脚本统一评估

### 第六步：讲工程现实

“为了让仿真更稳，仓库里加入了 fallback、tank 边界裁剪、单点能耗估算等工程保护。这些不是坏事，但要和论文原始设定区分开。”

---

## 9. 对学习者来说，现阶段最容易混淆的几个点

### 混淆点 1：`env.py` 和 `env_wntr.py` 谁才是主线？

答案：

- `env_wntr.py` 才是主线；
- `env.py` 是简化版、联调用。

### 混淆点 2：`r_benchmark` 是不是已经最终确定？

答案：

- 训练脚本默认用 `406.54`；
- 但环境类默认仍常见 `2000.0` 占位值；
- 真正要论文对齐，应该通过 benchmark 脚本和实验设定进一步核定。

### 混淆点 3：仓库是否已经“完全等于论文实现”？

答案：

- 不是；
- 它已经实现了很多核心逻辑；
- 但也明确保留了若干工程近似与待校准部分。

### 混淆点 4：E-PPO 是否有独立算法实现？

答案：

- 当前更多是通过 SB3 PPO 的 `ent_coef` 做映射；
- 因此它是“工程上的 E-PPO 对应实现”，不是自动等于论文源码级复刻。

### 混淆点 5：z-score 是否已经完全准备好？

答案：

- 接口和统计脚本都有；
- 但严格的大样本统计仍需要你实际运行统计脚本来生成。

---

## 10. 一页式总表：你应该记住什么

| 主题 | 你最该记住的结论 |
|---|---|
| 主线环境 | `src/epanet_rl/env_wntr.py` 的 `Net3WntrEnv` |
| 动作空间 | 两台泵 × 每台 8 档 = 64 个离散动作 |
| 状态 | 当前 demand + 当前 tank levels |
| 奖励 | `r_benchmark/24 - e_pump_t`，违例大罚，末步 tank penalty |
| 需求随机化 | 时间乘子 × 空间乘子 × 默认 pattern × base demand |
| INP 预处理 | 删除控制、关闭 Pipe330、设置效率与 TOU 电价 |
| 训练入口 | `scripts/train_ppo_net3.py` |
| 评估入口 | `scripts/evaluate_policy_net3.py` |
| benchmark 入口 | `scripts/compute_r_benchmark.py` |
| z-score 入口 | `scripts/compute_zscore_stats.py` |
| 最关键测试 | `test_env_wntr_contract.py`、`test_env_wntr_paper_requirements.py` |
| 最大工程近似 | fallback、tank 边界裁剪、single-point 能耗估算、E-PPO 到 `ent_coef` 的映射 |

---

## 11. 下一步学习建议（在“不改源码”的前提下）

如果你现在还不准备改代码，只想继续做“教学准备”，建议按这个顺序继续：

### 路线 A：面向算法理解

1. 手算几个动作 id 到速度组合的映射；
2. 手算几个 reward 例子；
3. 读 `train_ppo_net3.py` 里的 PPO 超参数与网络结构。

### 路线 B：面向环境理解

1. 从 `env_wntr.reset()` 开始；
2. 再读 `step()`；
3. 再读 `_simulate_single_step()`；
4. 最后对照 `tests/test_env_wntr_contract.py` 看数值稳定性分支。

### 路线 C：面向实验复现

1. 先跑测试；
2. 再跑 `compute_r_benchmark.py`；
3. 再跑 `compute_zscore_stats.py`；
4. 再做短步数 PPO smoke training；
5. 最后用 `evaluate_policy_net3.py` 做统一评估。

---

## 12. 最后的整体判断

如果只从当前仓库状态出发，我会这样评价它：

### 它已经具备的优点

- 结构清晰；
- 主线环境、奖励、动作、随机化、实验入口都分模块实现；
- 测试覆盖面不错；
- 明确区分了“简化环境”和“真实仿真环境”；
- 训练 / 评估 / benchmark / zscore 的实验链已经形成。

### 它仍需要继续澄清或强化的地方

- 某些论文口径与当前工程实现之间还存在“映射而非严格等价”；
- `r_benchmark` 最终口径需要结合实际统计结果确认；
- z-score 的严格大样本统计仍需实践运行；
- 能耗积分与违例判定是否完全与论文口径一致，还值得进一步核查。

### 但作为“教学准备材料”，它已经很适合做什么？

非常适合用来做：

- 代码走读；
- 环境建模教学；
- RL + 水力仿真的结合示例；
- 论文复现中的“实现 vs 近似”辨析训练。

---

如果后续你要继续，我建议下一步不是立刻改功能，而是继续产出两份补充材料：

1. **`docs/ENV_STEP_FLOW_CN.md`**：把 `reset/step/_simulate_single_step` 画成流程图式说明。
2. **`docs/PAPER_VS_CODE_GAP_CN.md`**：把“论文目标、当前实现、待验证差异”做成逐条核对表。

这样后面再改代码，会更稳，也更不容易把“工程修复”和“论文对齐”混在一起。
