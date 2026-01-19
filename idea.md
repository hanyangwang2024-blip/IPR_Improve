# Graph-IPR: Learning Large Language Model Agents via Dynamic State-Action Graph Refinement

*(Graph-IPR：通过动态状态-动作图优化进行 LLM 智能体学习)*

---

## 1. 完善后的故事线 (The Story)

### 现状 (The Gap)

现有的 IPR 框架像是一个"复读机"，它在专家走过的每一个路口（State）停下来，尝试做一些不同的动作，然后通过采样看能不能回到终点。这种方法有两个致命伤：

1. **知识孤岛 (Isolated Experience)：** 第 1 步探索产生的失败经验，无法被第 10 步利用；第 1 轮迭代产生的轨迹，在第 2 轮被弃之如敝履。每次采样都是"用完即弃"，造成巨大的样本浪费。

2. **专家天花板 (Expert Ceiling)：** 它总是试图让智能体"表现得像专家"，如果专家绕路了，智能体也会被教导去绕路，因为它缺乏全局视野来对比不同路径。这导致模型永远无法超越次优专家。

### 我们的突破 (Our Insight)

智能体探索的本质是在**状态空间**中寻找路径。如果我们把所有迭代、所有采样产生的 `(s, a, s', r)` 碎片拼接成一张图，智能体就不再是盲目探索，而是在一张"逐渐清晰的地图"上进行规划。

**关键洞察**：即使某次探索最终失败了，它走过的其中一段路可能正好连接了两个关键节点，这条"桥梁"对于优化全局策略至关重要。这正是 Offline RL 中 **Trajectory Stitching** 思想在 LLM Agent 领域的首次系统性应用。

### 与 Process Reward Model 的联系

Graph-IPR 可以被视为一种 **Implicit Process Reward Model (Implicit PRM)**：
- 传统 PRM 需要人工标注每一步的奖励信号，成本高昂
- Graph-IPR 通过图结构自动传播和推断过程奖励，无需额外标注
- 图上的 Value Propagation 本质上是在学习一个 state-dependent 的 reward estimator

---

## 2. 核心技术细节 (Technical Deep-Dive)

### A. 状态对齐与图构建 (State Alignment & Graph Construction)

在 LLM Agent 任务中，状态 $s$ 通常由复杂的文本 Observation 和 History 组成。

#### A.1 软状态抽象 (Soft State Abstraction)

**放弃硬聚类，采用软对齐**：硬聚类面临阈值敏感问题，我们采用两种可选策略：

1. **LLM-based Functional Equivalence**：利用 LLM 自身判断两个状态是否"功能等价"
   ```
   Prompt: "Given two states S1 and S2, are they functionally equivalent
            for achieving the goal? Consider: same available actions,
            similar progress, equivalent information content."
   ```

2. **Embedding + Soft Assignment**：使用 Sentence Transformer 得到状态嵌入，计算相似度矩阵，允许一个状态以概率权重连接到多个图节点

#### A.2 边构建与权重

每一次动作执行形成一条边 $e = (s, a, s')$，边权重包含：
- 即时奖励 $R(s, a)$
- 访问频率 $n(s, a)$（用于估计转移概率）
- 置信度 $c(s, a) = \sqrt{n(s,a)}$（探索次数越多越可信）

#### A.3 图规模控制 (Graph Pruning)

为避免图爆炸，引入 **LRU-based State Pruning**：
- 维护最大节点数 $N_{max}$（如 10000）
- 定期清理低访问频率 + 低价值的节点
- 保留所有"关键路径"上的节点（成功轨迹经过的节点）

---

### B. 基于图的过程奖励传播 (Graph-based Reward Propagation)

这是替换原 `monte_carlo_sample.py` 逻辑的核心。

#### B.1 Model-Free Graph Value Propagation

**不再是简单的 Monte Carlo 平均值**。原 IPR 计算：
$$V_{MC}(s, a) = \frac{1}{N} \sum_{i=1}^{N} R_i^{terminal}$$

在 Graph-IPR 中，我们运行 **Incremental Q-Learning Style Update**（无需完整转移矩阵）：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot [R(s,a) + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a)]$$

其中：
- $\alpha$ 为学习率，可设为 $\frac{1}{n(s,a)}$ 实现渐进平均
- $\gamma \in [0.95, 0.99]$ 为折扣因子
- 转移 $s' \sim P(\cdot|s,a)$ 从图中已有边的经验分布采样

#### B.2 离线价值回溯 (Offline Value Iteration)

每一轮 `explore` 结束后，在图中运行 K 轮（如 K=10）的异步价值迭代：

```python
for _ in range(K):
    for s in shuffled(all_states):
        for a in available_actions(s):
            # 使用经验转移频率作为概率估计
            P_empirical = count(s, a, s') / count(s, a)
            Q[s][a] = R[s][a] + gamma * sum(P_empirical[s'] * max(Q[s']) for s' in successors)
```

使奖励信号从"终点"和"高分节点"反向渗透到整个图。

---

### C. 轨迹拼接对比学习 (Trajectory Stitching DPO)

在构建 `construct_preference.py` 数据时：

#### C.1 可达性检验 (Reachability Check)

**关键**：只拼接在图中**实际可达**的路径，避免 Distribution Shift：
- 从当前状态 $s$ 出发，只考虑图中有边相连的后继状态
- 拼接的路径必须是图中存在的连续边序列
- 引用 Offline RL 理论保证（类似 IQL 的 in-sample learning）

#### C.2 合成金标准 (Synthetic Gold Standard)

如果在图中发现一条路径 $\tau_{graph}$ 的累积奖励高于专家路径 $\tau_{expert}$：
- 将 $\tau_{graph}$ 中的动作作为 `chosen`
- 将 $\tau_{expert}$ 中对应位置的动作作为 `rejected`

#### C.3 跨轨迹对比 (Cross-Trajectory Comparison)

智能体可以对比"当前动作"与"图中已知最优动作"，而不仅仅是"专家动作"：

$$\text{chosen} = \arg\max_{a \in \{a_{expert}, a_{graph\_best}\}} Q(s, a)$$

这解决了专家路径次优的问题，实现 **Super-Expert Performance**。

---

## 3. 核心贡献 (Contributions)

1. **从链式监督到图式监督 (Chain → Graph Supervision)**
   - 提出全新的智能体训练范式，将零散的 Monte Carlo 采样转化为结构化的全局状态图
   - 显著提升采样利用率（Sample Efficiency），每条轨迹的每个片段都被复用

2. **隐式过程奖励模型 (Implicit Process Reward Model)**
   - Graph-IPR 通过图结构自动学习过程奖励，无需额外人工标注
   - 相比显式 PRM 更加 Scalable，与当前 PRM 研究热点形成互补

3. **理论保证的轨迹拼接 (Theoretically-Grounded Trajectory Stitching)**
   - 首次将 Offline RL 的 Trajectory Stitching 思想系统性引入 LLM Agent 训练
   - 通过可达性检验避免 Distribution Shift，保证拼接路径的可行性
   - 实现超越专家的 Super-Expert Performance

4. **低方差的 TD-driven 偏好优化 (TD-driven Preference Optimization)**
   - 将时序差分（TD）思想引入 DPO 数据构建
   - 相比 Monte Carlo 估计，TD 估计方差更小、收敛更快

5. **即插即用的迭代框架 (Plug-and-play Framework)**
   - 可直接嵌入现有 IPR 流程，只需替换采样和偏好构建模块
   - 在 WebShop、ALFWorld 等任务中验证有效性

---

## 4. 实验设计 (Experimental Design)

### 4.1 Baselines

| Method | Description |
|--------|-------------|
| Original IPR | 原始 Monte Carlo 采样 |
| IPR + Graph (no stitching) | 有图结构，但不做轨迹拼接 |
| IPR + Graph (MC value) | 有图+拼接，但用 MC 平均而非 TD |
| **Graph-IPR (Ours)** | 完整方案 |

### 4.2 Ablation Study

- **状态对齐方式**：Hard Clustering vs Soft Abstraction vs LLM-based
- **价值传播方式**：MC Average vs TD Update vs Full VI
- **图规模影响**：$N_{max}$ = 1k, 5k, 10k, 50k

### 4.3 实验中可以强调的"点" (Empirical Highlights)

1. **路径长度优化**：Graph-IPR 训练出的 Agent 完成任务的步数比专家更少（轨迹拼接的直接证据）

2. **冷启动鲁棒性**：即使初始 SFT 的专家数据质量很差（Suboptimal Experts），Graph-IPR 也能通过探索和图优化"自我修正"

3. **Sample Efficiency**：达到相同性能所需的采样次数对比

4. **可视化**：
   - 状态图随迭代从稀疏变稠密的演化过程
   - 最优路径如何从专家路径变为更优路径的对比图
   - 图中关键"桥梁"节点的发现过程

---

## 5. 潜在挑战与应对 (Potential Challenges)

| 挑战 | 应对策略 |
|------|----------|
| 状态聚类阈值敏感 | 采用 Soft Abstraction 或 LLM-based 判断，避免硬阈值 |
| 图规模爆炸 | LRU Pruning + 最大节点数限制 |
| 转移概率未知 | 使用经验频率估计，采用 Model-Free 更新 |
| 轨迹拼接的 Distribution Shift | 可达性检验 + In-sample Learning 保证 |
| 计算开销 | 报告实际图规模和内存占用，证明可接受 |

---

## 6. Related Work 定位

- **Iterative Preference Learning**: IPR, Self-Play, SPIN → 我们扩展为图结构
- **Process Reward Models**: Math-Shepherd, PRM800K → 我们是 Implicit PRM
- **Offline RL & Trajectory Stitching**: Decision Transformer, IQL → 首次应用于 LLM Agent
- **Graph-based RL**: Graph Neural Network for RL → 我们关注状态空间而非策略网络

---

## 7. 实现路线 (Implementation Roadmap)

### Phase 1: 基础框架
1. 修改 `construct_preference_monte_carlo_webshop.py`
2. 建立状态图数据结构：`{state_hash: {action: [(next_state, reward, count)]}}`
3. 实现状态 Embedding（使用 sentence-transformers）

### Phase 2: 核心算法
1. 实现 Soft State Abstraction
2. 实现 Graph Value Iteration / Q-Learning Update
3. 实现 Reachability-aware Trajectory Stitching

### Phase 3: 优化与实验
1. 添加 Graph Pruning 机制
2. 完整 Ablation Study
3. 可视化工具开发

---

## 8. 一句话总结 (One-liner)

> **Graph-IPR 将 LLM Agent 的探索经验从"用完即弃的轨迹"升级为"持续积累的知识图谱"，通过图上的价值传播和轨迹拼接，首次实现了无需额外标注的隐式过程奖励学习，并突破了专家性能的天花板。**
