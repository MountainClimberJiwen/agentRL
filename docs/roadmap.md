# agentRL 整合方案：Memory-T1 × Hermes Self-Evolution

> **约束**：不改 LLM 模型权重。所有进化发生在 **Prompt 空间** 和 **Memory 空间**。

---

## 一、两个项目的核心思想拆解

### Memory-T1（RL for Temporal Memory Selection）

- **问题**：多 session 对话中，历史记忆越长越 noisy，模型难以找到时间相关的证据
- **方法**：两阶段 Coarse-to-Fine
  1. **Candidate Generation**：时间过滤（预测 query 的时间窗口）+ 相关性过滤（retriever 排序）→ 候选集 C
  2. **Fine-grained Selection**：RL 智能体从 C 中选择 precise evidence sessions，并生成答案
- **奖励**：三级奖励函数
  - `R_acc`：答案准确性
  - `R_ground`：证据 grounding（是否引用了正确的 session）
  - `R_temp`：时间一致性（session-level 时间邻近性 + utterance-level 时间密度）
- **关键**：他们用 RL **微调了模型权重**（这是我们要避免的）

### Hermes Self-Evolution（GEPA: Genetic-Pareto Prompt Evolution）

- **问题**：Agent 的 skill、tool description、system prompt 需要持续优化
- **方法**：DSPy + GEPA 遗传进化
  1. 读取执行轨迹，理解**为什么失败**（不只是失败了）
  2. 生成候选 prompt/skill 变体
  3. 在 eval dataset 上评估（LLM-as-judge + 基准测试）
  4. 选择最优变体，通过 PR 部署
- **关键**：**完全不碰模型权重**，只变异和评估文本
- **约束门**：测试通过、字符限制、语义保持、缓存兼容

---

## 二、我们的整合思路：Prompt-as-Policy

Memory-T1 的 RL 微调了一个**模型**来做记忆选择。我们不碰模型，而是把**记忆选择策略**编码成 Prompt 模板，用 Hermes 的 GEPA 思路来进化这些模板。

核心洞察：**Prompt 本身就是一种可学习的策略**。

```
┌─────────────────────────────────────────────────────────────────┐
│                     Prompt-as-Policy Architecture               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Query                                                    │
│      │                                                          │
│      ▼                                                          │
│   ┌─────────────────────┐                                       │
│   │ Memory Router Prompt│  ← 学习：什么 query 该检索什么记忆     │
│   │ (learned template)  │                                       │
│   └──────────┬──────────┘                                       │
│              │ 输出：检索策略（时间窗口、项目过滤、关键词等）      │
│              ▼                                                  │
│   ┌─────────────────────┐                                       │
│   │   Candidate Pool    │  ← 从 memory_feedback DB 粗过滤       │
│   │   (Coarse Filter)   │                                       │
│   └──────────┬──────────┘                                       │
│              │                                                  │
│              ▼                                                  │
│   ┌─────────────────────┐                                       │
│   │ Evidence Selection  │  ← 学习：从候选中选择 precise evidence │
│   │    Prompt Template  │                                       │
│   │   (learned template)│                                       │
│   └──────────┬──────────┘                                       │
│              │ 输出：选中的 sessions + 组装好的 context           │
│              ▼                                                  │
│   ┌─────────────────────┐                                       │
│   │  System Prompt +    │                                       │
│   │  Retrieved Context  │  ──►  Frozen LLM                     │
│   │  + User Query       │                                       │
│   └─────────────────────┘                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**两个可学习的 Prompt 模板**：
1. **Memory Router Prompt**：给定 user query，输出检索参数（时间范围、项目、相关性阈值等）
2. **Evidence Selection Prompt**：给定候选 sessions，输出哪些应该被注入到最终 prompt 中

---

## 三、三级奖励函数（直接复用并扩展 Memory-T1）

我们已有 `rewards.py` 中的三级奖励，直接对应 Memory-T1 的设计：

| 奖励 | Memory-T1 对应 | 我们已有 | 扩展方向 |
|------|---------------|---------|---------|
| `R_acc` | Answer Accuracy | `accuracy_reward(outcome)` | 保持不变 |
| `R_ground` | Evidence Grounding | `grounding_reward(tool_calls, correction)` | 扩展为记忆引用准确性 |
| `R_temp` | Temporal Consistency | `temporal_reward(query, correction)` | 扩展为 session-level + utterance-level |

### 扩展后的 Temporal Consistency Reward（对标 Memory-T1）

```python
def temporal_consistency_reward(
    query: str,
    selected_sessions: list[dict],  # 被选中的记忆
    query_time_scope: tuple[datetime, datetime] | None,  # query 的时间范围
) -> float:
    """
    Memory-T1 风格的时间一致性奖励：
    - Session-level：选中的 session 时间是否在 query 时间窗口附近
    - Utterance-level：选中的 utterance 内的时间标记是否与 query 对齐
    """
    if not query_time_scope or not selected_sessions:
        return 0.0

    t_start, t_end = query_time_scope
    scores = []

    for sess in selected_sessions:
        sess_time = sess.get("timestamp")
        if not sess_time:
            continue

        # Session-level: 时间邻近性
        # 越接近 query 时间窗口，分数越高
        if t_start <= sess_time <= t_end:
            scores.append(1.0)
        else:
            # 指数衰减
            gap = min(abs(sess_time - t_start), abs(sess_time - t_end))
            scores.append(max(0, 1.0 - gap.total_seconds() / 86400))  # 按天衰减

    return sum(scores) / len(scores) if scores else 0.0
```

---

## 四、进化引擎：GEPA-style Prompt Evolution

### 4.1 可进化的 Prompt 资产

| 资产 | 当前位置 | 进化目标 | 评估方式 |
|------|---------|---------|---------|
| **Memory Router Prompt** | `src/agentrl/prompts/router.txt` (新建) | 让 LLM 输出更准确的检索参数 | 检索后 session 的 reward |
| **Evidence Selection Prompt** | `src/agentrl/prompts/selector.txt` (新建) | 让 LLM 从候选中选择更相关的证据 | 最终 answer 的 reward |
| **System Prompt 模板** | `src/agentrl/prompts/system.txt` (新建) | 整体行为指导 | 综合 reward |
| **Self-Reflection Prompt** | `src/agentrl/prompts/reflect.txt` (新建) | 让 agent 在输出前自检 | correction rate |
| **Tool Description 偏见** | `src/agentrl/prompts/tool_bias.txt` (新建) | 影响工具选择倾向 | tool call 后的 reward |

### 4.2 GEPA 进化循环（Hermes 风格，完全不改权重）

```
┌──────────────────────────────────────────────────────────────┐
│                    Prompt Evolution Loop                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. SELECT TARGET                                            │
│     选一个 prompt 资产（如 Memory Router Prompt）            │
│                                                              │
│  2. BUILD EVAL DATASET                                       │
│     从 memory_feedback DB 中采样：                           │
│     - 高 reward sessions（approved）→ 正例                   │
│     - 低 reward sessions（exited/corrected）→ 负例           │
│     - 分成 train / val / holdout                             │
│                                                              │
│  3. GENERATE CANDIDATES                                      │
│     用 LLM 生成 prompt 变体（遗传变异）：                    │
│     - Mutation：修改某一句措辞                               │
│     - Crossover：混合两个好 prompt 的部分                    │
│     - Reflection：让 LLM 分析失败案例，提出改进              │
│                                                              │
│  4. EVALUATE                                                 │
│     对每个变体，在 val set 上回测：                          │
│     - 用该 prompt 重新组装 context → 送给 frozen LLM         │
│     - 用 reward 函数打分                                     │
│     - 不用真的运行 agent，直接用历史数据中的 outcome         │
│                                                              │
│  5. SELECT & DEPLOY                                          │
│     选 reward 最高的变体                                     │
│     保存为新的 prompt 版本                                   │
│     在 holdout set 上验证没有 regression                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 关键设计：为什么不用真的运行 Agent？

因为我们有 **memory_feedback DB**，里面已经有：
- `user_input`（query）
- `assistant_response`（answer）
- `user_outcome`（最终 outcome）
- `computed_reward_*`（预计算的 reward）

所以对于 Prompt 变体的评估，我们只需要：
1. 用新 prompt 重新决定**检索什么记忆**
2. 比较检索结果与已知的高/低 reward sessions 的匹配度
3. 如果新 prompt 选中了更多 approved sessions 的证据，reward 就高

这是 **offline RL** 的思路，不需要与环境交互。

---

## 五、Memory-T1 Coarse-to-Fine 的 Prompt 实现

### Phase 1: Coarse Filtering（非学习部分，规则/启发式）

```python
def coarse_filter(query: str, db_conn: sqlite3.Connection) -> list[dict]:
    """
    粗过滤：快速缩小候选集，不做复杂决策
    """
    # 1. 时间过滤：检测 query 中的时间关键词
    time_scope = extract_time_scope(query)  # 已有 has_temporal_keywords

    # 2. 项目过滤：当前 working directory / project
    project_hint = extract_project_hint(query)

    # 3. 相关性过滤：简单的关键词匹配 / embedding 相似度
    # 可以用轻量级的 text similarity，不需要 LLM

    sql = """
    SELECT * FROM memory_feedback
    WHERE (%s OR created_at > datetime('now', '-7 days'))
      AND (%s OR session_id LIKE ?)
    ORDER BY computed_reward_accuracy DESC
    LIMIT 50
    """
    # 返回候选集 C
```

### Phase 2: Fine-grained Selection（Prompt-as-Policy）

```
【Evidence Selection Prompt Template】(可进化)

You are a memory selection agent. Given a user query and a list of candidate
past sessions, select the most relevant ones to include in the context.

User Query: {query}
Query Time Scope: {time_scope}
Current Project: {project}

Candidate Sessions:
{formatted_candidates}

Instructions:
- Prioritize sessions from the same project
- Prioritize sessions with outcome="approved" 
- For temporal queries, prioritize sessions close to the query time scope
- Avoid sessions with outcome="exited" unless strongly relevant
- Return at most {max_sessions} session IDs, ranked by relevance

Output format: JSON list of {"session_id": "...", "reason": "..."}
```

这个 prompt 模板就是**策略**。我们通过 GEPA 进化它来优化选择行为。

---

## 六、整体 Self-Evolution 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        agentRL Self-Evolution                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐      Extract       ┌──────────────┐               │
│  │   Sessions  │ ─────────────────► │ UnifiedTurns │               │
│  │  (4 backends)                    │  + Rewards   │               │
│  └─────────────┘                    └──────┬───────┘               │
│                                            │                        │
│                                            ▼                        │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              SQLite Experience Buffer                   │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │       │
│  │  │memory_feedback│  │prompt_registry│  │eval_results    │  │       │
│  │  │(session data) │  │(prompt versions)│  │(offline eval)  │  │       │
│  │  └─────────────┘  └─────────────┘  └────────────────┘  │       │
│  └─────────────────────────────────────────────────────────┘       │
│                           ▲                    │                    │
│                           │                    │                    │
│        ┌──────────────────┘                    │                    │
│        │                                       ▼                    │
│        │  ┌─────────────────────────────────────────────┐          │
│        │  │         Prompt Evolution Engine             │          │
│        │  │  ┌─────────────┐      ┌───────────────┐     │          │
│        │  │  │   GEPA      │◄────►│  LLM Judge    │     │          │
│        │  │  │  Mutator    │      │  (scorer)     │     │          │
│        │  │  └──────┬──────┘      └───────────────┘     │          │
│        │  │         │                                    │          │
│        │  │         ▼                                    │          │
│        │  │  ┌─────────────────────────────────────┐    │          │
│        │  │  │      Prompt Version Registry        │    │          │
│        │  │  │  router_v1, router_v2, ...          │    │          │
│        │  │  │  selector_v1, selector_v2, ...      │    │          │
│        │  │  └─────────────────────────────────────┘    │          │
│        │  └─────────────────────────────────────────────┘          │
│        │                                                           │
│        └────────────── 部署最优 prompt ────────────────────────────┘
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Runtime Agent                             │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │   │
│  │  │User Query   │───►│Memory Router│───►│Coarse Filter│     │   │
│  │  │             │    │  Prompt     │    │  (rules)    │     │   │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘     │   │
│  │                                                │             │   │
│  │  ┌─────────────┐    ┌─────────────┐           │             │   │
│  │  │  Frozen LLM │◄───│ Final Prompt│◄──────────┘             │   │
│  │  │             │    │Assembler    │                          │   │
│  │  └─────────────┘    └─────────────┘                          │   │
│  │                                 ▲                             │   │
│  │  ┌─────────────────────────────┘                             │   │
│  │  │ Evidence Selection Prompt + Retrieved Context             │   │
│  │  └───────────────────────────────────────────────────────────┘   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 七、实施路线图

### Phase 1: 基础设施（1-2 周）

1. **Prompt 模板系统**
   - 创建 `src/agentrl/prompts/` 目录
   - 实现 `PromptRegistry`：加载、版本管理、A/B 测试切换
   - 实现 `PromptAssembler`：组装 system prompt + memory context + user query

2. **Memory Router + Evidence Selector**
   - 实现 Coarse Filter（规则 + 轻量相似度）
   - 实现 Memory Router Prompt（让 LLM 输出检索参数）
   - 实现 Evidence Selection Prompt（让 LLM 选择证据 sessions）
   - **初始 prompt 用手写，不进化**

3. **扩展 Reward 函数**
   - 在 `rewards.py` 中增加 `temporal_consistency_reward`（Memory-T1 风格）
   - 增加 `memory_grounding_reward`（检查是否引用了正确的历史 session）

### Phase 2: Offline Evaluation（1 周）

1. **回测框架**
   - 给定一个 prompt 版本，在历史 session 上"重放"记忆选择
   - 对比选中的 sessions 与已知的 reward，计算 aggregate score
   - 不用真的调用 LLM 生成 answer，只评估记忆选择质量

2. **建立 holdout set**
   - 从 DB 中划出 20% 的 sessions 作为测试集
   - 确保各类 outcome 均衡

### Phase 3: GEPA Prompt Evolution（2-3 周）

1. **Prompt 变异器**
   - 用 LLM 生成 prompt 变体（Mutation / Crossover / Reflection）
   - 约束：字符数不增加超过 20%、语义保持、保留关键指令

2. **评估循环**
   - 对每个变体，在 val set 上运行回测
   - 用 aggregate reward 作为 fitness
   - 保留 Pareto 前沿（reward vs prompt length）

3. **部署**
   - 在 holdout set 上验证无 regression
   - 保存为新版本，支持回滚

### Phase 4: 闭环 Self-Evolution（持续）

1. **自动触发**
   - 当某个 prompt 版本的平均 reward 连续下降时，触发进化
   - 每周自动跑一次完整进化流程

2. **数据飞轮**
   - 新 session → extract → reward → DB
   - DB 增长 → eval dataset 更 rich → 进化效果更好

---

## 八、与现有代码的对接

| 新组件 | 对接现有代码 |
|--------|------------|
| `prompts/` 模板 | 新增目录，不影响现有代码 |
| `PromptRegistry` | 新增模块，可选使用 |
| 扩展 `rewards.py` | 在现有函数基础上增加 |
| `evolution/` GEPA 引擎 | 新增包，离线运行 |
| `eval/` 回测框架 | 读 `memory_feedback` DB，不改动 schema |

---

## 九、核心原则（再次强调）

1. **Frozen LLM**：LLM 是黑盒 API，绝不调用 `train()`、`fit()` 或任何权重更新
2. **Prompt is Policy**：策略 = 文本模板，学习 = 文本进化
3. **Memory is Part of State**：检索什么、如何组装，都是可优化的策略参数
4. **Offline RL**：用已有 session 数据做回测，不需要在线交互
5. **Pareto Selection**：不只看 reward，还要考虑 prompt 长度、延迟、可读性

---

> **总结**：Memory-T1 告诉我们「时间感知记忆选择」很重要，Hermes 告诉我们「Prompt 遗传进化」很有效。我们把两者结合：**用 GEPA 进化 Prompt 模板来实现 Memory-T1 的记忆选择策略**，奖励信号来自真实用户反馈，完全不碰模型权重。
